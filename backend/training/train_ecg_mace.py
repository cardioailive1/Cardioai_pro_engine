"""
CardioAI Pro - train the ECG -> MACE network
============================================
Run:  python -m training.train_ecg_mace          (synthetic smoke test)

Same task and metrics as training/train_model.py (30-90 day MACE, patient-level
split, AUC / AP / Brier, sensitivity-specificity-PPV at fixed and matched-80%
sensitivity), so results are directly comparable to the feature-based model.
On Mayo data, call train_ecg_mace_net(examples) with the output of
training.mayo.ecg_mace_dataset.build_examples.

The synthetic run only proves the pipeline works. It says nothing about
clinical performance, and inference/models.py must keep returning None
until this is trained on real, adjudicated outcomes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ecg_models.ecg_mace_net import ECGMACENet
from training.train_model import evaluate, evaluate_at_matched_sensitivity


def split_by_patient(examples, train_frac=0.7, val_frac=0.15, seed=23):
    rng = np.random.RandomState(seed)
    ids = sorted({e["patient_id"] for e in examples}); rng.shuffle(ids)
    n = len(ids)
    tr, va = set(ids[:int(n * train_frac)]), set(ids[int(n * train_frac):int(n * (train_frac + val_frac))])
    pick = lambda s: [e for e in examples if e["patient_id"] in s]
    te = set(ids) - tr - va
    return pick(tr), pick(va), pick(te)


def _tensors(exs, device):
    x = torch.tensor(np.stack([e["x"] for e in exs]), device=device)
    c = torch.tensor(np.stack([e["context"] for e in exs]), device=device)
    y = torch.tensor([e["y"] for e in exs], dtype=torch.float32, device=device)
    return x, c, y


@torch.no_grad()
def _logits(model, exs, device, use_context=False, bs=64):
    model.eval(); out = []
    for i in range(0, len(exs), bs):
        x, c, _ = _tensors(exs[i:i + bs], device)
        out.append(model(x, c if use_context else None).cpu().numpy())
    return np.concatenate(out)


def _scores(model, exs, device, bs=64, use_context=False):
    return 1 / (1 + np.exp(-_logits(model, exs, device, use_context, bs)))


def wearable_augment(x: torch.Tensor) -> torch.Tensor:
    """Roughen clean clinical traces toward wearable quality: baseline wander,
    broadband noise, random gain. A PROXY only - it does not replace training or
    validating on real wearable ECG."""
    B, L, T = x.shape
    t = torch.linspace(0, 1, T, device=x.device)
    wander = 0.3 * torch.sin(2 * np.pi * (0.1 + 0.3 * torch.rand(B, 1, 1, device=x.device)) * 10 * t
                             + 6.28 * torch.rand(B, 1, 1, device=x.device))
    noise = (0.05 + 0.25 * torch.rand(B, 1, 1, device=x.device)) * torch.randn_like(x)
    return x * (0.7 + 0.6 * torch.rand(B, 1, 1, device=x.device)) + wander + noise


def train_ecg_mace_net(examples, n_epochs=15, batch_size=32, lr=1e-3, use_context=False, seed=23, device=None, wearable_aug=False):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = split_by_patient(examples, seed=seed)
    n_leads = examples[0]["x"].shape[0]
    model = ECGMACENet(n_leads=n_leads, n_context=2 if use_context else 0).to(device)
    y_tr = np.array([e["y"] for e in train])
    pos_weight = torch.tensor((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), device=device)  # rare positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state = -1.0, None
    from sklearn.metrics import roc_auc_score
    for ep in range(n_epochs):
        model.train(); perm = np.random.RandomState(seed + ep).permutation(len(train))
        for i in range(0, len(perm), batch_size):
            x, c, y = _tensors([train[j] for j in perm[i:i + batch_size]], device)
            if wearable_aug:
                x = wearable_augment(x)
            opt.zero_grad(); loss = loss_fn(model(x, c if use_context else None), y); loss.backward(); opt.step()
        yv = np.array([e["y"] for e in val])
        auc = roc_auc_score(yv, _scores(model, val, device, use_context=use_context)) if 0 < yv.sum() < len(yv) else float("nan")
        print(f"epoch {ep + 1:2d}  last-batch loss={loss.item():.4f}  val AUC={auc:.3f}")
        if auc == auc and auc > best:
            best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)   # model selection on VAL only; test untouched
    y_te, s_te = np.array([e["y"] for e in test]), _scores(model, test, device, use_context=use_context)
    metrics = evaluate("ECG-MACE network", y_te, s_te)
    matched = evaluate_at_matched_sensitivity("ECG-MACE network (matched)", y_te, s_te)
    from ecg_models.compare import data_sufficiency, fit_platt
    # Calibrate on VAL logits (the weighted loss distorts raw probabilities).
    v_logit = _logits(model, val, device, use_context)
    platt = fit_platt(v_logit, np.array([e["y"] for e in val]))
    cal_te = 1 / (1 + np.exp(-(platt["a"] * _logits(model, test, device, use_context) + platt["b"])))
    return model, {"test": metrics, "matched": matched, "n_test": len(test), "test_scores": s_te, "test_y": y_te,
                   "test_scores_calibrated": cal_te, "platt": platt,
                   "sufficiency": data_sufficiency(int(y_tr.sum()), model.param_count()),
                   "n_leads": n_leads, "use_context": use_context}


def synthetic_examples(n=600, length=1000, n_leads=12, seed=5):
    """Fake ECG-like traces; positives get a subtle extra component. PIPELINE TEST ONLY."""
    rng = np.random.RandomState(seed); t = np.linspace(0, 10, length)
    out = []
    for i in range(n):
        y = int(rng.rand() < 0.2); hr = rng.uniform(55, 100) / 60
        base = np.sin(2 * np.pi * hr * t) ** 15 * 3
        sig = np.stack([base * rng.uniform(.8, 1.2) + 0.3 * rng.randn(length) + y * 0.4 * np.sin(2 * np.pi * 7 * t) for _ in range(n_leads)])
        sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-6)
        out.append({"patient_id": f"p{i // 2}", "x": sig.astype(np.float32), "y": y,
                    "context": np.array([rng.uniform(.5, 1.4), float(rng.rand() < .5)], dtype=np.float32)})
    return out


if __name__ == "__main__":
    ex = synthetic_examples()
    print(f"SYNTHETIC data: {len(ex)} traces, {sum(e['y'] for e in ex)} positives")
    train_ecg_mace_net(ex)
