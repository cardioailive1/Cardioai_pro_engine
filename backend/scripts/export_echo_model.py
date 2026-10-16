"""
Export MobileEchoNet for a device and measure its speed.

    python -m scripts.export_echo_model --checkpoint weights/echo_lvef_v1.pt --out export/ [--variant small]
    python -m scripts.export_echo_model --random-init --out export/       (pipeline / speed test only)

Writes TorchScript (<out>/echo_lvef.ts.pt) and, if the onnx package is installed, ONNX (<out>/echo_lvef.onnx).
Also prints parameter count, file sizes and median CPU latency per 8-frame loop. Latency here is on THIS machine's CPU;
it is not a phone or edge-device number, so re-run on the target. NOT done: int8 quantization. This network is mostly
convolutions, so it needs static quantization calibrated on real echo frames on the target runtime (dynamic quantization
does nothing useful for it and breaks the temporal layer).
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from imaging_models.echo_checkpoint import load_echo_checkpoint
from imaging_models.mobilenet_echo_model import MobileEchoNet


def _latency(model, x, runs=20):
    with torch.no_grad():
        for _ in range(3):
            model(x)
        ts = []
        for _ in range(runs):
            t = time.perf_counter(); model(x); ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint"); ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--variant", default="small"); ap.add_argument("--out", default="export")
    ap.add_argument("--n-frames", type=int, default=8); ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args(argv)
    if a.random_init:
        model, n_frames, img = MobileEchoNet(n_outputs=1, variant=a.variant).eval(), a.n_frames, a.img_size
        print("WARNING: random weights. Speed and format test only.")
    else:
        loaded = load_echo_checkpoint(a.checkpoint)
        if not loaded:
            raise SystemExit("checkpoint missing or not approved")
        model, meta = loaded; n_frames, img = meta["n_frames"], meta["img_size"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, n_frames, 1, img, img)
    result = {"parameters": model.param_count(), "float_ms_cpu": _latency(model, x)}
    ts = torch.jit.trace(model, x, check_trace=False); ts.save(str(out / "echo_lvef.ts.pt"))
    result["torchscript_mb"] = round((out / "echo_lvef.ts.pt").stat().st_size / 1e6, 1)
    try:
        torch.onnx.export(model, x, str(out / "echo_lvef.onnx"), input_names=["cine"], output_names=["lvef_fraction"], opset_version=17)
        result["onnx_mb"] = round((out / "echo_lvef.onnx").stat().st_size / 1e6, 1)
    except Exception as e:     # onnx package optional
        result["onnx"] = f"skipped ({type(e).__name__})"
    print(result)
    return result


if __name__ == "__main__":
    main()
