"""
CardioAI Pro - cost avoidance with a CONTROL group
==================================================
reports/cost_avoidance.py compares a member's cost trend before and after being flagged, with no control
group; its docstring says why that can't show the platform caused a change (regression to the mean,
natural care-seeking swings). This module is the controlled version: matched difference-in-differences.

  1. Propensity model (logistic regression) for being flagged / reached, from pre-period covariates INCLUDING
     pre-period cost, so members matched on it start from the same cost level (regression to the mean then
     hits both groups alike).
  2. Each treated member is matched to the nearest control(s) on the propensity logit within a caliper.
  3. Balance check: standardized mean difference per covariate after matching.
  4. Difference-in-differences on per-member-per-month (PMPM) cost:
        effect = mean(post - pre | treated) - mean(post - pre | matched controls)
     negative = costs grew less (or fell more) than comparable members' did.
  5. 95% interval from a bootstrap over treated members (their matched controls travel with them).

It only reports a number when it can defend it ('reportable'): enough matched members, good balance, and enough
overlap. Cost inputs should be PAID amounts, not billed charges (charge amounts overstate cost and inflate noise).

DESIGN MATTERS MORE THAN THE MODEL. Observational matching only corrects for what is measured. If members were flagged
by a risk-score threshold, flagged and unflagged members barely overlap and no honest control exists; the
overlap check will then refuse to report. The strongest control is a randomized holdout (design='randomized'):
a random share of flagged members who are not contacted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class MemberPeriods:
    member_id: str
    treated: bool                      # flagged AND reached (or randomized into outreach)
    covariates: dict                   # pre-period only: age, risk score, comorbidity count, ...
    pre_pmpm: float                    # paid cost per member-month before the anchor date
    post_pmpm: float                   # ... after it
    post_months: float = 12.0


def _smd(a: np.ndarray, b: np.ndarray, wb: Optional[np.ndarray] = None) -> float:
    wb = np.ones(len(b)) if wb is None else wb
    mb = np.average(b, weights=wb)
    sd = np.sqrt((a.var(ddof=1) + np.average((b - mb) ** 2, weights=wb)) / 2) if len(a) > 1 else 0.0
    return float(abs(a.mean() - mb) / sd) if sd > 0 else 0.0


def estimate_cost_avoidance(members: list[MemberPeriods], design: str = "observational", k: int = 1, caliper_sd: float = 0.2,
                            min_matched: int = 30, max_smd: float = 0.10, max_unmatched_share: float = 0.20,
                            n_boot: int = 1000, seed: int = 0) -> dict:
    """design: 'observational' (propensity matching) or 'randomized' (holdout: plain DiD, balance still checked)."""
    if design not in ("observational", "randomized"):
        raise ValueError("design must be 'observational' or 'randomized'")
    T = [m for m in members if m.treated]; C = [m for m in members if not m.treated]
    out = {"design": design, "n_treated": len(T), "n_control_pool": len(C), "reportable": False}
    if len(T) < min_matched or len(C) < min_matched:
        out["reason"] = f"need at least {min_matched} treated and {min_matched} control members; have {len(T)} and {len(C)}"
        return out
    names = sorted(T[0].covariates) + ["pre_pmpm"]
    X = lambda ms: np.array([[m.covariates[n] for n in names[:-1]] + [m.pre_pmpm] for m in ms], dtype=float)
    Xt, Xc = X(T), X(C)
    dt = np.array([m.post_pmpm - m.pre_pmpm for m in T]); dc = np.array([m.post_pmpm - m.pre_pmpm for m in C])
    if design == "randomized":
        match = [list(range(len(C))) for _ in T]; w = np.ones(len(C))         # everyone in the holdout is the control
        ctrl_effect = [dc.mean()] * len(T)
        unmatched = 0
    else:
        sc = StandardScaler().fit(np.vstack([Xt, Xc]))
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(np.vstack([Xt, Xc])), [1] * len(T) + [0] * len(C))
        lt, lc = lr.decision_function(sc.transform(Xt)), lr.decision_function(sc.transform(Xc))
        cal = caliper_sd * np.std(np.concatenate([lt, lc]))
        match, unmatched = [], 0
        for i, l in enumerate(lt):
            d = np.abs(lc - l); idx = np.argsort(d)[:k]
            idx = [j for j in idx if d[j] <= cal]
            if idx:
                match.append(idx)
            else:
                match.append([]); unmatched += 1
        kept = [i for i, m in enumerate(match) if m]
        T_idx, match = kept, [match[i] for i in kept]
        used = np.zeros(len(C)); 
        for m in match:
            for j in m:
                used[j] += 1 / len(m)
        w = used if used.sum() else np.ones(len(C))
        ctrl_effect = [dc[m].mean() for m in match]
        Xt, dt = Xt[T_idx], dt[T_idx]
    n_m = len(dt)
    out.update(n_matched_treated=n_m, unmatched_treated=unmatched, unmatched_share=round(unmatched / len(T), 3))
    if n_m < min_matched:
        out["reason"] = f"only {n_m} treated members found a comparable control (caliper {caliper_sd} SD); no defensible comparison"
        return out
    smd = {n: round(_smd(Xt[:, i], Xc[:, i], w if design == "observational" else None), 3) for i, n in enumerate(names)}
    out["balance_smd"] = smd
    eff = dt - np.array(ctrl_effect)
    est = float(eff.mean())
    rng = np.random.RandomState(seed)
    boots = [eff[rng.randint(0, n_m, n_m)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out.update(effect_pmpm=round(est, 2), ci95_pmpm=[round(float(lo), 2), round(float(hi), 2)],
               mean_pre_pmpm_treated=round(float(Xt[:, -1].mean()), 2))
    problems = []
    if max(smd.values()) > max_smd:
        problems.append(f"covariate imbalance after matching (max SMD {max(smd.values()):.2f} > {max_smd})")
    if design == "observational" and unmatched / len(T) > max_unmatched_share:
        problems.append(f"poor overlap: {unmatched / len(T):.0%} of treated members had no comparable control; likely flagged by a threshold, so use a randomized holdout")
    out["reportable"] = not problems
    out["problems"] = problems
    if out["reportable"]:
        total = -est * float(np.mean([m.post_months for m in T]) * n_m)
        out["avoided_cost_total"] = round(total, 0)   # among matched treated members; positive = avoided
        out["narrative"] = (f"Among {n_m} matched members, post-outreach cost per member-month changed by {est:+.2f} relative to comparable "
                            f"members (95% interval {lo:+.2f} to {hi:+.2f}). Negative means cost grew less than for comparable members.")
    else:
        out["narrative"] = "Not reported: " + "; ".join(problems)
    return out
