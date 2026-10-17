import numpy as np

from reports.cost_avoidance_control import MemberPeriods, estimate_cost_avoidance


def _cohort(true_effect=-80.0, n=1500, seed=1, deterministic=False, randomized=False):
    rng = np.random.RandomState(seed); ms = []
    for i in range(n):
        level = rng.gamma(3, 200)                      # persistent member cost level
        risk = level / 600 + rng.randn() * .3
        pre = level + rng.randn() * 150                # noisy pre-period
        if randomized:
            treated = rng.rand() < .5
        elif deterministic:
            treated = pre > 700
        else:
            treated = rng.rand() < 1 / (1 + np.exp(-(pre - 700) / 150))      # flagged more often when pre cost is high
        post = level + rng.randn() * 150 + (true_effect if treated else 0)    # regression to the mean for everyone
        ms.append(MemberPeriods(f"m{i}", bool(treated), {"age": 50 + rng.randint(30), "risk": risk}, max(pre, 0), max(post, 0)))
    return ms


def test_matching_recovers_effect_while_naive_pre_post_is_biased():
    ms = _cohort(true_effect=-80)
    naive = np.mean([m.post_pmpm - m.pre_pmpm for m in ms if m.treated])
    r = estimate_cost_avoidance(ms, k=2)
    assert r["reportable"] and abs(naive - (-80)) > 25                       # naive overstates 'savings' via regression to the mean
    assert r["ci95_pmpm"][0] <= -80 <= r["ci95_pmpm"][1] or abs(r["effect_pmpm"] + 80) < 40
    assert abs(r["effect_pmpm"] - (-80)) < abs(naive - (-80))                # controlled estimate is closer to the truth
    assert max(r["balance_smd"].values()) <= 0.10


def test_refuses_when_flagged_by_threshold_no_overlap():
    r = estimate_cost_avoidance(_cohort(deterministic=True))
    assert r["reportable"] is False


def test_randomized_holdout_design():
    r = estimate_cost_avoidance(_cohort(true_effect=-80, randomized=True), design="randomized")
    assert r["reportable"] and r["ci95_pmpm"][0] < -80 < r["ci95_pmpm"][1]


def test_too_few_members():
    r = estimate_cost_avoidance(_cohort(n=40))
    assert r["reportable"] is False and "need at least" in r.get("reason", "")
