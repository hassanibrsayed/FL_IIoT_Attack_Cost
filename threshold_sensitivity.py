"""Threshold sensitivity of the adaptive attack cost, recomputed from the existing per-run results (no new training).

For every seed / partition / aggregator the attacker's best result at each m (min over strategies) is compared with the
clean (m = 0) model under a range of thresholds:
  * macro-F1 drop  d in DROPS   (paper uses 0.10)
  * detection: Edge-IIoTset attack recall < floor (paper 0.8, varied in FLOORS with d fixed at 0.10);
               CIC-IoT2023 balanced detection dropping by more than d (same d as macro-F1).
Runs were stopped early once BOTH criteria were broken at the paper's thresholds, so for looser thresholds a seed may
not be broken by the last m that was run. Such seeds are reported as censored at that m ("unresolved") and counted
separately; they are not silently treated as ">5".
Outputs: analysis/threshold_sensitivity/{cost_by_threshold.csv, paired_by_threshold.csv, unresolved.csv}
"""
import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "analysis/threshold_sensitivity"
os.makedirs(OUT, exist_ok=True)
M, CENS = 5, 6
DROPS = [0.05, 0.10, 0.15, 0.20]
FLOORS = [0.7, 0.8, 0.9]
PARTS = ["iid", "dir0.5", "dir0.1"]
AGGS = ["fedavg", "median", "trimmed"]


def load(path, crit):
    r = pd.read_csv(path)
    r["det"] = (r.attack_recall + 1 - r.false_alarm) / 2 if crit == "bd" else r.attack_recall
    if crit == "bd" and "base_bd" not in r:
        raise SystemExit("base_bd missing in " + path)
    b = (r.groupby(["seed", "partition", "aggregator", "malicious"])
          .agg(f1=("macro_f1", "min"), det=("det", "min"), base_f1=("base_f1", "first"),
               base_bd=("base_bd", "first") if crit == "bd" else ("base_f1", "first")).reset_index())
    return b


def costs(best, crit, drop, floor):
    rows = []
    for (p, a, s), g in best.groupby(["partition", "aggregator", "seed"]):
        g = g.sort_values("malicious")
        det_bad = g.det < (g.base_bd - drop if crit == "bd" else floor)
        broke = det_bad | (g.f1 < g.base_f1 - drop)
        last = int(g.malicious.max())
        if broke.any():
            c, unresolved = int(g.malicious[broke].min()), False
        else:
            c, unresolved = CENS, last < M      # stopped early (at paper thresholds) but not broken at these
        rows.append(dict(partition=p, aggregator=a, seed=s, cost=c, unresolved=unresolved, last_m=last))
    return pd.DataFrame(rows)


def fmt(v):
    return ">5" if v >= CENS else f"{v:g}"


def sign_p(k1, k2):
    n = k1 + k2
    if n == 0:
        return 1.0
    k = min(k1, k2)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


cells, pairs, unres = [], [], []
for ds, path, crit in [("Edge-IIoTset", "adaptive_out/adaptive_results.csv", "recall"),
                       ("CIC-IoT2023", "cic_adaptive/adaptive_results.csv", "bd")]:
    best = load(path, crit)
    settings = [(d, 0.8) for d in DROPS]
    if crit == "recall":
        settings += [(0.10, f) for f in FLOORS if f != 0.8]
    for drop, floor in settings:
        label = f"drop {drop:.2f}" + (f", recall floor {floor:.1f}" if crit == "recall" else "")
        c = costs(best, crit, drop, floor)
        for p in PARTS:
            for a in AGGS:
                v = c[(c.partition == p) & (c.aggregator == a)]
                x = v.cost.to_numpy(float)
                cells.append(dict(dataset=ds, setting=label, drop=drop, floor=floor if crit == "recall" else np.nan,
                                  partition=p, aggregator=a, n=len(v),
                                  cost=f"{fmt(np.median(x))} [{fmt(x.min())}..{fmt(x.max())}]",
                                  median=np.median(x), n_unresolved=int(v.unresolved.sum())))
        for a in ["median", "trimmed"]:
            A = c[(c.partition == "iid") & (c.aggregator == a)].set_index("seed")
            for q in ["dir0.5", "dir0.1"]:
                B = c[(c.partition == q) & (c.aggregator == a)].set_index("seed")
                idx = A.index.intersection(B.index)
                d = (A.cost[idx] - B.cost[idx]).to_numpy(float)
                nu = int(A.unresolved[idx].sum() + B.unresolved[idx].sum())
                pairs.append(dict(dataset=ds, setting=label, drop=drop, aggregator=a, comparison=f"iid vs {q}",
                                  n=len(d), mean_diff=round(d.mean(), 2), cheaper=int((d > 0).sum()),
                                  equal=int((d == 0).sum()), costlier=int((d < 0).sum()),
                                  p=sign_p(int((d > 0).sum()), int((d < 0).sum())), n_unresolved=nu))
        u = c[c.unresolved]
        if len(u):
            u = u.assign(dataset=ds, setting=label)
            unres.append(u)

pd.DataFrame(cells).to_csv(f"{OUT}/cost_by_threshold.csv", index=False)
pd.DataFrame(pairs).to_csv(f"{OUT}/paired_by_threshold.csv", index=False)
(pd.concat(unres) if unres else pd.DataFrame()).to_csv(f"{OUT}/unresolved.csv", index=False)
print(pd.DataFrame(cells).pivot_table(index=["dataset", "partition", "aggregator"], columns="setting",
                                      values="cost", aggfunc="first").to_string())
print()
print(pd.DataFrame(pairs)[["dataset", "setting", "aggregator", "comparison", "n", "mean_diff", "cheaper", "equal",
                           "costlier", "p", "n_unresolved"]].to_string(index=False))
print("\nunresolved seed-cells:", sum(len(u) for u in unres))
