"""Mechanism check: does the spread of honest updates explain the lower attack cost under non-IID data?

Joins the clean-run dispersion logs (disp_edge/, disp_cic/, from adaptive_attack.py --dispersion) with the per-seed
attack costs of the adaptive study (adaptive_out/, cic_adaptive/) on seed, partition and aggregator, and reports
  * dispersion per partition (median [min..max] over seeds),
  * the Spearman rank correlation between dispersion and attack cost over all seed x partition cells, with a
    permutation p-value (10,000 permutations),
  * the same correlation within each partition (does more spread mean a cheaper attack even at fixed alpha?).
Outputs: analysis/mechanism/{dispersion_by_partition.csv, correlation.csv}
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
OUT = "analysis/mechanism"
os.makedirs(OUT, exist_ok=True)
METRICS = ["rel_spread_last10", "pair_cos_last10", "coord_snr_last10"]
PARTS = ["iid", "dir0.5", "dir0.1"]
rng = np.random.default_rng(0)


def spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def perm_p(x, y, n=10000):
    r0 = spearman(x, y)
    if np.isnan(r0):
        return r0, float("nan")
    ry = pd.Series(y).rank().to_numpy()
    rx = pd.Series(x).rank().to_numpy()
    hits = sum(abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(r0) - 1e-12 for _ in range(n))
    return r0, (hits + 1) / (n + 1)


by_part, corr = [], []
for ds, dpath, cpath in [("Edge-IIoTset", "disp_edge/dispersion.csv", "adaptive_out/adaptive_cost_per_seed.csv"),
                         ("CIC-IoT2023", "disp_cic/dispersion.csv", "cic_adaptive/adaptive_cost_per_seed.csv")]:
    if not os.path.exists(dpath):
        print("missing", dpath, "- run run_revision.ps1 step 1 first")
        continue
    d = pd.read_csv(dpath)
    c = pd.read_csv(cpath)
    j = d.merge(c, on=["seed", "partition", "aggregator"], how="inner")
    j["cost"] = j.cost_any.astype(float).clip(upper=6)
    for a, g in j.groupby("aggregator"):
        for p in PARTS:
            gp = g[g.partition == p]
            if len(gp) == 0:
                continue
            row = dict(dataset=ds, aggregator=a, partition=p, n=len(gp), clean_f1=round(gp.macro_f1.mean(), 3),
                       cost_median=float(np.median(gp.cost)))
            for mtr in METRICS:
                v = gp[mtr]
                row[mtr] = f"{v.median():.3g} [{v.min():.3g}..{v.max():.3g}]"
            by_part.append(row)
        for mtr in METRICS:
            r, pv = perm_p(g[mtr].to_numpy(), g.cost.to_numpy())
            within = []
            for p in PARTS:
                gp = g[g.partition == p]
                if len(gp) > 2:
                    within.append(f"{p}: {spearman(gp[mtr], gp.cost):+.2f}")
            corr.append(dict(dataset=ds, aggregator=a, metric=mtr, n=len(g), spearman=round(r, 3), perm_p=pv,
                             within_partition="; ".join(within)))

pd.DataFrame(by_part).to_csv(f"{OUT}/dispersion_by_partition.csv", index=False)
pd.DataFrame(corr).to_csv(f"{OUT}/correlation.csv", index=False)
print(pd.DataFrame(by_part).to_string(index=False))
print()
print(pd.DataFrame(corr).to_string(index=False))
