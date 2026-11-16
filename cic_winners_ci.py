"""Bootstrap CI of the median attack cost and winning strategy at the break point (CIC-IoT2023, bd criterion).
usage: python cic_winners_ci.py [dir]   (default cic_adaptive)"""
import sys
import numpy as np
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "cic_adaptive"
r = pd.read_csv(f"{d}/adaptive_results.csv")
r = r[r.malicious > 0].copy()
r["bd"] = (r.attack_recall + 1 - r.false_alarm) / 2
r["broke"] = (r.macro_f1 < r.base_f1 - 0.10) | (r.bd < r.base_bd - 0.10)

CENS = 6
rows, wins = [], []
for (p, a), g in r.groupby(["partition", "aggregator"]):
    costs = []
    for s, gs in g.groupby("seed"):
        b = gs[gs.broke]
        if b.empty:
            costs.append(CENS)
            continue
        m = b.malicious.min()
        costs.append(m)
        at = b[b.malicious == m]
        best = at.loc[at.macro_f1.idxmin()]
        wins.append(dict(partition=p, aggregator=a, seed=s, cost=m, winner=best.strategy,
                         n_breaking=len(at), breaking=",".join(sorted(at.strategy))))
    c = np.array(costs, float)
    rng = np.random.default_rng(0)
    meds = np.median(rng.choice(c, (10000, len(c))), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    f = lambda v: ">5" if v >= CENS else f"{v:g}"
    rows.append(dict(partition=p, aggregator=a, n_seeds=len(c), median=f(np.median(c)),
                     ci_low=f(lo), ci_high=f(hi), min=f(c.min()), max=f(c.max()),
                     censored=int((c >= CENS).sum())))

ci = pd.DataFrame(rows)
w = pd.DataFrame(wins)
ci.to_csv(f"{d}/adaptive_cost_bootstrap_ci.csv", index=False)
w.to_csv(f"{d}/adaptive_winners_per_seed.csv", index=False)
tab = (w.groupby(["partition", "aggregator", "winner"]).size().rename("seeds").reset_index()
       .sort_values(["partition", "aggregator", "seeds"], ascending=[True, True, False]))
tab.to_csv(f"{d}/adaptive_winners.csv", index=False)
print(ci.to_string(index=False))
print()
print(tab.to_string(index=False))
