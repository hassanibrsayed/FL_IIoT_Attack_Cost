"""Figures for the paper. Reads the result CSVs, writes paper/fig_*.png. Re-run after results change."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "paper"
PARTS = ["iid", "dir0.5", "dir0.1"]
PLAB = {"iid": "IID", "dir0.5": "Dir(0.5)", "dir0.1": "Dir(0.1)"}
AGGS = ["fedavg", "median", "trimmed"]
ALAB = {"fedavg": "FedAvg", "median": "Median", "trimmed": "Trimmed mean"}
COL = {"fedavg": "#C0504D", "median": "#4F81BD", "trimmed": "#9BBB59"}
CENS = 6
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "font.family": "DejaVu Sans"})


def cost_ci(per_seed_csv, aggs, col="cost_any", seed=0):
    ps = pd.read_csv(per_seed_csv)
    rng = np.random.default_rng(seed)
    out = {}
    for (p, a), g in ps.groupby(["partition", "aggregator"]):
        v = g[col].astype(float).clip(upper=CENS).to_numpy()
        b = np.median(rng.choice(v, (5000, len(v))), axis=1)
        out[(p, a)] = (np.median(v), *np.percentile(b, [2.5, 97.5]), len(v))
    return out


def fig_cost(ax, per_seed, title, aggs=AGGS, labels=ALAB, colors=COL):
    ci = cost_ci(per_seed, aggs)
    w = 0.8 / len(aggs)
    for j, a in enumerate(aggs):
        xs, med, lo, hi = [], [], [], []
        for i, p in enumerate(PARTS):
            if (p, a) not in ci:
                continue
            m, l, h, _ = ci[(p, a)]
            xs.append(i + (j - (len(aggs) - 1) / 2) * w); med.append(m); lo.append(m - l); hi.append(h - m)
        ax.bar(xs, med, w * 0.92, yerr=[lo, hi], capsize=2, color=colors.get(a, "#888"),
               label=labels.get(a, a), edgecolor="none", error_kw=dict(lw=0.8))
    ax.set_xticks(range(len(PARTS))); ax.set_xticklabels([PLAB[p] for p in PARTS])
    ax.set_ylabel("Attack cost (clients of 10)")
    ax.set_ylim(0, 5.6); ax.set_yticks(range(0, 6))
    ax.axhline(5, color="#888", lw=0.6, ls=":")
    ax.set_title(title, fontsize=9, loc="left")


# Fig 1: adaptive attack cost, both datasets
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
fig_cost(axs[0], "adaptive_out/adaptive_cost_per_seed.csv", "(a) Edge-IIoTset, 10 seeds")
fig_cost(axs[1], "cic_adaptive/adaptive_cost_per_seed.csv", "(b) CIC-IoT2023, 10 seeds")
axs[1].set_ylabel("")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(f"{OUT}/fig_adaptive_cost.png"); plt.close(fig)

# Fig 2: static (label-flip) vs adaptive cost, Edge-IIoTset
st = pd.read_csv("poc_full2/fl_break_points.csv")
st = st[(st.attack == "labelflip") & (st.aggregator.isin(AGGS))]
def first_num(s):
    s = str(s).split(" ")[0]
    return CENS if s.startswith(">") else float(s)
ad = pd.read_csv("adaptive_out/adaptive_cost.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, a in zip(axs, AGGS):
    x = np.arange(len(PARTS))
    s = [first_num(st[(st.partition == p) & (st.aggregator == a)].break_f1.iloc[0]) for p in PARTS]
    d = [first_num(ad[(ad.partition == p) & (ad.aggregator == a)].cost_any.iloc[0]) for p in PARTS]
    ax.bar(x - 0.19, s, 0.36, color="#B7B7B7", label="Static label-flip")
    ax.bar(x + 0.19, d, 0.36, color=COL[a], label="Adaptive")
    ax.set_xticks(x); ax.set_xticklabels([PLAB[p] for p in PARTS]); ax.set_title(ALAB[a], fontsize=9)
    ax.set_ylim(0, 5.6); ax.set_yticks(range(0, 6))
axs[0].set_ylabel("Attack cost (clients)")
axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_static_vs_adaptive.png"); plt.close(fig)

# Fig 3: share of seeds in which the model is broken with <= m malicious clients (both datasets)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
pc = {"iid": "#444444", "dir0.5": "#E69F00", "dir0.1": "#D55E00"}
for src, ls, lab in [("adaptive_out", "-", "Edge-IIoTset"), ("cic_adaptive", "--", "CIC-IoT2023")]:
    ps = pd.read_csv(f"{src}/adaptive_cost_per_seed.csv")
    for ax, a in zip(axs, AGGS):
        for p_ in PARTS:
            v = ps[(ps.aggregator == a) & (ps.partition == p_)].cost_any.astype(float).to_numpy()
            ax.step(range(0, 7), [(v <= m).mean() for m in range(0, 7)], where="post", ls=ls, lw=1.4, color=pc[p_])
for ax, a in zip(axs, AGGS):
    ax.set_title(ALAB[a], fontsize=9); ax.set_xlabel("Malicious clients m (of 10)"); ax.set_xlim(0, 5.2)
    ax.set_xticks(range(0, 6)); ax.set_ylim(-0.03, 1.05)
axs[0].set_ylabel("Share of seeds broken")
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=pc[p_], lw=1.6, label=PLAB[p_]) for p_ in PARTS] +      [Line2D([], [], color="k", ls="-", lw=1.2, label="Edge-IIoTset"), Line2D([], [], color="k", ls="--", lw=1.2, label="CIC-IoT2023")]
fig.legend(handles=hs, frameon=False, fontsize=7.5, loc="upper center", ncol=5)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(f"{OUT}/fig_f1_curves.png"); plt.close(fig)

# Fig 4: defences, Edge-IIoTset (median attack cost)
def med_cost(csv, agg_name, col="cost_any"):
    ps = pd.read_csv(csv)
    g = ps[ps.aggregator == agg_name].groupby("partition")[col].apply(lambda s: np.median(s.clip(upper=CENS)))
    return [g.get(p, np.nan) for p in PARTS]

cfg = [("None (trimmed)", "adaptive_out/adaptive_cost_per_seed.csv", "trimmed", "#B7B7B7"),
       ("Norm clip (trimmed)", "defense_out/adaptive_cost_per_seed.csv", "trimmed_clip", "#4F81BD"),
       ("Cos-weight", "cos_out/adaptive_cost_per_seed.csv", "cosw", "#9BBB59"),
       ("Cos-drop", "cos_out/adaptive_cost_per_seed.csv", "cosdrop", "#8064A2")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, (title, cfgs) in zip(axs, [("(a) Edge-IIoTset: defences", cfg), ("(b) FedAvg with and without clipping", None)]):
    if cfgs is not None:
        w = 0.8 / len(cfgs)
        for j, (lab, csv, a, c) in enumerate(cfgs):
            ax.bar(np.arange(3) + (j - 1.5) * w, med_cost(csv, a), w * 0.92, color=c, label=lab)
    else:
        w = 0.36
        ax.bar(np.arange(3) - 0.19, med_cost("adaptive_out/adaptive_cost_per_seed.csv", "fedavg"), w, color="#B7B7B7", label="FedAvg")
        ax.bar(np.arange(3) + 0.19, med_cost("defense_out/adaptive_cost_per_seed.csv", "fedavg_clip"), w, color="#4F81BD", label="FedAvg + clip")
    ax.set_xticks(range(3)); ax.set_xticklabels([PLAB[p] for p in PARTS])
    ax.set_ylim(0, 6.6); ax.set_yticks(range(0, 6)); ax.set_title(title, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
axs[0].set_ylabel("Attack cost (median, clients)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_defences_edge.png"); plt.close(fig)

# Fig 5: CIC clipping defence (whatever seeds exist)
p = "cic_defense_partial/adaptive_cost_per_seed.csv"
if os.path.exists(p):
    nd = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    cfgc = [("No defence", "cic_adaptive/adaptive_cost_per_seed.csv", {"fedavg": "fedavg", "median": "median", "trimmed": "trimmed"}, 0.0),
            ("Norm clip", p, {"fedavg": "fedavg_clip", "median": "median_clip", "trimmed": "trimmed_clip"}, 0.0)]
    w = 0.13
    k = 0
    for ci_, (lab, csv, mp, _) in enumerate(cfgc):
        for a in AGGS:
            v = med_cost(csv, mp[a])
            ax.bar(np.arange(3) + (k - 2.5) * w, v, w * 0.92, color=COL[a], alpha=1.0 if ci_ else 0.45,
                   hatch="" if ci_ else "//", edgecolor="white", lw=0.3)
            k += 1
    ax.set_xticks(range(3)); ax.set_xticklabels([PLAB[q] for q in PARTS])
    ax.set_ylim(0, 6.6); ax.set_yticks(range(0, 6)); ax.set_ylabel("Attack cost (median, clients)")
    ax.set_title("CIC-IoT2023: clipping (solid) vs none (hatched)", fontsize=8, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL[a], label=ALAB[a]) for a in AGGS], frameon=False, fontsize=7, loc="upper left", ncol=3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_defence_cic.png"); plt.close(fig)
print("figures written:", sorted(f for f in os.listdir(OUT) if f.startswith("fig_")))
