"""
Adaptive (aggregator-aware, full-knowledge, colluding) attacker for the FL-IDS study.

Threat model (worst case, so the numbers are a LOWER bound on attack cost):
  * the attacker controls m of K clients and they collude;
  * each round it sees the honest clients' updates (full knowledge) and the
    aggregation rule;
  * its goal is to EVADE detection (push attacks to "Normal", lowering attack
    recall) or to wreck the classifier (lower macro-F1);
  * offline search: it tries a small family of crafted-update strategies and the
    reported damage for each (seed, partition, aggregator, m) is the best of them.

Strategies (all malicious clients send the same crafted vector c):
  boost1   c = u                 plain label-flip update (u = update trained on flipped labels)
  boostK   c = (K/m) * u         model-replacement style boosting
  lie1     c = mu + 1.0*sd*sign(u-mu)   "A Little Is Enough" pushed toward the poison direction
  lie25    c = mu + 2.5*sd*sign(u-mu)
  rev3     c = -3 * mu           reverse the honest direction
(mu, sd = coordinate-wise mean / std of the honest updates this round.)

Heterogeneity-aware defences (added for the revision):
  cclip            centred clipping (Karimireddy et al., ICML 2021): v <- v + mean_i clip(x_i - v, tau), started from the
                   previous round's aggregate, --cclip-iters iterations, tau = median distance ||x_i - v|| (parameter-free)
  bucket_<rule>    bucketing (Karimireddy et al., ICLR 2022): random buckets of --bucket-size updates are averaged,
                   then <rule> (median, trimmed or cclip) is applied to the bucket means
--dispersion mode: trains only the clean (m = 0) models and logs, every round, how spread out the honest updates are
  (rel_spread = mean ||h_i - mean|| / ||mean||, pair_cos = mean pairwise cosine, cos_to_mean, coord_snr, norm_cv);
  output dispersion.csv (mean over all rounds and over the last 10 rounds). Used to test the mechanism behind the
  lower attack cost under non-IID data.

The partition, malicious-client order and seeds are reproduced exactly from
poc_edge_iiot.py, so m=0 baselines are read from fl_results.csv of the static run.

Outputs (in --out):
  adaptive_results.csv        every run (seed, partition, aggregator, strategy, m, metrics)
  adaptive_best.csv           best-of-strategies per (seed, partition, aggregator, m)
  adaptive_cost.csv           attack cost per (partition, aggregator): median [min..max] over seeds
  adaptive_vs_static.csv      static vs adaptive cost side by side
  adaptive_cost.png           plot
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

import poc_edge_iiot as P

warnings.filterwarnings("ignore")
STRATS = ["boost1", "boostK", "lie1", "lie25", "rev3"]


def craft(strat, mu, sd, u, K, m, hn=None):
    # hn = median L2 norm of the honest updates this round. "revn"/"boostn" are the
    # defence-aware strategies: same directions as rev3/boostK but with the norm
    # capped at hn, i.e. right at the clipping bound, so clipping cannot shrink them.
    # direction-defence-aware strategies: stay at the median norm and keep a positive cosine
    # with the honest direction while bending toward the poison direction
    if strat in ("blend5", "blend8", "revblend"):
        mh = mu / mu.norm().clamp_min(1e-12)
        uh = u / u.norm().clamp_min(1e-12)
        v = {"blend5": 0.5 * mh + 0.5 * uh, "blend8": 0.2 * mh + 0.8 * uh, "revblend": -mh + uh}[strat]
        return v / v.norm().clamp_min(1e-12) * hn
    if strat == "revn":
        return -mu / mu.norm().clamp_min(1e-12) * hn
    if strat == "boostn":
        return u / u.norm().clamp_min(1e-12) * hn
    if strat == "boost1":
        return u
    if strat == "boostK":
        return (K / m) * u
    if strat == "lie1":
        return mu + 1.0 * sd * (u - mu).sign()
    if strat == "lie25":
        return mu + 2.5 * sd * (u - mu).sign()
    if strat == "rev3":
        return -3.0 * mu
    raise ValueError(strat)


def run(args):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p

    os.makedirs(args.out, exist_ok=True)
    X, y, meta = P.load(args.data, args.fl_rows, args.seed, args.time_key)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    classes = np.unique(y)
    cid = {c: i for i, c in enumerate(classes)}
    normal = cid["Normal"]
    yi = np.array([cid[c] for c in y], dtype=np.int64)
    tr, te = P.split_indices("temporal", y, meta, args.seed)
    Xtr, Xte = P.standardize(*P.encode(X.iloc[tr], X.iloc[te]))
    ytr, yte = yi[tr], yi[te]
    Xte_t = torch.tensor(Xte, device=dev)
    d, C, K = Xtr.shape[1], len(classes), args.clients

    def make():
        return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64),
                             nn.ReLU(), nn.Linear(64, C)).to(dev)

    def partition(alpha, rng):  # identical to poc_edge_iiot.py
        parts = [[] for _ in range(K)]
        if alpha is None:
            for i, ch in enumerate(np.array_split(rng.permutation(len(ytr)), K)):
                parts[i] = list(ch)
        else:
            for c in range(C):
                ci = rng.permutation(np.where(ytr == c)[0])
                if len(ci) == 0:
                    continue
                p = rng.dirichlet(alpha * np.ones(K))
                cuts = (np.cumsum(p) * len(ci)).astype(int)[:-1]
                for i, ch in enumerate(np.split(ci, cuts)):
                    parts[i] += list(ch)
        return [np.array(p, dtype=int) for p in parts]

    def local_train(work, gvec, Xc, yc, gen):
        v2p(gvec.clone(), work.parameters())
        opt = torch.optim.Adam(work.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        n = len(yc)
        for _ in range(args.local_epochs):
            perm = torch.randperm(n, generator=gen).to(dev)
            for s in range(0, n, 256):
                b = perm[s:s + 256]
                opt.zero_grad()
                lossf(work(Xc[b]), yc[b]).backward()
                opt.step()
        return p2v(work.parameters()).detach() - gvec

    STATE = {}   # per-training-run state: centred-clipping history and the bucketing generator

    def aggregate(name, D, sizes):
        if name.startswith("bucket_"):
            s_ = args.bucket_size
            perm = torch.randperm(len(D), generator=STATE["gen"]).to(D.device)
            Dp = D[perm]
            D = torch.stack([Dp[i:i + s_].mean(0) for i in range(0, len(Dp), s_)])
            sizes = torch.ones(len(D), device=D.device)
            name = name[len("bucket_"):]
        if name == "cclip":
            v = STATE.get("v")
            v = torch.zeros_like(D[0]) if v is None else v
            for _ in range(args.cclip_iters):
                diff = D - v
                dn = diff.norm(dim=1).clamp_min(1e-12)
                tau = dn.median()
                v = v + (diff * (tau / dn).clamp(max=1.0)[:, None]).mean(0)
            STATE["v"] = v
            return v
        if name in ("cosw", "cosdrop"):
            # direction-based defences. Reference direction = coordinate-wise median update.
            nr = D.norm(dim=1).clamp_min(1e-12)
            ref = D.median(0).values
            cs = (D @ ref) / (nr * ref.norm().clamp_min(1e-12))
            Dn = D * (nr.median() / nr)[:, None]           # every update rescaled to the median norm
            if name == "cosw":                              # FLTrust-style: weight = ReLU(cosine)
                w = cs.clamp_min(0.0)
                if w.sum() < 1e-12:
                    return torch.zeros_like(D[0])
                return (w[:, None] * Dn).sum(0) / w.sum()
            keep = cs > 0.0                                 # cosdrop: drop non-positive cosine, then trimmed mean
            Dk = Dn[keep] if keep.sum() >= 1 else Dn
            t = max(1, int(0.2 * len(Dk))) if len(Dk) > 2 else 0
            return Dk.sort(0).values[t:len(Dk) - t].mean(0)
        if name.endswith("_clip"):
            # defence: clip every update's L2 norm to the median norm of the K received
            # updates (parameter-free), then apply the base aggregator
            nr = D.norm(dim=1).clamp_min(1e-12)
            D = D * (nr.median() / nr).clamp(max=1.0)[:, None]
            name = name[:-5]
        if name == "fedavg":
            w = sizes / sizes.sum()
            return (w[:, None] * D).sum(0)
        if name == "median":
            return D.median(0).values
        if name == "trimmed":
            t = max(1, int(0.2 * len(D)))
            return D.sort(0).values[t:len(D) - t].mean(0)
        if name == "krum":
            f = max(1, int(0.3 * len(D)))
            nn_ = max(1, len(D) - f - 2)
            dist = torch.cdist(D, D) ** 2
            sc = dist.sort(1).values[:, 1:nn_ + 1].sum(1)
            return D[sc.argmin()]
        raise ValueError(name)

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            p = model(Xte_t).argmax(1).cpu().numpy()
        model.train()
        atk = yte != normal
        return (f1_score(yte, p, labels=np.unique(yte), average="macro"),
                float((p[atk] != normal).mean()), float((p[~atk] != normal).mean()))

    res_path = os.path.join(args.out, "adaptive_results.csv")
    rows, done = [], set()
    if os.path.exists(res_path):
        rows = pd.read_csv(res_path).to_dict("records")
        done = {(int(r["seed"]), r["partition"], r["aggregator"], r["strategy"], int(r["malicious"]))
                for r in rows}
        print(f"resume: {len(rows)} adaptive runs already saved")

    # m=0 baselines from the static study (same seeds / partitions / aggregators)
    base = {}
    if args.static != "none":
        stat = pd.read_csv(args.static)
        base = {(int(r.seed), r.partition, r.aggregator): (r.macro_f1, r.attack_recall, r.false_alarm)
                for r in stat[(stat.malicious == 0) & (stat.attack == "labelflip")].itertuples()}

    bl_path = os.path.join(args.out, "adaptive_baselines.csv")
    bl = pd.read_csv(bl_path).to_dict("records") if os.path.exists(bl_path) else []
    for r in bl:
        base[(int(r["seed"]), r["partition"], r["aggregator"])] = (
            r["macro_f1"], r["attack_recall"], r.get("false_alarm", 0.0))

    def disp_stats(H):
        nr = H.norm(dim=1)
        H = H[nr > 1e-12]; nr = nr[nr > 1e-12]
        mu = H.mean(0); mn = mu.norm().clamp_min(1e-12)
        Hn = H / nr[:, None]
        pc = Hn @ Hn.T
        iu = torch.triu_indices(len(H), len(H), 1)
        return dict(rel_spread=float((H - mu).norm(dim=1).mean() / mn),
                    pair_cos=float(pc[iu[0], iu[1]].mean()),
                    cos_to_mean=float(((H @ mu) / (nr * mn)).mean()),
                    coord_snr=float((H.std(0) / mu.abs().clamp_min(1e-12)).median()),
                    norm_cv=float(nr.std() / nr.mean()))

    disp_path = os.path.join(args.out, "dispersion.csv")
    disp_rows = pd.read_csv(disp_path).to_dict("records") if os.path.exists(disp_path) else []
    disp_done = {(int(r["seed"]), r["partition"], r["aggregator"]) for r in disp_rows}

    def fit(agg, mal, strat, m, seed, data, sizes, rec=None):
        torch.manual_seed(args.seed + seed)
        gen = torch.Generator().manual_seed(args.seed + seed)
        STATE.clear()
        STATE["gen"] = torch.Generator().manual_seed(args.seed + seed + 7)
        model, work = make(), make()
        g = p2v(model.parameters()).detach().clone()
        honest_ids = [k for k in range(K) if k not in mal]
        for _ in range(args.rounds):
            honest, poison = [], []
            for k in range(K):
                Xc, yc = data[k]
                if len(yc) == 0:
                    (poison if k in mal else honest).append(torch.zeros_like(g))
                    continue
                if k in mal:
                    yf = torch.where(yc != normal, torch.full_like(yc, normal), yc)
                    poison.append(local_train(work, g, Xc, yf, gen))
                else:
                    honest.append(local_train(work, g, Xc, yc, gen))
            H = torch.stack(honest)
            if rec is not None:
                rec.append(disp_stats(H))
            if m == 0:
                D, sz = H, sizes[honest_ids]
            else:
                mu, sd = H.mean(0), H.std(0)
                c = craft(strat, mu, sd, torch.stack(poison).mean(0), K, m, H.norm(dim=1).median())
                D = torch.cat([H, c.unsqueeze(0).repeat(m, 1)])
                sz = torch.cat([sizes[honest_ids], sizes[sorted(mal)]])
            g = g + aggregate(agg, D, sz)
        v2p(g.clone(), model.parameters())
        return evaluate(model)

    def broken(f1, det, b):
        # det = attack recall (crit "recall") or balanced detection (rec + 1 - false alarm)/2 (crit "bd")
        if args.crit == "bd":
            return det < (b[1] + 1 - b[2]) / 2 - args.drop, f1 < b[0] - args.drop
        return det < args.recall_floor, f1 < b[0] - args.drop

    parts_cfg = [("iid", None)] + [(f"dir{a}", a) for a in args.alphas]
    for seed in range(args.seeds):
        rng = np.random.default_rng(args.seed + seed)
        for pname, alpha in parts_cfg:
            parts = partition(alpha, rng)
            mal_order = rng.permutation(K)             # same draw order as the static study
            sizes = torch.tensor([max(len(p), 1) for p in parts], dtype=torch.float32, device=dev)
            data = [(torch.tensor(Xtr[p], device=dev), torch.tensor(ytr[p], device=dev)) for p in parts]
            for agg in args.aggs:
                if args.dispersion:
                    if (seed, pname, agg) in disp_done:
                        continue
                    t0 = time.time()
                    rec = []
                    f1d, recd, fard = fit(agg, set(), None, 0, seed, data, sizes, rec)
                    R = pd.DataFrame(rec)
                    row = dict(seed=seed, partition=pname, aggregator=agg, macro_f1=f1d, attack_recall=recd, false_alarm=fard)
                    for c_ in R.columns:
                        row[c_] = R[c_].mean()
                        row[c_ + "_last10"] = R[c_].iloc[-10:].mean()
                    disp_rows.append(row)
                    pd.DataFrame(disp_rows).to_csv(disp_path, index=False)
                    print(f"[{time.time()-t0:5.1f}s] dispersion {row}", flush=True)
                    continue
                bk = (seed, pname, agg)
                if bk not in base:      # seed not in the static study: train the clean (m=0) model here
                    t0 = time.time()
                    f1b, recb, farb = fit(agg, set(), None, 0, seed, data, sizes)
                    base[bk] = (f1b, recb, farb)
                    bl.append(dict(seed=seed, partition=pname, aggregator=agg, macro_f1=f1b,
                                   attack_recall=recb, false_alarm=farb))
                    pd.DataFrame(bl).to_csv(bl_path, index=False)
                    print(f"[{time.time()-t0:5.1f}s] baseline {bk}: f1={f1b:.4f} recall={recb:.4f}", flush=True)
                b = base[bk]
                got_rec = got_f1 = False
                for m in range(1, args.max_mal + 1):
                    mal = set(mal_order[:m].tolist())
                    cell = []
                    for strat in args.strategies:
                        key = (seed, pname, agg, strat, m)
                        if key in done:
                            r = next(x for x in rows if (int(x["seed"]), x["partition"], x["aggregator"],
                                     x["strategy"], int(x["malicious"])) == key)
                            cell.append((r["macro_f1"], (r["attack_recall"] + 1 - r["false_alarm"]) / 2
                                         if args.crit == "bd" else r["attack_recall"]))
                            continue
                        t0 = time.time()
                        f1, rec, far = fit(agg, mal, strat, m, seed, data, sizes)
                        rows.append(dict(seed=seed, partition=pname, aggregator=agg, strategy=strat,
                                         malicious=m, macro_f1=f1, attack_recall=rec, false_alarm=far,
                                         base_f1=b[0], base_bd=(b[1] + 1 - b[2]) / 2))
                        print(f"[{time.time()-t0:5.1f}s] {rows[-1]}", flush=True)
                        pd.DataFrame(rows).to_csv(res_path, index=False)   # save after every run
                        cell.append((f1, (rec + 1 - far) / 2 if args.crit == "bd" else rec))
                    best_f1 = min(c_[0] for c_ in cell)
                    best_rec = min(c_[1] for c_ in cell)
                    br, bf = broken(best_f1, best_rec, b)
                    got_rec, got_f1 = got_rec or br, got_f1 or bf
                    if got_rec and got_f1:             # both criteria already broken: larger m not needed
                        break
    return pd.DataFrame(rows)


def summarize(args):
    out = args.out
    df = pd.read_csv(os.path.join(out, "adaptive_results.csv"))
    M = args.max_mal
    fmt = lambda x: f">{M}" if x > M else f"{x:g}"
    if args.crit == "bd":       # balanced detection replaces attack recall as the detection metric
        df["attack_recall"] = (df.attack_recall + 1 - df.false_alarm) / 2
    else:
        df["base_bd"] = args.recall_floor + args.drop   # so that base_bd - drop == recall_floor
    best = (df.groupby(["seed", "partition", "aggregator", "malicious"])
              .agg(best_recall=("attack_recall", "min"), best_f1=("macro_f1", "min"),
                   base_f1=("base_f1", "first"), base_bd=("base_bd", "first")).reset_index())
    best["strat_recall"] = [df[(df.seed == r.seed) & (df.partition == r.partition) &
                               (df.aggregator == r.aggregator) & (df.malicious == r.malicious)]
                            .sort_values("attack_recall").iloc[0].strategy for r in best.itertuples()]
    best["strat_f1"] = [df[(df.seed == r.seed) & (df.partition == r.partition) &
                           (df.aggregator == r.aggregator) & (df.malicious == r.malicious)]
                        .sort_values("macro_f1").iloc[0].strategy for r in best.itertuples()]
    best.to_csv(os.path.join(out, "adaptive_best.csv"), index=False)

    per = []
    for (p, ag, sd), g in best.groupby(["partition", "aggregator", "seed"]):
        g = g.sort_values("malicious")
        r = g[g.best_recall < g.base_bd - args.drop].malicious
        f = g[g.best_f1 < g.base_f1 - args.drop].malicious
        cr = r.min() if len(r) else M + 1
        cf = f.min() if len(f) else M + 1
        per.append(dict(partition=p, aggregator=ag, seed=sd, cost_recall=cr, cost_f1=cf,
                        cost_any=min(cr, cf)))
    per = pd.DataFrame(per)
    per.to_csv(os.path.join(out, "adaptive_cost_per_seed.csv"), index=False)
    agg = []
    for (p, ag), g in per.groupby(["partition", "aggregator"]):
        row = dict(partition=p, aggregator=ag, n_seeds=len(g))
        for col in ["cost_recall", "cost_f1", "cost_any"]:
            v = g[col]
            row[col] = f"{fmt(v.median())} [{fmt(v.min())}..{fmt(v.max())}]"
        agg.append(row)
    agg = pd.DataFrame(agg)
    agg.to_csv(os.path.join(out, "adaptive_cost.csv"), index=False)

    # static vs adaptive side by side (static: best of label-flip / sign-flip per seed)
    sp = args.static_breaks
    if os.path.exists(sp):
        st = pd.read_csv(sp)
        st = st.rename(columns={"break_recall": "static_labelflip_recall", "break_f1": "static_labelflip_f1"})
        cmp = agg.merge(st[st.attack == "labelflip"][["partition", "aggregator",
                        "static_labelflip_recall", "static_labelflip_f1"]],
                        on=["partition", "aggregator"], how="left")
        cmp.to_csv(os.path.join(out, "adaptive_vs_static.csv"), index=False)
        print("\n=== static vs adaptive attack cost (min #malicious of 10; median [min..max]) ===")
        print(cmp.to_string(index=False))
    else:
        print(agg.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        parts = ["iid", "dir0.5", "dir0.1"]
        for ax, col, ttl in zip(axs, ["cost_recall", "cost_f1", "cost_any"],
                                ["evade detection (recall<0.8)", "macro-F1 drop >0.10", "either"]):
            w = 0.8 / max(1, per.aggregator.nunique())
            for i, ag in enumerate(sorted(per.aggregator.unique())):
                med = [per[(per.partition == p) & (per.aggregator == ag)][col].median() for p in parts]
                ax.bar(np.arange(3) + i * w, med, w, label=ag)
            ax.set_xticks(np.arange(3) + w)
            ax.set_xticklabels(parts)
            ax.set_title(ttl, fontsize=10)
            ax.axhline(M + 0.5, color="gray", lw=0.6, ls="--")
        axs[0].set_ylabel(f"adaptive attack cost (# of 10 clients; {M+1} = never broke)")
        axs[0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, "adaptive_cost.png"), dpi=130)
    except Exception as e:  # plotting must never lose the CSVs
        print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--static", default="poc_full2/fl_results.csv")
    ap.add_argument("--static-breaks", default="poc_full2/fl_break_points.csv")
    ap.add_argument("--out", default="adaptive_out")
    ap.add_argument("--fl-rows", type=int, default=100000)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--max-mal", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 0.1])
    ap.add_argument("--aggs", nargs="*", default=["fedavg", "median", "trimmed"])
    ap.add_argument("--strategies", nargs="*", default=STRATS)
    ap.add_argument("--recall-floor", type=float, default=0.8)
    ap.add_argument("--drop", type=float, default=0.10)
    ap.add_argument("--crit", choices=["recall", "bd"], default="recall",
                    help="detection criterion: attack recall < floor, or balanced detection "
                         "(recall + 1 - false alarm)/2 dropping by --drop below its m=0 baseline")
    ap.add_argument("--time-key", default="row")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--dispersion", action="store_true", help="clean runs only; log honest-update dispersion")
    ap.add_argument("--bucket-size", type=int, default=2)
    ap.add_argument("--cclip-iters", type=int, default=3)
    args = ap.parse_args()
    if args.dispersion:
        run(args)
        print(f"\ndispersion saved in ./{args.out}/dispersion.csv")
        return
    if not args.summary_only:
        run(args)
    summarize(args)
    print(f"\nall outputs saved in ./{args.out}/")


if __name__ == "__main__":
    main()
