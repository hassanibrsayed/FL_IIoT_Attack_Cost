"""
Proof-of-concept for the Edge-IIoTset study.

Experiment A (--exp leakage): how much of the reported accuracy survives a
  split that separates near-duplicate flows (per-class temporal split, or
  held-out source hosts), and does selecting features on the full data
  (before the final split) inflate the score?

Experiment B (--exp fl): federated learning with K clients, IID vs Dirichlet
  non-IID partitions, m malicious clients (label-flip / sign-flip), and four
  aggregators (FedAvg, Median, TrimmedMean, Krum). Reports how many malicious
  clients it takes to break each aggregator.

Usage:
  pip install pandas numpy scikit-learn torch matplotlib
  python poc_edge_iiot.py --data "path/to/ML-EdgeIIoT-dataset.csv" --exp all

Quick smoke test:
  python poc_edge_iiot.py --data ... --exp all --quick
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

warnings.filterwarnings("ignore")

# columns the original paper drops (Step 5) plus the two label columns
DROP = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
        "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
        "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
        "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]
LABEL_BIN, LABEL_MC = "Attack_label", "Attack_type"


# ----------------------------------------------------------------- data ----
def load(path, max_rows, seed, time_key="row"):
    df = pd.read_csv(path, low_memory=False)
    print(f"loaded {len(df):,} rows, {df.shape[1]} columns")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    print(f"after NaN/inf/duplicate removal: {len(df):,} rows")
    vc = df[LABEL_MC].value_counts()
    df = df[df[LABEL_MC].isin(vc[vc >= 50].index)]
    if max_rows and len(df) > max_rows:
        # cap every class so rare attacks (MITM, Fingerprinting) survive
        cap = max(200, max_rows // df[LABEL_MC].nunique())
        rng = np.random.default_rng(seed)
        keep = []
        for _, pos in df.groupby(LABEL_MC).indices.items():
            keep.append(rng.choice(pos, min(len(pos), cap), replace=False))
        df = df.iloc[np.sort(np.concatenate(keep))]
        print(f"class-capped subsample (cap {cap}/class): {len(df):,} rows, file order kept")

    # ordering key for the temporal split. In DNN-EdgeIIoT-dataset.csv the file
    # is grouped by class and rows inside a class are in capture order, while
    # frame.time has no date, so file row order is the reliable key.
    if time_key == "frame":
        t = pd.to_datetime(df.get("frame.time"), errors="coerce")
        key = t.astype("int64").to_numpy()
        print("temporal key: parsed frame.time (no date in this dataset; use with care)")
    else:
        key = df.index.to_numpy()
        print("temporal key: file row order (capture order within each class)")
    meta = {"key": key,
            "src": df.get("ip.src_host", pd.Series("", index=df.index)).astype(str).to_numpy()}
    y = df[LABEL_MC].astype(str).to_numpy()
    X = df.drop(columns=[c for c in DROP + [LABEL_BIN, LABEL_MC] if c in df.columns])
    print(f"{X.shape[1]} features, {len(np.unique(y))} classes")
    print(pd.Series(y).value_counts().to_string())
    return X.reset_index(drop=True), y, meta


def split_indices(kind, y, meta, seed, test=0.3):
    idx = np.arange(len(y))
    if kind == "random":
        return train_test_split(idx, test_size=test, stratify=y, random_state=seed)
    if kind == "temporal":  # last 30% of every class in time order is test
        tr, te = [], []
        for c in np.unique(y):
            ci = idx[y == c]
            ci = ci[np.argsort(meta["key"][ci], kind="stable")]
            cut = int(len(ci) * (1 - test))
            tr.append(ci[:cut])
            te.append(ci[cut:])
        return np.concatenate(tr), np.concatenate(te)
    if kind == "source":  # hold out whole source hosts inside every class
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for c in np.unique(y):
            ci = idx[y == c]
            hosts = np.unique(meta["src"][ci])
            m = None
            if len(hosts) >= 3:
                nte = max(1, int(round(len(hosts) * test)))
                te_hosts = set(rng.choice(hosts, nte, replace=False))
                m = np.array([h in te_hosts for h in meta["src"][ci]])
                if m.sum() == 0 or (~m).sum() < 0.3 * len(ci):
                    m = None  # one host dominates the class: cannot hold it out
            if m is None:  # fall back to a temporal cut for this class
                ci = ci[np.argsort(meta["key"][ci], kind="stable")]
                cut = int(len(ci) * (1 - test))
                tr.append(ci[:cut]); te.append(ci[cut:])
            else:
                tr.append(ci[~m]); te.append(ci[m])
        return np.concatenate(tr), np.concatenate(te)
    raise ValueError(kind)


def encode(Xtr, Xte):
    """Ordinal-encode text columns using categories seen in train only."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in Xtr.columns:
        if not pd.api.types.is_numeric_dtype(Xtr[c]):
            cats = {v: i for i, v in enumerate(pd.unique(Xtr[c].astype(str)))}
            Xtr[c] = Xtr[c].astype(str).map(cats).astype(float)
            Xte[c] = Xte[c].astype(str).map(cats).fillna(-1).astype(float)
    return Xtr.to_numpy(np.float32), Xte.to_numpy(np.float32)


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    f = lambda A: np.clip((A - mu) / sd, -10, 10).astype(np.float32)
    return f(Xtr), f(Xte)


# ------------------------------------------------- experiment A: leakage ----
def importance_ranking(X, y, seed):
    rf = RandomForestClassifier(50, n_jobs=-1, random_state=seed).fit(X, y)
    return np.argsort(rf.feature_importances_)[::-1]


def run_leakage(X, y, meta, args, out):
    rows, per_class = [], []
    k = min(args.k, X.shape[1])
    # importance computed on ALL data (the leaky variant)
    Xall, _ = encode(X, X.iloc[:1])
    rank_full = importance_ranking(Xall, y, args.seed)[:k]

    for split in ["random", "temporal", "source"]:
        tr, te = split_indices(split, y, meta, args.seed)
        Xtr, Xte = encode(X.iloc[tr], X.iloc[te])
        ytr, yte = y[tr], y[te]
        rank_train = importance_ranking(Xtr, ytr, args.seed)[:k]
        for fs, cols in [("all_features", slice(None)),
                         ("top_k_fit_on_full_data", rank_full),
                         ("top_k_fit_on_train_only", rank_train)]:
            rf = RandomForestClassifier(args.trees, n_jobs=-1, random_state=args.seed)
            rf.fit(Xtr[:, cols], ytr)
            p = rf.predict(Xte[:, cols])
            labs = np.unique(yte)
            rows.append(dict(split=split, features=fs, n_train=len(tr), n_test=len(te),
                             accuracy=accuracy_score(yte, p),
                             macro_f1=f1_score(yte, p, labels=labs, average="macro")))
            print(rows[-1])
            if fs == "all_features":
                per_class.append(pd.DataFrame({"split": split, "class": labs,
                    "f1": f1_score(yte, p, labels=labs, average=None),
                    "n_train": [int((ytr == c).sum()) for c in labs],
                    "n_test": [int((yte == c).sum()) for c in labs]}))
    pd.concat(per_class).to_csv(os.path.join(out, "leakage_per_class.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "leakage_results.csv"), index=False)
    print("\n=== Experiment A: leakage ===")
    print(df.pivot(index="features", columns="split", values="macro_f1").round(4))
    return df


# ------------------------------------------------------ experiment B: FL ---
def run_fl(X, y, meta, args, out):
    import torch
    import torch.nn as nn
    from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    classes = np.unique(y)
    cid = {c: i for i, c in enumerate(classes)}
    normal = cid.get("Normal", None)
    if normal is None:
        raise SystemExit("class 'Normal' not found in Attack_type")
    yi = np.array([cid[c] for c in y], dtype=np.int64)

    tr, te = split_indices("temporal", y, meta, args.seed)
    Xtr, Xte = standardize(*encode(X.iloc[tr], X.iloc[te]))
    ytr, yte = yi[tr], yi[te]
    Xte_t = torch.tensor(Xte, device=dev)
    d, C, K = Xtr.shape[1], len(classes), args.clients

    def make():
        return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64),
                             nn.ReLU(), nn.Linear(64, C)).to(dev)

    def partition(alpha, rng):
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

    def aggregate(name, D, sizes):
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
                float((p[atk] != normal).mean()),          # attack recall
                float((p[~atk] != normal).mean()))         # false-alarm rate

    res_path = os.path.join(out, "fl_results.csv")
    ref_path = os.path.join(out, "fl_reference.csv")
    rows, done = [], set()
    if args.resume and os.path.exists(res_path):
        prev = pd.read_csv(res_path)
        rows = prev.to_dict("records")
        done = {(int(r["seed"]), r["partition"], r["attack"], r["aggregator"], int(r["malicious"]))
                for r in rows}
        print(f"resume: {len(rows)} runs already in {res_path}, they will be skipped")

    # centralised reference: same model, same total number of epochs
    torch.manual_seed(args.seed)
    ref = make()
    opt = torch.optim.Adam(ref.parameters(), lr=1e-3)
    Xt_all, yt_all = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    if args.resume and os.path.exists(ref_path):
        print("resume: keeping the existing centralised reference")
    else:
        for _ in range(args.rounds * args.local_epochs):
            perm = torch.randperm(len(yt_all), device=dev)
            for st in range(0, len(perm), 256):
                bt = perm[st:st + 256]
                opt.zero_grad()
                nn.CrossEntropyLoss()(ref(Xt_all[bt]), yt_all[bt]).backward()
                opt.step()
        f1c, recc, farc = evaluate(ref)
        print(f"centralised reference: macro_f1={f1c:.4f} attack_recall={recc:.4f} false_alarm={farc:.4f}")
        pd.DataFrame([dict(macro_f1=f1c, attack_recall=recc, false_alarm=farc)]).to_csv(
            ref_path, index=False)
    del Xt_all, yt_all

    parts_cfg = [("iid", None)] + [(f"dir{a}", a) for a in args.alphas]
    for seed in range(args.seeds):
        rng = np.random.default_rng(args.seed + seed)
        gen = torch.Generator().manual_seed(args.seed + seed)
        torch.manual_seed(args.seed + seed)
        for pname, alpha in parts_cfg:
            parts = partition(alpha, rng)
            sizes = torch.tensor([max(len(p), 1) for p in parts], dtype=torch.float32, device=dev)
            data = [(torch.tensor(Xtr[p], device=dev), torch.tensor(ytr[p], device=dev)) for p in parts]
            mal_order = rng.permutation(K)
            for attack in args.attacks:
                for agg in args.aggs:
                    for m in range(0, args.max_mal + 1):
                        if (seed, pname, attack, agg, m) in done:
                            continue
                        t0 = time.time()
                        mal = set(mal_order[:m].tolist())
                        torch.manual_seed(args.seed + seed)
                        model, work = make(), make()
                        g = p2v(model.parameters()).detach().clone()
                        for _ in range(args.rounds):
                            D = []
                            for k in range(K):
                                Xc, yc = data[k]
                                if len(yc) == 0:
                                    D.append(torch.zeros_like(g)); continue
                                if k in mal and attack == "labelflip":
                                    yc = torch.where(yc != normal, torch.full_like(yc, normal), yc)
                                delta = local_train(work, g, Xc, yc, gen)
                                if k in mal and attack == "signflip":
                                    delta = -3.0 * delta
                                D.append(delta)
                            g = g + aggregate(agg, torch.stack(D), sizes)
                        v2p(g.clone(), model.parameters())
                        f1, rec, far = evaluate(model)
                        rows.append(dict(seed=seed, partition=pname, attack=attack,
                                         aggregator=agg, malicious=m, macro_f1=f1,
                                         attack_recall=rec, false_alarm=far))
                        print(f"[{time.time()-t0:5.1f}s] {rows[-1]}")
                        pd.DataFrame(rows).to_csv(os.path.join(out, "fl_results.csv"), index=False)
    df = pd.DataFrame(rows)
    summarize_fl(df, args, out)
    return df


def summarize_fl(df, args, out):
    M = args.max_mal
    disp = lambda x: f">{M}" if x > M else f"{x:g}"
    per = []
    for (a, p, ag, sd), g in df.groupby(["attack", "partition", "aggregator", "seed"]):
        g = g.sort_values("malicious")
        base = g.iloc[0]
        br = g[(g.attack_recall < args.recall_floor) & (g.malicious > 0)]
        bf = g[(g.macro_f1 < base.macro_f1 - args.drop) & (g.malicious > 0)]
        per.append(dict(attack=a, partition=p, aggregator=ag, seed=sd,
                        base_f1=base.macro_f1, base_recall=base.attack_recall,
                        baseline_ok=bool(base.attack_recall >= args.recall_floor),
                        break_recall=int(br.malicious.min()) if len(br) else M + 1,
                        break_f1=int(bf.malicious.min()) if len(bf) else M + 1))
    per = pd.DataFrame(per)
    per.to_csv(os.path.join(out, "fl_break_points_per_seed.csv"), index=False)

    def cell(g, col):
        if (~g.baseline_ok).mean() > 0.5:
            return "base<floor"
        v = g[col].to_numpy()
        return f"{disp(np.median(v))} [{disp(v.min())}..{disp(v.max())}]"

    rows = []
    for (a, p, ag), g in per.groupby(["attack", "partition", "aggregator"]):
        rows.append(dict(attack=a, partition=p, aggregator=ag,
                         base_f1=round(g.base_f1.mean(), 3),
                         base_recall=round(g.base_recall.mean(), 3),
                         break_recall=cell(g, "break_recall"),
                         break_f1=cell(g, "break_f1")))
    b = pd.DataFrame(rows)
    b.to_csv(os.path.join(out, "fl_break_points.csv"), index=False)
    for col, title in [("break_recall", f"attack_recall < {args.recall_floor} (security-relevant)"),
                       ("break_f1", f"macro-F1 drop > {args.drop}")]:
        print(f"\n=== min malicious clients out of {args.clients}: {title}; "
              f"median [min..max] over seeds; >{M} = never within range ===")
        print(b.pivot(index=["attack", "partition"], columns="aggregator", values=col))
    print("\nbaseline (m=0) macro-F1 / attack-recall:")
    print(b.pivot(index=["attack", "partition"], columns="aggregator", values="base_f1"))

    summ = df.groupby(["attack", "partition", "aggregator", "malicious"])[
        ["macro_f1", "attack_recall", "false_alarm"]].agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(out, "fl_summary.csv"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = df.groupby(["attack", "partition", "aggregator", "malicious"])[
            ["macro_f1", "attack_recall"]].agg(["mean", "std"]).reset_index()
        for a in df.attack.unique():
            for met in ["macro_f1", "attack_recall"]:
                parts = list(df.partition.unique())
                fig, axs = plt.subplots(1, len(parts), figsize=(4.2 * len(parts), 3.4), sharey=True)
                axs = np.atleast_1d(axs)
                for ax, p in zip(axs, parts):
                    for ag, s_ in g[(g.attack == a) & (g.partition == p)].groupby("aggregator"):
                        ax.errorbar(s_.malicious, s_[(met, "mean")], yerr=s_[(met, "std")].fillna(0),
                                    marker="o", capsize=2, label=ag)
                    ax.set_title(p)
                    ax.set_xlabel("malicious clients")
                axs[0].set_ylabel(met)
                axs[-1].legend()
                fig.suptitle(a)
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"fl_{a}_{met}.png"), dpi=150)
                plt.close(fig)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--exp", choices=["leakage", "fl", "all"], default="all")
    ap.add_argument("--out", default="poc_results")
    ap.add_argument("--max-rows", type=int, default=200000, help="rows for experiment A")
    ap.add_argument("--fl-rows", type=int, default=60000, help="rows for experiment B")
    ap.add_argument("--k", type=int, default=30, help="top-k features in experiment A")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--max-mal", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 0.1])
    ap.add_argument("--attacks", nargs="*", default=["labelflip", "signflip"])
    ap.add_argument("--aggs", nargs="*", default=["fedavg", "median", "trimmed", "krum"])
    ap.add_argument("--recall-floor", type=float, default=0.8,
                    help="attack recall below this counts as evaded")
    ap.add_argument("--drop", type=float, default=0.10, help="macro-F1 drop that counts as broken")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted --exp fl run: skip runs already in <out>/fl_results.csv "
                         "(use the same arguments as the original run)")
    ap.add_argument("--time-key", choices=["row", "frame"], default="row")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.max_rows, args.fl_rows, args.trees = 30000, 20000, 30
        args.rounds, args.max_mal, args.alphas, args.seeds = 3, 2, [0.1], 2
        args.aggs, args.attacks = ["fedavg", "median"], ["labelflip"]
    os.makedirs(args.out, exist_ok=True)

    if args.exp in ("leakage", "all"):
        X, y, meta = load(args.data, args.max_rows, args.seed, args.time_key)
        run_leakage(X, y, meta, args, args.out)
    if args.exp in ("fl", "all"):
        X, y, meta = load(args.data, args.fl_rows, args.seed, args.time_key)
        run_fl(X, y, meta, args, args.out)
    print(f"\nresults in ./{args.out}/")


if __name__ == "__main__":
    main()
