"""
Build a class-capped sample of CIC-IoT2023 that poc_edge_iiot.py / adaptive_attack.py can read
(--data ciciot_sample.csv), the same way they read DNN-EdgeIIoT-dataset.csv.

Facts about the downloaded CSV version (checked on disk):
  * one folder per class, many .pcap.csv chunks per folder, 39 numeric features, NO label column,
    NO IP addresses, NO timestamps. The class is the folder name.
  * every row is an aggregate over a window of 10 packets.
So: label = folder name; "time" = (chunk number, row order) which is capture order inside a class,
which is what the per-class temporal split in poc_edge_iiot.py uses (rows are written class by class,
in that order, so the row index of the output file is the ordering key).

Output columns: 39 features + Attack_type + Attack_label  ("Benign" is renamed "Normal", the name the
FL scripts look for).
"""
import argparse
import os
import re

import numpy as np
import pandas as pd


def chunk_key(name):
    m = re.search(r"(\d+)\.pcap\.csv$", name)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ciciot2023")
    ap.add_argument("--out", default="ciciot_sample.csv")
    ap.add_argument("--cap", type=int, default=6000, help="max rows kept per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    classes = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    parts, report = [], []
    for cls in classes:
        folder = os.path.join(args.root, cls)
        files = sorted((f for f in os.listdir(folder) if f.endswith(".csv")), key=chunk_key)
        frames = [pd.read_csv(os.path.join(folder, f), dtype=np.float32) for f in files]
        df = pd.concat(frames, ignore_index=True)
        n_raw = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        n_clean = len(df)
        df = df.drop_duplicates()                     # keeps first occurrence, so capture order is kept
        n_dedup = len(df)
        if n_dedup > args.cap:
            keep = np.sort(rng.choice(n_dedup, args.cap, replace=False))
            df = df.iloc[keep]
        name = "Normal" if cls.lower().startswith("benign") else cls
        df = df.copy()
        df["Attack_type"] = name
        df["Attack_label"] = 0 if name == "Normal" else 1
        parts.append(df)
        report.append((name, len(files), n_raw, n_clean, n_dedup, len(df)))
        print(f"{name:28s} files={len(files):3d} raw={n_raw:>9,} clean={n_clean:>9,} "
              f"unique={n_dedup:>9,} kept={len(df):>6,}", flush=True)

    out = pd.concat(parts, ignore_index=True)       # class by class, capture order inside a class
    out.to_csv(args.out, index=False)
    pd.DataFrame(report, columns=["class", "files", "raw", "clean", "unique", "kept"]).to_csv(
        os.path.splitext(args.out)[0] + "_report.csv", index=False)
    print(f"\nwrote {args.out}: {len(out):,} rows, {out.shape[1]-2} features, "
          f"{out.Attack_type.nunique()} classes")


if __name__ == "__main__":
    main()
