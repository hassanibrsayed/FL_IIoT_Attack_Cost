import json, sys, io, contextlib
import numpy as np
sys.path.insert(0, ".")
import poc_edge_iiot as P
out = {}
for name, path in [("edge", "archive (8)/DNN-EdgeIIoT-dataset.csv"), ("cic", "ciciot_sample.csv")]:
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, meta = P.load(path, 100000, 42, "row")
        tr, te = P.split_indices("temporal", y, meta, 42)
    u, c = np.unique(y, return_counts=True)
    out[name] = dict(rows=int(len(y)), features=int(X.shape[1]), classes=int(len(u)),
                     normal=int(c[list(u).index("Normal")]), train=int(len(tr)), test=int(len(te)),
                     min_class=int(c.min()), max_class=int(c.max()))
    print(name, out[name], flush=True)
json.dump(out, open("paper/data_stats.json", "w"), indent=1)
