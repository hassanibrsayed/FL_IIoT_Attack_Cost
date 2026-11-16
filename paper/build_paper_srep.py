"""Scientific Reports version of the manuscript; numbers computed from the result CSVs."""
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
CENS = 6
PARTS = ["iid", "dir0.5", "dir0.1"]
PLAB = {"iid": "IID", "dir0.5": "Dir(0.5)", "dir0.1": "Dir(0.1)"}
ALAB = {"fedavg": "FedAvg", "median": "Median", "trimmed": "Trimmed mean",
        "fedavg_clip": "FedAvg+clip", "median_clip": "Median+clip", "trimmed_clip": "Trimmed+clip",
        "cosw": "Cos-weight", "cosdrop": "Cos-drop", "krum": "Krum"}
AGGS = ["fedavg", "median", "trimmed"]
stats = json.load(open("paper/data_stats.json"))


# ------------------------------------------------------------------ data helpers
def fmt(v):
    return ">5" if v >= CENS else f"{v:g}"


def cost_str(df, part, agg, col="cost_any"):
    v = df[(df.partition == part) & (df.aggregator == agg)][col].astype(float).to_numpy()
    if len(v) == 0:
        return "–"
    return f"{fmt(np.median(v))} [{fmt(v.min())}..{fmt(v.max())}]"


def boot_ci(df, part, agg, col="cost_any"):
    v = df[(df.partition == part) & (df.aggregator == agg)][col].astype(float).clip(upper=CENS).to_numpy()
    if len(v) == 0:
        return "–"
    b = np.median(np.random.default_rng(0).choice(v, (10000, len(v))), axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"[{fmt(lo)}, {fmt(hi)}]"


def nseeds(df):
    return df.seed.nunique() if "seed" in df else int(df.groupby(["partition", "aggregator"]).size().min())


def sign_p(k_less, k_more):
    n = k_less + k_more
    if n == 0:
        return 1.0
    k = min(k_less, k_more)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def paired(ps, a_part, b_part, agg_a, agg_b=None):
    """cost(a) - cost(b) per seed; positive = b cheaper for the attacker."""
    agg_b = agg_b or agg_a
    A = ps[(ps.partition == a_part) & (ps.aggregator == agg_a)].set_index("seed").cost_any.astype(float).clip(upper=CENS)
    B = ps[(ps.partition == b_part) & (ps.aggregator == agg_b)].set_index("seed").cost_any.astype(float).clip(upper=CENS)
    idx = A.index.intersection(B.index)
    d = (A[idx] - B[idx]).to_numpy()
    return dict(n=len(d), mean=d.mean() if len(d) else float("nan"), pos=int((d > 0).sum()),
                zero=int((d == 0).sum()), neg=int((d < 0).sum()), p=sign_p(int((d > 0).sum()), int((d < 0).sum())))


def winners(res_csv, crit, floor=0.8, drop=0.10):
    r = pd.read_csv(res_csv)
    r = r[r.malicious > 0].copy()
    r["bd"] = (r.attack_recall + 1 - r.false_alarm) / 2
    det = r.attack_recall < floor if crit == "recall" else r.bd < r.base_bd - drop
    r["broke"] = det | (r.macro_f1 < r.base_f1 - drop)
    out = {}
    for (p, a), g in r.groupby(["partition", "aggregator"]):
        c, n = Counter(), 0
        for s, gs in g.groupby("seed"):
            b = gs[gs.broke]
            if b.empty:
                continue
            n += 1
            at = b[b.malicious == b.malicious.min()]
            for st in set(at.strategy):
                c[st] += 1
        out[(p, a)] = (c, n)
    return out


def win_str(w, p, a):
    if (p, a) not in w:
        return "–"
    c, n = w[(p, a)]
    items = [f"{k} {v}" for k, v in c.most_common(3)]
    return ", ".join(items) + f" (of {n})" if items else "none broke"


def bsl(path_or_df):
    return pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df


# ===================================================================================================
# Scientific Reports version. Header (imports + data helpers) is copied from build_paper.py lines 4-105.
# Builds paper/FL_IIoT_Attack_Cost_SciRep.docx and paper/FL_IIoT_Attack_Cost_SciRep_Supplementary.docx
# ===================================================================================================
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
stats = json.load(open("paper/data_stats.json"))

# ------------------------------------------------------------------ references (Nature style)
REFS = {
    "mcmahan": "McMahan, B., Moore, E., Ramage, D., Hampson, S. & Agüera y Arcas, B. Communication-efficient learning of deep networks from decentralized data. In Proc. 20th International Conference on Artificial Intelligence and Statistics (AISTATS) 1273–1282 (2017).",
    "bhagoji": "Bhagoji, A. N., Chakraborty, S., Mittal, P. & Calo, S. Analyzing federated learning through an adversarial lens. In Proc. 36th International Conference on Machine Learning (ICML) 634–643 (2019).",
    "fang": "Fang, M., Cao, X., Jia, J. & Gong, N. Z. Local model poisoning attacks to Byzantine-robust federated learning. In Proc. 29th USENIX Security Symposium (2020).",
    "tolpegin": "Tolpegin, V., Truex, S., Gursoy, M. E. & Liu, L. Data poisoning attacks against federated learning systems. In Computer Security – ESORICS 2020, Lecture Notes in Computer Science vol. 12308 (Springer, 2020).",
    "yin": "Yin, D., Chen, Y., Kannan, R. & Bartlett, P. Byzantine-robust distributed learning: towards optimal statistical rates. In Proc. 35th International Conference on Machine Learning (ICML) 5650–5659 (2018).",
    "blanchard": "Blanchard, P., El Mhamdi, E. M., Guerraoui, R. & Stainer, J. Machine learning with adversaries: Byzantine tolerant gradient descent. In Advances in Neural Information Processing Systems 30 (NeurIPS) (2017).",
    "fltrust": "Cao, X., Fang, M., Liu, J. & Gong, N. Z. FLTrust: Byzantine-robust federated learning via trust bootstrapping. In Proc. Network and Distributed System Security Symposium (NDSS) (2021).",
    "baruch": "Baruch, G., Baruch, M. & Goldberg, Y. A little is enough: circumventing defenses for distributed learning. In Advances in Neural Information Processing Systems 32 (NeurIPS) (2019).",
    "shejwalkar": "Shejwalkar, V., Houmansadr, A., Kairouz, P. & Ramage, D. Back to the drawing board: a critical evaluation of poisoning attacks on production federated learning. In Proc. 2022 IEEE Symposium on Security and Privacy (SP) 1354–1371 (2022).",
    "hsu": "Hsu, T.-M. H., Qi, H. & Brown, M. Measuring the effects of non-identical data distribution for federated visual classification. Preprint at https://arxiv.org/abs/1909.06335 (2019).",
    "bucketing": "Karimireddy, S. P., He, L. & Jaggi, M. Byzantine-robust learning on heterogeneous datasets via bucketing. In Proc. 10th International Conference on Learning Representations (ICLR) (2022).",
    "cclip": "Karimireddy, S. P., He, L. & Jaggi, M. Learning from history for Byzantine robust optimization. In Proc. 38th International Conference on Machine Learning (ICML) 5311–5319 (2021).",
    "mixing": "Allouah, Y. et al. Fixing by mixing: a recipe for optimal Byzantine ML under heterogeneity. In Proc. 26th International Conference on Artificial Intelligence and Statistics (AISTATS) 1232–1300 (2023).",
    "diot": "Nguyen, T. D. et al. DÏoT: a federated self-learning anomaly detection system for IoT. In Proc. 39th IEEE International Conference on Distributed Computing Systems (ICDCS) 756–767 (2019).",
    "rahman": "Rahman, S. A., Tout, H., Talhi, C. & Mourad, A. Internet of Things intrusion detection: centralized, on-device, or federated learning? IEEE Netw. 34, 310–317 (2020).",
    "mothukuri": "Mothukuri, V. et al. Federated-learning-based anomaly detection for IoT security attacks. IEEE Internet Things J. 9, 2545–2554 (2022).",
    "friha": "Friha, O. et al. FELIDS: federated learning-based intrusion detection system for agricultural Internet of Things. J. Parallel Distrib. Comput. 165, 17–31 (2022).",
    "campos": "Campos, E. M. et al. Evaluating federated learning for intrusion detection in Internet of Things: review and challenges. Comput. Netw. 203, 108661 (2022).",
    "agrawal": "Agrawal, S. et al. Federated learning for intrusion detection system: concepts, challenges and future directions. Comput. Commun. 195, 346–361 (2022).",
    "nguyenpois": "Nguyen, T. D., Rieger, P., Miettinen, M. & Sadeghi, A.-R. Poisoning attacks on federated learning-based IoT intrusion detection system. In Proc. Workshop on Decentralized IoT Systems and Security (DISS), NDSS Symposium (2020).",
    "secfed": "Zhang, Z., Zhang, Y., Guo, D., Yao, L. & Li, Z. SecFedNIDS: robust defense for poisoning attack against federated learning-based network intrusion detection system. Future Gener. Comput. Syst. 134, 154–169 (2022).",
    "thein": "Thein, T. T., Shiraishi, Y. & Morii, M. Personalized federated learning-based intrusion detection system: poisoning attack and defense. Future Gener. Comput. Syst. 153, 182–192 (2024).",
    "bagdasaryan": "Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D. & Shmatikov, V. How to backdoor federated learning. In Proc. 23rd International Conference on Artificial Intelligence and Statistics (AISTATS) 2938–2948 (2020).",
    "bulyan": "El Mhamdi, E. M., Guerraoui, R. & Rouault, S. The hidden vulnerability of distributed learning in Byzantium. In Proc. 35th International Conference on Machine Learning (ICML) 3521–3530 (2018).",
    "rfa": "Pillutla, K., Kakade, S. M. & Harchaoui, Z. Robust aggregation for federated learning. IEEE Trans. Signal Process. 70, 1142–1154 (2022).",
    "foolsgold": "Fung, C., Yoon, C. J. M. & Beschastnikh, I. The limitations of federated learning in sybil settings. In Proc. 23rd International Symposium on Research in Attacks, Intrusions and Defenses (RAID) (2020).",
    "flame": "Nguyen, T. D. et al. FLAME: taming backdoors in federated learning. In Proc. 31st USENIX Security Symposium (2022).",
    "manip": "Shejwalkar, V. & Houmansadr, A. Manipulating the Byzantine: optimizing model poisoning attacks and defenses for federated learning. In Proc. Network and Distributed System Security Symposium (NDSS) (2021).",
    "zhaonon": "Zhao, Y. et al. Federated learning with non-IID data. Preprint at https://arxiv.org/abs/1806.00582 (2018).",
    "fedprox": "Li, T. et al. Federated optimization in heterogeneous networks. In Proc. Machine Learning and Systems (MLSys) 2, 429–450 (2020).",
    "kairouz": "Kairouz, P. et al. Advances and open problems in federated learning. Found. Trends Mach. Learn. 14, 1–210 (2021).",
    "edge": "Ferrag, M. A., Friha, O., Hamouda, D., Maglaras, L. & Janicke, H. Edge-IIoTset: a new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning. IEEE Access 10, 40281–40306 (2022).",
    "cic": "Neto, E. C. P. et al. CICIoT2023: a real-time dataset and benchmark for large-scale attacks in IoT environment. Sensors 23, 5941 (2023).",
    "sun": "Sun, Z., Kairouz, P., Suresh, A. T. & McMahan, H. B. Can you really backdoor federated learning? Preprint at https://arxiv.org/abs/1911.07963 (2019).",
    "adam": "Kingma, D. P. & Ba, J. Adam: a method for stochastic optimization. In Proc. 3rd International Conference on Learning Representations (ICLR) (2015).",
}
CITED = []


def cite_nums(keys):
    nums = []
    for k in keys:
        if k not in CITED:
            CITED.append(k)
        nums.append(CITED.index(k) + 1)
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


TOK = re.compile(r"(\*\*.+?\*\*|\[@[^\]]+\])")


def add_runs(p, text, bold=False, italic=False, size=None):
    for t in TOK.split(text):
        if not t:
            continue
        if t.startswith("**"):
            r = p.add_run(t[2:-2]); r.bold = True
        elif t.startswith("[@"):
            r = p.add_run(cite_nums([k.strip() for k in t[2:-1].split(",")])); r.font.superscript = True; r.bold = bold
        else:
            r = p.add_run(t); r.bold = bold
        r.italic = italic
        if size:
            r.font.size = Pt(size)


# ------------------------------------------------------------------ docx helpers
doc = None


def new_doc():
    global doc
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.5)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"; st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.5
    for name, size in [("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)]:
        h = doc.styles[name]
        h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True; h.font.italic = False
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        h.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
        h.paragraph_format.space_after = Pt(4); h.paragraph_format.keep_with_next = True
    # page + line numbers (reviewers expect line numbers)
    sectPr = sec._sectPr
    ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); sectPr.append(ln)
    p = sec.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for tag, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
        if tag:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r._r.append(e)
    r.font.size = Pt(9)


def P(text="", bold=False, italic=False, align="j", size=None, after=None, keep=False, spacing=None):
    p = doc.add_paragraph()
    add_runs(p, text, bold, italic, size)
    if align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "j":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if spacing is not None:
        p.paragraph_format.line_spacing = spacing
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def H(text, lvl=1):
    return doc.add_heading(text, level=lvl)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color)
    tcPr.append(sh)


def borders(t):
    b = OxmlElement("w:tblBorders")
    for edge, sz in [("top", 8), ("bottom", 8), ("insideH", 2)]:
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "808080" if edge == "insideH" else "000000"); b.append(e)
    t._tbl.tblPr.append(b)


def caption(label, text):
    p = doc.add_paragraph()
    r = p.add_run(f"{label}. "); r.bold = True; r.font.size = Pt(9.5)
    add_runs(p, text, size=9.5)
    p.paragraph_format.space_after = Pt(4); p.paragraph_format.line_spacing = 1.0
    if label.startswith(("Table", "Supplementary Table")):
        p.paragraph_format.keep_with_next = True


def table(header, rows, widths=None, size=8.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""
        r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
        shade(c, "E7E6E6")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(size)
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.0
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if ri < len(t.rows) - 1:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.keep_with_next = True
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, width_cm, label, cap):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(width_cm))
    p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(2); p.paragraph_format.line_spacing = 1.0
    caption(label, cap)


# ------------------------------------------------------------------ load all results
E_ps = pd.read_csv("adaptive_out/adaptive_cost_per_seed.csv")
C_ps = pd.read_csv("cic_adaptive/adaptive_cost_per_seed.csv")
CD_ps = pd.read_csv("cic_defense_partial/adaptive_cost_per_seed.csv")
CC_ps = pd.read_csv("cic_cos_partial/adaptive_cost_per_seed.csv")
CC_ci = pd.read_csv("cic_cos_partial/adaptive_cost_bootstrap_ci.csv")
n_runs = {k: len(pd.read_csv(f"{k}/adaptive_results.csv")) for k in
          ["adaptive_out", "defense_out", "cos_out", "cic_adaptive", "cic_defense", "cic_cos", "cic_hetdef"]}
n_runs["static"] = len(pd.read_csv("poc_full2/fl_results.csv"))
cd_seeds = int(CD_ps.groupby(["partition", "aggregator"]).seed.nunique().min())
cc_seeds = int(CC_ps.groupby(["partition", "aggregator"]).seed.nunique().min())

pr = {}
for ds, ps in [("E", E_ps), ("C", C_ps)]:
    for a in ["median", "trimmed"]:
        for q in ["dir0.5", "dir0.1"]:
            pr[(ds, a, q)] = paired(ps, "iid", q, a)
W_E = winners("adaptive_out/adaptive_results.csv", "recall")
W_C = winners("cic_adaptive/adaptive_results.csv", "bd")
W_CD = winners("cic_defense_partial/adaptive_results.csv", "bd")
W_CC = winners("cic_cos_partial/adaptive_results.csv", "bd")
HD_ps = pd.read_csv("cic_hetdef/adaptive_cost_per_seed.csv")
HD_ci = pd.read_csv("cic_hetdef/adaptive_cost_bootstrap_ci.csv")
W_HD = winners("cic_hetdef/adaptive_results.csv", "bd")
_a = HD_ps[(HD_ps.aggregator == "cclip") & (HD_ps.partition == "iid")].set_index("seed").cost_any
_b = HD_ps[(HD_ps.aggregator == "cclip") & (HD_ps.partition == "dir0.1")].set_index("seed").cost_any
HD_CC_CHEAP, HD_CC_N = int((_a - _b > 0).sum()), len(_a.index.intersection(_b.index))


TS_C = pd.read_csv("analysis/threshold_sensitivity/cost_by_threshold.csv")
TS_P = pd.read_csv("analysis/threshold_sensitivity/paired_by_threshold.csv")
_p01 = TS_P[TS_P.comparison == "iid vs dir0.1"]
TS_MIN_CHEAP, TS_MAX_COST, TS_MAX_P = int(_p01.cheaper.min()), int(_p01.costlier.max()), _p01.p.max()


MECH_B = pd.read_csv("analysis/mechanism/dispersion_by_partition.csv")
MECH_C = pd.read_csv("analysis/mechanism/correlation.csv")
_rs = MECH_C[MECH_C.metric == "rel_spread_last10"]
MECH_RHO_LO, MECH_RHO_HI, MECH_PMAX = _rs.spearman.min(), _rs.spearman.max(), _rs.perm_p.max()


def pp(ds, a, q="dir0.1"):
    r = pr[(ds, a, q)]
    return f"{r['pos']} of {r['n']}"


all01 = all(pr[(d, a, "dir0.1")]["pos"] == pr[(d, a, "dir0.1")]["n"] for d in "EC" for a in ["median", "trimmed"])
n01 = pr[("E", "median", "dir0.1")]["n"]
TITLE = "Client Heterogeneity Lowers the Attack Cost of Byzantine-Robust Federated Intrusion Detection"
ABSTRACT = (
    "Federated learning lets Industrial IoT sites train a shared intrusion detector without pooling their traffic, but compromised "
    "sites can poison the shared model. The question an operator needs answered is addressed here: how many of ten participating clients must "
    "an attacker control before the detector fails? This empirical attack cost is measured on Edge-IIoTset (15 classes) and CIC-IoT2023 "
    "(34 classes) for FedAvg, coordinate-wise median and trimmed mean, under IID and Dirichlet non-IID partitions with a temporal "
    "train/test split. Static label-flip and sign-flip attacks need three to five or more clients. A colluding white-box attacker that "
    "adapts its updates to the aggregator needs far fewer: one client breaks FedAvg in every setting, and the cost of the median and "
    "trimmed mean falls from five and four clients under IID data to about two under strongly non-IID data, "
    + (f"a drop seen in all {n01} paired seeds on both datasets. " if all01 else "a drop seen in most paired seeds on both datasets. ") +
    "Norm clipping rescues FedAvg under IID data, but none of the defences tested, including centred clipping designed for "
    "heterogeneous clients, keeps the robust rules at their IID cost under non-IID data. Robustness measured on IID data with static attacks therefore "
    "overstates the protection available to federated intrusion detection.")

# =================================================================================== MAIN MANUSCRIPT
new_doc()
P(TITLE, bold=True, align="c", size=16, after=8, spacing=1.15)
P("Hassan I. Ahmed*", align="c", after=2)
P("Cybersecurity and Privacy Research Lab, Informatics Research Department, Electronics Research Institute, Cairo, Egypt. "
  "*email: hassanibrsayed@gmail.com",
  align="c", size=10, italic=True, after=10)

H("Abstract", 1)
P(ABSTRACT)

# ------------------------------------------------------------------ INTRODUCTION
H("Introduction", 1)
P("Industrial IoT deployments produce network traffic that operators are reluctant to share, which makes federated learning (FL) "
  "attractive for training intrusion detection systems (IDS): each site trains locally and a server aggregates model updates[@mcmahan]. "
  "Federated IDS have been built for smart-home, IoT and industrial networks[@diot,rahman,mothukuri,friha], and recent reviews map "
  "the design space[@campos,agrawal]. The price is a new attack surface. A compromised site, or a malicious participant, can submit "
  "crafted updates and steer the shared model[@bhagoji,fang,tolpegin,bagdasaryan], and poisoning has already been demonstrated "
  "against federated IDS, for example to make attack traffic look normal[@nguyenpois,secfed,thein]. A large literature proposes "
  "Byzantine-robust aggregation rules, such as coordinate-wise median and trimmed mean[@yin], Krum[@blanchard], Bulyan[@bulyan] and "
  "the geometric median[@rfa], and trust-, similarity- or clustering-based filters[@fltrust,foolsgold,flame], and evaluates them by "
  "accuracy under attack.")
P("Two features of that literature limit its usefulness to an IDS operator. First, results are usually reported as accuracy at a "
  "fixed attacker fraction, not as the resource the attacker needs. An operator who knows that two of ten partners are poorly secured "
  "wants to know whether that is enough. Second, evidence is often collected on IID partitions or with non-adaptive attacks, although "
  "real sites see very different traffic mixes and robust rules are known to be fragile against attackers that adapt to "
  "them[@baruch,fang,manip,shejwalkar]. Intrusion detection also raises a metric problem: with imbalanced classes, accuracy hides the failure "
  "that matters, namely attack traffic labelled normal.")
P("Heterogeneous (non-IID) client data are a defining difficulty of federated learning even without an attacker[@zhaonon,fedprox,kairouz], "
  "and are usually simulated with Dirichlet label-skew partitions[@hsu]. Theoretical work has shown "
  "that heterogeneity weakens the guarantees of robust aggregators, because it widens the natural spread of honest updates that these "
  "rules use to separate outliers, and has proposed remedies such as bucketing[@bucketing], centred clipping[@cclip] and nearest-neighbour "
  "mixing[@mixing]. A related study by the author (H. I. Ahmed, manuscript under review) applied the same attack-cost view to "
  "false data injection in perioperative patient monitoring, where redundancy between independent devices plays the role that the "
  "aggregation rule plays here. The present study differs in three ways: the attacker poisons model updates rather than sensor "
  "readings; the reported costs are lower bounds for a strong white-box attacker rather than upper bounds from a bounded search; and "
  "its central question, how client heterogeneity changes the cost of robust aggregation, has no counterpart in that setting. "
  "What is missing for intrusion detection is an empirical, operator-level number: how many compromised sites does it "
  "take, and how much does heterogeneity change that number?")
P("The **empirical attack cost** is therefore measured: the smallest number of malicious clients, out of ten, at which the global "
  "model degrades beyond a stated threshold. The questions examined are how this cost for FedAvg, median and trimmed mean depends on how the data are "
  "distributed (IID versus Dirichlet non-IID); how much cheaper the attack becomes when the attacker adapts to the aggregator instead of "
  "following a fixed poisoning recipe; whether defences, from simple norm clipping and cosine filtering to centred clipping and "
  "bucketing designed for heterogeneous clients, restore the cost against that adaptive attacker; and whether the answers hold on a second, structurally different dataset. "
  "Edge-IIoTset[@edge] and CIC-IoT2023[@cic] are studied with a temporal train/test split, every cost is reported with its spread "
  "over seeds and a bootstrap confidence interval, and all scripts and per-run results are released.")

# ------------------------------------------------------------------ RESULTS
H("Results", 1)
H("Study design in brief", 2)
P("A server coordinates K = 10 clients for 30 rounds with full participation; m of the clients (m = 0, …, 5) are malicious. The global "
  "model is a small multilayer perceptron. Clients are partitioned IID or with Dirichlet(α) label skew, α = 0.5 or α = 0.1 (smaller α "
  "means stronger heterogeneity). Static attackers flip labels or flip the sign of their update; the adaptive attacker is white-box and "
  "colluding, observes the honest updates in every round and crafts a common malicious update with one of several strategies (Table 1). "
  "A configuration is **broken** when its macro-F1 drops by more than 0.10 relative to the clean model with the same seed, or when its "
  "detection of attacks fails (attack recall below 0.8 on Edge-IIoTset; balanced detection down by more than 0.10 on CIC-IoT2023). The "
  "**attack cost** is the smallest m that breaks the configuration; “>5” means it survived five malicious clients. Full details are given "
  "in Methods.")

H("Clean federated models", 2)
sr = pd.read_csv("poc_full2/fl_results.csv")
eb = pd.concat([sr[(sr.attack == "labelflip") & (sr.malicious == 0)][["seed", "partition", "aggregator", "macro_f1", "attack_recall", "false_alarm"]],
                pd.read_csv("adaptive_out/adaptive_baselines.csv").reindex(columns=["seed", "partition", "aggregator", "macro_f1", "attack_recall", "false_alarm"])])
cb = pd.read_csv("cic_adaptive/adaptive_baselines.csv")[["seed", "partition", "aggregator", "macro_f1", "attack_recall", "false_alarm"]]
n_fa = eb.dropna(subset=["false_alarm"]).seed.nunique()


def bl_cell(df, p, a, col):
    v = df[(df.partition == p) & (df.aggregator == a)][col].dropna()
    return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else "–"


rows = []
for p in PARTS:
    for a in AGGS:
        rows.append([PLAB[p], ALAB[a], bl_cell(eb, p, a, "macro_f1"), bl_cell(eb, p, a, "attack_recall"), bl_cell(eb, p, a, "false_alarm"),
                     bl_cell(cb, p, a, "macro_f1"), bl_cell(cb, p, a, "attack_recall"), bl_cell(cb, p, a, "false_alarm")])
caption("Table 2", f"**Clean (m = 0) federated models.** Mean ± s.d. over seeds: Edge-IIoTset macro-F1 and recall over {eb.seed.nunique()} seeds, "
        f"Edge-IIoTset false alarm over {n_fa} seeds (the clean false-alarm rate was not logged for the remaining seeds); CIC-IoT2023 over "
        f"{cb.seed.nunique()} seeds. False alarm = share of Normal test rows classified as an attack. Centralised reference (same network, same "
        "number of epochs, one seed): Edge-IIoTset macro-F1 0.783, recall 1.000, false alarm 0.006; CIC-IoT2023 macro-F1 0.547, recall 0.969, "
        "false alarm 0.430.")
table(["Partition", "Aggregator", "Edge F1", "Edge recall", "Edge FA", "CIC F1", "CIC recall", "CIC FA"], rows,
      widths=[1.8, 2.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], size=8)
P("On Edge-IIoTset, federated training costs 11 points of macro-F1 under IID data and 15 to 30 points under Dirichlet partitions "
  "relative to the centralised reference (0.783). Attack recall is 1.000 and the false-alarm rate about 0.002, except for one median-rule "
  "seed at Dir(0.1) whose clean model already raises false alarms on almost all Normal rows (Table 2). On CIC-IoT2023 the 34-class "
  "problem is much harder for this small network: the centralised macro-F1 is 0.547 and the clean false-alarm rate 0.43, because Normal "
  "is only 3% of the data and several attack classes overlap with it. The CIC-IoT2023 models are therefore weak detectors to begin with, "
  "which is why recall alone is not a usable failure criterion there, and at Dir(0.1) some clean models are close to collapsed "
  "(false alarm up to 1.0). CIC-IoT2023 is therefore treated as a qualitative replication.")

H("An adaptive attacker needs one client against FedAvg and two to five against robust rules", 2)
av = pd.read_csv("adaptive_out/adaptive_vs_static.csv")
rows = []
for p in PARTS:
    for a in AGGS:
        r = av[(av.partition == p) & (av.aggregator == a)].iloc[0]
        rows.append([PLAB[p], ALAB[a], cost_str(E_ps, p, a), boot_ci(E_ps, p, a), r.static_labelflip_f1,
                     cost_str(C_ps, p, a), boot_ci(C_ps, p, a)])
caption("Table 3", "**Attack cost of the adaptive attacker.** Clients of 10 needed to break the model (either criterion); median [min..max] "
        f"over {E_ps.seed.nunique()} seeds with the 95% bootstrap confidence interval of the median. The static label-flip cost on Edge-IIoTset "
        "(5 seeds, macro-F1 criterion) is shown for comparison; the full static grid is in Supplementary Table S1.")
table(["Partition", "Aggregator", "Edge adaptive", "95% CI", "Edge static LF", "CIC adaptive", "95% CI"], rows,
      widths=[1.8, 2.3, 2.5, 2.0, 2.5, 2.5, 2.0], size=8.5)
figure("paper/fig_adaptive_cost.png", 15.5, "Figure 1",
       "**Attack cost of the adaptive attacker.** Median over 10 seeds with 95% bootstrap confidence interval of the median, "
       "(a) Edge-IIoTset and (b) CIC-IoT2023. Lower bars mean a cheaper attack.")
P("**FedAvg.** One malicious client is enough in every partition on both datasets (median cost 1; at most 3 in any seed). The attacker "
  "simply scales its update (boostK) so that a single client carries the weight of the whole federation; linear aggregation offers no "
  "resistance.")
P("**Median and trimmed mean.** Under IID data the attacker needs 5 clients for the median and 4 for the trimmed mean on both datasets, "
  "essentially a majority or near-majority, in line with the breakdown points of these rules. Under Dirichlet(0.1) the median costs 2.5 "
  "clients on Edge-IIoTset and 2 on CIC-IoT2023, and the trimmed mean costs 2 on both (Table 3). Figure 2 shows the share of seeds in "
  "which the model is already broken with at most m clients: under IID data the curves jump at 4 or 5 clients, under non-IID data they "
  "rise from 1–2 clients onwards.")
figure("paper/fig_f1_curves.png", 15.5, "Figure 2",
       "**Share of seeds in which the model is broken with at most m malicious clients.** Solid lines: Edge-IIoTset; dashed lines: CIC-IoT2023.")
P("**Adaptive versus static attacks.** In every cell the adaptive cost is at most the static label-flip cost, and it is strictly lower "
  "in 7 of 9 cells, including all FedAvg cells (static cost 5 under IID data and 3–5 under non-IID data; adaptive cost 1; Supplementary "
  "Table S1 and Supplementary Fig. S1). Under IID data the static label-flip attack already needs five or more clients against the median, "
  "and the adaptive attacker does not improve on that: the IID median is the most robust undefended configuration found. The static "
  "study also exposes a metric pitfall. The sign-flip attack never pushes attack recall below 0.8 with up to five clients, yet three "
  "clients break FedAvg under IID data by the macro-F1 criterion, and two under Dirichlet data; an evaluation based on recall alone would "
  "declare every rule safe against sign-flip. Both criteria are therefore used throughout. Krum collapses under non-IID data even without "
  "attack (clean macro-F1 0.18–0.19 at Dir(0.1)) and was not carried forward.")

H("Non-IID data lower the cost of robust aggregation", 2)
rows = []
for ds, nm in [("E", "Edge-IIoTset"), ("C", "CIC-IoT2023")]:
    for a in ["median", "trimmed"]:
        for q in ["dir0.5", "dir0.1"]:
            r = pr[(ds, a, q)]
            rows.append([nm, ALAB[a], f"IID vs {PLAB[q]}", r["n"], f"{r['mean']:.1f}", f"{r['pos']} / {r['zero']} / {r['neg']}", f"{r['p']:.3g}"])
caption("Table 4", "**Paired comparison of attack cost, IID minus non-IID, per seed.** Same seed = same partition and same malicious clients. "
        "Cheaper / equal / costlier counts the seeds in which the non-IID cost is lower / equal / higher than the IID cost. Two-sided exact "
        "sign test, ties excluded; p-values are descriptive (no multiple-comparison correction).")
table(["Dataset", "Aggregator", "Comparison", "Seeds", "Mean diff.", "Cheaper / equal / costlier", "p"], rows,
      widths=[2.5, 2.4, 2.9, 1.3, 1.9, 3.6, 1.4], size=8.5)
P("Heterogeneity makes the median and trimmed-mean rules cheaper to attack on both datasets (Table 4). On Edge-IIoTset, Dirichlet(0.1) "
  f"is cheaper than IID in {pp('E','median')} seeds for the median (mean difference {pr[('E','median','dir0.1')]['mean']:.1f} clients) "
  f"and in {pp('E','trimmed')} for the trimmed mean ({pr[('E','trimmed','dir0.1')]['mean']:.1f}); on CIC-IoT2023 the corresponding "
  f"counts are {pp('C','median')} ({pr[('C','median','dir0.1')]['mean']:.1f}) and {pp('C','trimmed')} "
  f"({pr[('C','trimmed','dir0.1')]['mean']:.1f}). Dirichlet(0.5) is intermediate, but the difference between Dirichlet(0.5) and "
  "Dirichlet(0.1) is not established: their bootstrap intervals overlap (Table 3).")
P("**Why heterogeneity lowers the cost.** Robust rules separate malicious updates from honest ones by their distance from the "
  "bulk; label skew should widen that bulk, so that a crafted update no longer stands out. To test this, the spread of the "
  "honest updates was logged in every clean run (Methods). Label skew roughly doubles it: the mean distance of an honest update from the honest "
  "mean, relative to the norm of that mean, rises from about 1.2 under IID data to 2.6–3.4 under the Dirichlet partitions on "
  "Edge-IIoTset, and from about 1.5 to 2.1–2.6 on CIC-IoT2023, while the mean pairwise cosine similarity of honest updates falls from "
  "0.23–0.36 to below 0.10 (Supplementary Table S10). Across seed × partition cells the spread is negatively rank-correlated with the "
  f"attack cost for both rules on both datasets (Spearman ρ from {f'{MECH_RHO_LO:.2f}'.replace('-', '−')} to {f'{MECH_RHO_HI:.2f}'.replace('-', '−')}, permutation p ≤ "
  f"{max(MECH_PMAX, 0.001):.3f}; Supplementary Table S11). Within a single partition the correlation is weak and inconsistent, and the "
  "spread is similar under Dirichlet(0.5) and Dirichlet(0.1), in line with the unresolved cost difference between them. The "
  "measurements therefore support this mechanism for the contrast between IID and non-IID data, but do not show that it is the only "
  "cause.")
P("The IID versus Dirichlet(0.1) contrast does not depend on the thresholds that define a break (Supplementary Tables S8 and S9). "
  "With macro-F1 and balanced-detection drops of 0.05, 0.15 or 0.20 instead of 0.10, and with recall floors of 0.7 or 0.9 instead of "
  f"0.8, Dirichlet(0.1) remains cheaper than IID for both rules on both datasets in at least {TS_MIN_CHEAP} of 10 paired seeds and costlier "
  f"in at most {TS_MAX_COST} (sign test p ≤ {TS_MAX_P:.3f}). Individual costs move by about one client across this range. The weaker "
  "Dirichlet(0.5) contrast is less stable on CIC-IoT2023 at loose thresholds, partly because runs that stopped early at the original "
  "thresholds leave some seeds unresolved there.")

H("Cheap strategies are sufficient", 2)
P("Several strategies often break a configuration at the same minimum cost (Supplementary Table S2), so the counts show which recipes "
  "are sufficient rather than which is strongest. FedAvg is broken by boostK in every seed. Under IID data the median and trimmed mean "
  "are broken by the reversal rev3 and the boosted poison direction boostK in essentially all seeds. Under non-IID data the set of "
  "successful strategies widens to include the coordinate-wise “a little is enough” attacks (lie1, lie25)[@baruch], while rev3 and boostK "
  "still succeed in most seeds. A cheap strategy that exploits the arithmetic of the aggregation rule is enough; elaborate per-coordinate "
  "stealth is not needed.")

H("Defences do not restore the cost under non-IID data", 2)
dc = pd.read_csv("cos_out/defense_comparison.csv")
rows = [[PLAB[r.partition], r["fedavg none"], r["fedavg clip"], r["median none"], r["median clip"], r["trimmed none"], r["trimmed clip"],
         r["cosw"], r["cosdrop"]] for _, r in dc.iterrows()]
caption("Table 5", "**Attack cost on Edge-IIoTset without and with defences.** Clients of 10; median [min..max] over 5 seeds, either "
        "criterion. Undefended entries come from the 10-seed adaptive study restricted to the same 5 seeds. +clip = norm clipping.")
table(["Partition", "FedAvg", "+clip", "Median", "+clip", "Trimmed", "+clip", "Cos-weight", "Cos-drop"], rows,
      widths=[1.8, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.9, 1.9], size=8)
figure("paper/fig_defences_edge.png", 15.5, "Figure 3",
       "**Defences against the adaptive attacker on Edge-IIoTset.** (a) Trimmed mean without defence, with norm clipping and with the "
       "two cosine filters; (b) FedAvg with and without norm clipping.")
P("**Norm clipping** is highly effective for FedAvg: its cost rises from 1 to 5 clients under IID data (all five seeds improve), to 3 "
  "under Dirichlet(0.5) and to 2 under Dirichlet(0.1) (Table 5; paired results in Supplementary Table S3). It does almost nothing for the "
  "median and trimmed mean under non-IID data (mean paired gain between −0.4 and +0.4 clients); for the trimmed mean under IID data it "
  "adds one client (4 → 5). Against clipped aggregators the norm-matched strategies succeed in nearly every cell: revn, a reversed update "
  "at exactly the clipping radius, breaks the model in at least 3 of 5 seeds in every clipped cell. Clipping does not harm clean "
  "performance (clean macro-F1 0.674, 0.609–0.628 and 0.480–0.495 for IID, Dir(0.5) and Dir(0.1), against 0.673–0.675, 0.607–0.631 "
  "and 0.481–0.517 without clipping).")
P("**Cosine filters** (cos-weight and cos-drop) reach 5 clients under IID data and 3 to 4 under the Dirichlet partitions. Against them "
  "the attacker also used norm-matched “blend” strategies that keep a positive cosine with the honest direction while bending the "
  "aggregate towards the poison direction; one cos-drop seed was broken by a single client. The winning strategies are the norm-matched "
  "ones (boostn, revn, revblend). No defended configuration survived five clients under non-IID data, and the most robust non-IID cell "
  "has median cost 4 (cos-drop, Dir(0.5)).")
rows = []
for p_ in PARTS:
    for a_, lab in [("cclip", "Centred clipping"), ("bucket_cclip", "Bucketing + CC"), ("bucket_median", "Bucketing + median")]:
        r_ = HD_ci[(HD_ci.partition == p_) & (HD_ci.aggregator == a_)].iloc[0]
        rows.append([PLAB[p_], lab, cost_str(HD_ps, p_, a_), f"[{r_.ci_low}, {r_.ci_high}]",
                     cost_str(C_ps[C_ps.seed < 5], p_, "median"), cost_str(C_ps[C_ps.seed < 5], p_, "trimmed"), win_str(W_HD, p_, a_)])
caption("Table 6", "**Defences designed for heterogeneous clients, CIC-IoT2023.** Attack cost of the adaptive attacker (clients of 10; "
        "median [min..max] over 5 seeds, 95% bootstrap CI of the median; balanced-detection and macro-F1 criteria). CC = centred "
        "clipping; buckets of two. The undefended median and trimmed mean are shown for the same 5 seeds.")
table(["Partition", "Defence", "Cost", "95% CI", "Median (none)", "Trimmed (none)", "Winning strategies"], rows,
      widths=[1.6, 2.8, 1.9, 1.5, 2.1, 2.1, 4.4], size=8)
P("**Defences designed for heterogeneous clients.** Centred clipping[@cclip] and bucketing[@bucketing], which were "
  "proposed for this setting, were also tested on CIC-IoT2023 (Table 6). Centred clipping is the most robust rule in the study: it needs 5 clients under "
  "IID data, like the median, and keeps 4 under Dirichlet(0.5). It still loses ground as heterogeneity grows, to 3 clients (range 2–3) "
  f"under Dirichlet(0.1), and {HD_CC_CHEAP} of {HD_CC_N} seeds are cheaper under Dirichlet(0.1) than under IID data. Bucketing with "
  "buckets of two lowers the cost to 3 clients under IID data, whether the median or centred clipping is applied to the bucket means, "
  "and to 2 under Dirichlet(0.1): with ten clients there are only five buckets, so three malicious clients can contaminate a majority "
  "of them. None of these rules changes clean performance (clean macro-F1 0.50, 0.46–0.47 and 0.39–0.41 for IID, Dir(0.5) and Dir(0.1)). "
  "Against centred clipping the reversal strategies (rev3, revn) and revblend break the model in every seed at the minimum cost; "
  "against bucketing, rev3 and boostK do.")

H("Replication on CIC-IoT2023", 2)
P("The adaptive study reproduces the qualitative findings of Edge-IIoTset (Table 3, Fig. 1, Table 4): FedAvg costs one client "
  "everywhere; the median costs 5, 3 and 2 clients and the trimmed mean 4, 3 and 2 for IID, Dirichlet(0.5) and Dirichlet(0.1) data; and "
  "the IID versus Dirichlet(0.1) difference is consistent across seeds. The defence studies (5 seeds each) also agree. With norm clipping, "
  f"FedAvg costs {fmt(np.median(CD_ps[(CD_ps.partition=='iid') & (CD_ps.aggregator=='fedavg_clip')].cost_any.clip(upper=CENS)))} clients "
  f"under IID data and {fmt(np.median(CD_ps[(CD_ps.partition=='dir0.5') & (CD_ps.aggregator=='fedavg_clip')].cost_any.clip(upper=CENS)))} / "
  f"{fmt(np.median(CD_ps[(CD_ps.partition=='dir0.1') & (CD_ps.aggregator=='fedavg_clip')].cost_any.clip(upper=CENS)))} under "
  "Dirichlet(0.5) / Dirichlet(0.1) data, while the median and trimmed mean gain at most about one client under non-IID data (Fig. 4, "
  "Supplementary Table S4). The two cosine filters reach 5 clients under IID data and 3 to 4 under the Dirichlet partitions "
  "(Supplementary Table S5). The replication has three caveats: the clean models are weak, so the balanced-detection criterion is "
  "measured around a poor baseline; the Dirichlet(0.1) cells are the least reliable; and the set of successful strategies differs "
  "somewhat from Edge-IIoTset. The static-attack grid was not run on CIC-IoT2023.")
figure("paper/fig_defence_cic.png", 9.5, "Figure 4",
       "**Norm clipping on CIC-IoT2023.** Attack cost with norm clipping (solid) versus without a defence (hatched), median over 5 seeds.")

# ------------------------------------------------------------------ DISCUSSION
H("Discussion", 1)
P("In a ten-partner federation the question “are two compromised partners enough?” has a direct empirical answer under the "
  "assumptions of this study: under IID data and a robust aggregator, no (cost 4 to 5); under non-IID data, yes for the median and the trimmed mean "
  "(cost 2 to 3); and for FedAvg a single partner suffices. Reporting accuracy at a fixed malicious fraction hides this, because the "
  "quantity of interest is where the curve in Fig. 2 falls.")
P("IID evaluation is optimistic. The attacker needs half of the federation only under IID data, with the median rule or with one of "
  "the tested defences. A robust aggregation rule validated on IID partitions will overstate its protection for sites whose traffic "
  "differs, which is the normal situation in Industrial IoT. It is therefore recommended that robustness studies of federated IDS report at least one "
  "strongly non-IID partition and an adaptive attacker.")
P("The defences tested here do not close the gap. Clipping fixes the single-client weakness of FedAvg but not the weakness of the "
  "median and trimmed mean under non-IID data, because there the adaptive attacker already operates inside the honest spread; the "
  "cosine filters add at most one client. Centred clipping, designed for heterogeneous clients, is the strongest rule tested, but "
  "on CIC-IoT2023 its cost still falls from 5 to 3 clients between IID and Dirichlet(0.1) data, and bucketing in a federation of ten "
  "clients lowers the cost because it leaves only five buckets to corrupt; larger federations may favour bucketing. Nearest-neighbour "
  "mixing[@mixing], trust bootstrapped from server-side data[@fltrust] and the heterogeneity-aware defences on Edge-IIoTset remain to be "
  "evaluated, and the conclusions about defences are limited to the rules tested.")
P("Several results are negative or partial: none of the tested defences reached five clients under non-IID data; the static label-flip "
  "attack under IID data is already expensive, so the adaptive attacker brings no benefit there against the median; and the gradient "
  "between Dirichlet(0.5) and Dirichlet(0.1) is not established.")
P("**Limitations.** One architecture (a 128–64 multilayer perceptron), one federation size (K = 10, full participation), 30 "
  "rounds and one local epoch per round were used; attack cost depends on K and on training length, which were not varied. The modest clean "
  "performance, particularly on CIC-IoT2023, may make the models easier to degrade than a fully converged detector. The adaptive attacker "
  "is white-box and colluding and observes honest updates in every round, so the costs are lower bounds on what a capable adversary needs, "
  "not predictions for a real one; its strategy set is small and hand-designed, and a stronger attacker, such as an optimisation-based attack tailored to each aggregation rule[@manip], would lower the costs "
  "further. "
  "Costs are integers between 1 and 5 (or censored) defined relative to a 0.10 drop; other thresholds shift individual costs by about "
  "one client, although the IID versus Dirichlet(0.1) contrast held at every threshold tried. With 10 "
  "(attack) and 5 (defence) seeds the confidence intervals are coarse. The temporal split reduces, but does not remove, optimism from "
  "near-duplicate flows; rows are flows or packet windows rather than full sessions, and the capture order of both datasets is inferred "
  "from file order. The heterogeneity-aware defences were evaluated on CIC-IoT2023 only. Differential privacy, secure aggregation, client sampling, targeted backdoors and feature-skew heterogeneity were "
  "not evaluated.")
P("In summary, a colluding adaptive attacker needs one of ten clients against FedAvg and two to five against the median and trimmed "
  "mean, and the cost falls as client data become more heterogeneous (median rule: 5 → 3 → 2–2.5 clients from IID to Dirichlet(0.1) on "
  "both datasets). Norm clipping rescues FedAvg but not the robust rules under non-IID data, two cosine filters buy at most one client, and "
  "centred clipping, the strongest defence tested, still falls from five to three clients. Robustness results obtained on IID data with static attacks should not be taken as evidence of protection in practice. Future "
  "work should cover larger federations with client sampling, learned attackers, further heterogeneity-aware defences on both datasets, stronger baseline models, "
  "and the static grid and 10-seed defence runs on CIC-IoT2023.")

# ------------------------------------------------------------------ METHODS
H("Methods", 1)
H("Federated setting", 2)
P("A server coordinates K = 10 clients for T = 30 rounds. In each round every client trains the current global model locally and "
  "returns its update (the difference between local and global parameters); the server aggregates the K updates. All clients participate "
  "in every round. Of the K clients, m are malicious (m = 0, 1, …, 5) and the other K − m are honest. Malicious clients are chosen by a "
  "fixed random order per seed and partition, so the malicious set for m is contained in the set for m + 1.")
H("Data and temporal split", 2)
P("The DNN-EdgeIIoT table of Edge-IIoTset[@edge] and the per-capture CSV feature files of CIC-IoT2023[@cic] were used (each row "
  "aggregates a window of packets; labels are given by the folder and there are no IP addresses or timestamps). Both were cleaned (rows with NaN or "
  "infinite values and duplicate rows removed) and class-capped, so that rare classes survive, to about 100,000 rows before the 70/30 "
  "split (Supplementary Table S6). Edge-IIoTset was reduced to 46 features after removing identifier-like and payload columns "
  "(timestamps, IP addresses, ports, URIs and payload fields); categorical columns were ordinal-encoded using training categories only. "
  "CIC-IoT2023 keeps its 39 features. Features were standardised with training statistics and clipped to ±10.")
P("For every class the first 70% of rows in capture order form the training set and the last 30% the test set. Random splits can leak "
  "near-duplicate flows across train and test: on Edge-IIoTset a random-forest classifier (100 trees) reaches macro-F1 0.908 on a random "
  "split and 0.884 on the temporal split (accuracy 0.926 versus 0.908), a 2.4-point difference that is small but in the optimistic "
  "direction. The temporal split was used throughout. Capture order is taken as file order within each class (Edge-IIoTset timestamps "
  "carry no date; for CIC-IoT2023 the order is chunk number plus row order).")
H("Dispersion of honest updates", 2)
P("To examine the mechanism, the clean (m = 0) runs for the median and trimmed mean were repeated with the same seeds while logging, "
  "in every round, the honest updates h₁, …, h_K and their mean h̄. Three measures are reported, each averaged over the last 10 rounds: the "
  "relative spread (mean of ‖hᵢ − h̄‖ divided by ‖h̄‖), the mean pairwise cosine similarity between honest updates, and the median "
  "over coordinates of the per-coordinate standard deviation divided by |h̄|. The rank correlation between spread and attack cost over the 30 "
  "seed × partition cells of each dataset and rule is tested with 10,000 label permutations.")
H("Model and partitions", 2)
P("Every client trains the same multilayer perceptron (input → 128 → 64 → classes, ReLU) with Adam[@adam] (learning rate 10⁻³, batch "
  "size 256, one local epoch per round). Partitions are IID (random equal shards) or Dirichlet(α) label skew[@hsu] with α = 0.5 and "
  "α = 0.1, where for each class the proportions across clients are drawn from Dir(α·1). The partition, malicious-client order and model "
  "initialisation depend only on the seed, so every attack and defence is evaluated on the same federation as its clean baseline.")
H("Attackers", 2)
P("Static attackers follow a fixed recipe and do not observe other clients. **Label-flip**: the malicious client relabels all attack "
  "samples in its local data as Normal and trains as usual. **Sign-flip**: the malicious client trains honestly and submits −3 times its "
  "update. The **adaptive attacker** is white-box and colluding. In every round it observes the honest updates, computes their mean μ "
  "and per-coordinate standard deviation σ, and the mean u of the label-flip updates its own clients would produce. All m malicious "
  "clients then submit the same crafted vector c, built with one of the strategies in Table 1. The attacker tries every strategy, and the "
  "reported cost is the smallest m at which any strategy succeeds; this makes the cost a conservative (low) estimate of what a capable "
  "adversary needs.")
caption("Table 1", "**Update-crafting strategies of the adaptive attacker.** K = 10 clients, m malicious, hn = median L2 norm of the honest "
        "updates in the round. The last three rows are the norm-matched variants used against norm clipping and cosine filtering.")
table(["Strategy", "Malicious update c", "Idea"],
      [["boost1", "u", "poison direction, unscaled"],
       ["boostK", "(K/m)·u", "scale so that m clients carry the weight of K"],
       ["lie1", "μ + 1.0·σ·sign(u − μ)", "stay within one honest standard deviation per coordinate"],
       ["lie25", "μ + 2.5·σ·sign(u − μ)", "larger coordinate-wise shift"],
       ["rev3", "−3·μ", "push against the honest mean"],
       ["revn", "−μ/‖μ‖ · hn", "reverse direction at the honest median norm"],
       ["boostn", "u/‖u‖ · hn", "poison direction at the honest median norm"],
       ["blend5, blend8, revblend", "normalised mixes of μ/‖μ‖ and u/‖u‖, scaled to hn", "keep a positive cosine with the honest direction"]],
      widths=[3.3, 5.8, 6.9], size=8.5)
H("Aggregators and defences", 2)
P("**FedAvg** is the size-weighted mean; **coordinate-wise median** and **trimmed mean** (20% trimmed from each side, at least one "
  "update) follow Yin et al.[@yin]. Krum[@blanchard] was evaluated in the static study only. Defences are composed with these rules. "
  "**Norm clipping** (+clip)[@sun] rescales every received update to at most the median L2 norm of the K updates and then applies the "
  "base rule. **Cos-weight** rescales all updates to the median norm and weights each by ReLU(cosine similarity to the coordinate-wise "
  "median update), an FLTrust-style rule[@fltrust] without a root dataset. **Cos-drop** discards updates whose cosine similarity to the "
  "median update is ≤ 0, rescales the rest and applies the trimmed mean. Against these defences the attacker uses the norm-matched "
  "strategies of Table 1. **Centred clipping**[@cclip] starts from the previous round's aggregate v and applies three iterations of "
  "v ← v + (1/K)·Σᵢ clip(xᵢ − v, τ), with τ the median of the distances ‖xᵢ − v‖, so that no parameter is tuned. **Bucketing**[@bucketing] "
  "randomly groups the K updates into buckets of two, averages each bucket and applies the coordinate-wise median or centred clipping "
  "to the five bucket means. Against these rules the attacker used boostK, rev3, lie1, lie25, revn, boostn and revblend.")
H("Attack cost", 2)
P("A configuration is broken when the global model, evaluated on the held-out temporal test set, satisfies either (a) its macro-F1 is "
  "more than 0.10 below that of the m = 0 model trained with the same aggregator, partition and seed, or (b) its detection of attacks "
  "fails. For Edge-IIoTset, (b) means attack recall < 0.8 (clean models reach recall ≈ 1.0). For CIC-IoT2023, where clean models already "
  "have a high false-alarm rate, balanced detection is used instead: bd = (recall + 1 − false-alarm rate)/2 dropping more than 0.10 below its m = 0 "
  "value. The attack cost is the smallest m ∈ {1, …, 5} for which the configuration is broken by (a) or (b); if none is, the cost is "
  "censored at “>5” and counted as 6 in medians and bootstrap intervals. The median over seeds is reported with the [min..max] range and a "
  "percentile bootstrap (10,000 resamples) 95% confidence interval of the median.")
H("Runs and statistics", 2)
P(f"Static study on Edge-IIoTset: 5 seeds, m = 0…5, both static attacks, four aggregators and three partitions ({n_runs['static']:,} runs). "
  f"Adaptive study: 10 seeds per dataset, all strategies, m = 1…5 with early stopping once both criteria are broken (Edge-IIoTset "
  f"{n_runs['adaptive_out']:,} runs; CIC-IoT2023 {n_runs['cic_adaptive']:,} runs). Defence studies, 5 seeds each: on Edge-IIoTset, norm "
  f"clipping {n_runs['defense_out']:,} runs and cosine filters {n_runs['cos_out']:,} runs; on CIC-IoT2023, norm clipping "
  f"{n_runs['cic_defense']:,} runs, cosine filters {n_runs['cic_cos']:,} runs and heterogeneity-aware defences {n_runs['cic_hetdef']:,} runs. Paired comparisons use per-seed costs (same seed = same "
  "partition and same malicious clients) and report the mean difference, the number of seeds with lower, equal and higher cost, and a "
  "two-sided exact sign test that ignores ties. No correction for multiple comparisons was applied; p-values are descriptive. For the "
  "threshold sensitivity analysis the adaptive-study costs were recomputed from the same runs with drops of 0.05, 0.10, 0.15 and 0.20 "
  "and, on Edge-IIoTset, recall floors of 0.7, 0.8 and 0.9; no model was retrained. Because each run sequence stopped once both "
  "criteria were broken at the original thresholds, a seed that is not broken at a looser threshold by the last m that was run is "
  "unresolved; it is counted as “>5” and flagged.")

# ------------------------------------------------------------------ END MATTER
H("Data availability", 1)
P("Edge-IIoTset and CIC-IoT2023 are publicly available from their original providers[@edge,cic]. The per-run result files generated in "
  "this study are available at [repository URL / DOI to be added].")
H("Code availability", 1)
P("All scripts used to prepare the data, run the experiments and build the tables and figures are available at [repository URL / DOI "
  "to be added] (file overview in Supplementary Table S7).")

H("References", 1)
for i, k in enumerate(CITED, 1):
    p = P(f"{i}. {REFS[k]}", align=None, size=10, after=2, spacing=1.15)
    p.paragraph_format.left_indent = Cm(0.7); p.paragraph_format.first_line_indent = Cm(-0.7)
unused = [k for k in REFS if k not in CITED]
if unused:
    print("WARNING unused refs:", unused)

H("Acknowledgements", 1)
P("[To be completed.]")
H("Author contributions", 1)
P("H.I.A. conceived the study, wrote the code, ran the experiments, analysed the results and wrote the manuscript.")
H("Competing interests", 1)
P("The author declares no competing interests.")
H("Additional information", 1)
P("**Supplementary information** accompanies this paper (Supplementary Tables S1–S11 and Supplementary Fig. S1).")

out_main = "paper/FL_IIoT_Attack_Cost_SciRep.docx"
doc.save(out_main)

# =================================================================================== SUPPLEMENTARY
new_doc()
P("Supplementary Information", bold=True, align="c", size=16, after=4)
P(TITLE, italic=True, align="c", size=11, after=2)
P("Hassan I. Ahmed", align="c", size=10, after=10)

fb = pd.read_csv("poc_full2/fl_break_points.csv")
rows = []
for p in PARTS:
    for a in ["fedavg", "median", "trimmed", "krum"]:
        def g(att, col):
            return fb[(fb.attack == att) & (fb.partition == p) & (fb.aggregator == a)][col].iloc[0]
        rows.append([PLAB[p], ALAB[a], f"{g('labelflip','base_f1'):.3f}", g("labelflip", "break_recall"), g("labelflip", "break_f1"),
                     g("signflip", "break_recall"), g("signflip", "break_f1")])
caption("Supplementary Table S1", "**Static attacks on Edge-IIoTset, 5 seeds.** Attack cost (clients of 10; median [min..max]; “>5” = not "
        "broken with up to 5) under the recall criterion (recall < 0.8) and the macro-F1 criterion (drop > 0.10). Base F1 = clean macro-F1.")
table(["Partition", "Aggregator", "Base F1", "Label-flip recall", "Label-flip F1", "Sign-flip recall", "Sign-flip F1"], rows,
      widths=[1.9, 2.3, 1.6, 2.7, 2.7, 2.7, 2.7], size=8)
figure("paper/fig_static_vs_adaptive.png", 15.0, "Supplementary Figure S1",
       "**Static label-flip versus adaptive attack cost on Edge-IIoTset** (macro-F1 criterion for the static attack).")

rows = [[PLAB[p], ALAB[a], win_str(W_E, p, a), win_str(W_C, p, a)] for p in PARTS for a in AGGS]
caption("Supplementary Table S2", "**Strategies that break the model at the minimum cost**, counted over seeds (several strategies can tie in a "
        "seed; top three shown; “of n” = seeds in which the model was broken).")
table(["Partition", "Aggregator", "Edge-IIoTset", "CIC-IoT2023"], rows, widths=[2.0, 2.5, 5.8, 5.8], size=8.5)

dn = pd.read_csv("defense_out/defense_vs_nodefence.csv")
rows = [[PLAB[r.partition], ALAB[r.aggregator], r.n, r.no_defence, r.with_clip, f"{r.mean_gain:+.1f}", f"{r.seeds_better} / {r.seeds_worse}"]
        for r in dn.itertuples()]
caption("Supplementary Table S3", "**Effect of norm clipping on attack cost on Edge-IIoTset** (paired per seed). Gain = cost with clipping "
        "minus cost without; positive = the defence helps.")
table(["Partition", "Aggregator", "Seeds", "No defence", "With clip", "Mean gain", "Better / worse seeds"], rows,
      widths=[2.0, 2.4, 1.3, 2.5, 2.5, 1.8, 3.1], size=8.5)

rows = []
for p in PARTS:
    for a in AGGS:
        sub = C_ps[(C_ps.partition == p) & (C_ps.aggregator == a)].set_index("seed").cost_any.astype(float)
        sub2 = CD_ps[(CD_ps.partition == p) & (CD_ps.aggregator == a + "_clip")].set_index("seed").cost_any.astype(float)
        idx = sub.index.intersection(sub2.index)
        d = (sub2[idx] - sub[idx]).to_numpy()
        rows.append([PLAB[p], ALAB[a], len(idx), cost_str(C_ps[C_ps.seed.isin(idx)], p, a), cost_str(CD_ps, p, a + "_clip"),
                     f"{d.mean():+.1f}" if len(d) else "–", f"{int((d > 0).sum())} / {int((d < 0).sum())}" if len(d) else "–",
                     win_str(W_CD, p, a + "_clip")])
caption("Supplementary Table S4", f"**Norm clipping on CIC-IoT2023** ({cd_seeds} seeds; balanced-detection and macro-F1 criteria). Costs are "
        "median [min..max]; the undefended cost is restricted to the same seeds; gain = cost with clipping minus cost without.")
table(["Partition", "Aggregator", "Seeds", "No defence", "With clip", "Mean gain", "Better / worse", "Winning strategies (with clip)"], rows,
      widths=[1.7, 2.0, 1.1, 2.2, 2.2, 1.5, 1.8, 4.3], size=8)

rows = []
for p in PARTS:
    for a, lab in [("cosw", "Cos-weight"), ("cosdrop", "Cos-drop")]:
        r_ = CC_ci[(CC_ci.partition == p) & (CC_ci.aggregator == a)].iloc[0]
        rows.append([PLAB[p], lab, int(r_.n_seeds), cost_str(CC_ps, p, a), f"{r_.ci_low}–{r_.ci_high}",
                     cost_str(C_ps, p, "median"), cost_str(C_ps, p, "trimmed"), win_str(W_CC, p, a)])
caption("Supplementary Table S5", f"**Cosine defences on CIC-IoT2023** ({cc_seeds} seeds; balanced-detection and macro-F1 criteria). Cost is "
        "median [min..max] with the bootstrap 95% CI of the median; the undefended median and trimmed-mean costs (10 seeds, main Table 3) "
        "are repeated for reference.")
table(["Partition", "Defence", "Seeds", "Cost", "95% CI", "Median (no defence)", "Trimmed (no defence)", "Winning strategies"], rows,
      widths=[1.6, 2.0, 1.1, 2.0, 1.5, 2.3, 2.3, 4.0], size=8)

caption("Supplementary Table S6", "**Data used in the federated experiments** (after cleaning and class capping).")
table(["", "Edge-IIoTset", "CIC-IoT2023"],
      [["Rows used", f"{stats['edge']['rows']:,}", f"{stats['cic']['rows']:,}"],
       ["Features", stats["edge"]["features"], stats["cic"]["features"]],
       ["Classes (incl. Normal)", stats["edge"]["classes"], stats["cic"]["classes"]],
       ["Class size min / max", f"{stats['edge']['min_class']:,} / {stats['edge']['max_class']:,}", f"{stats['cic']['min_class']:,} / {stats['cic']['max_class']:,}"],
       ["Normal share", f"{100*stats['edge']['normal']/stats['edge']['rows']:.1f}%", f"{100*stats['cic']['normal']/stats['cic']['rows']:.1f}%"],
       ["Train / test rows", f"{stats['edge']['train']:,} / {stats['edge']['test']:,}", f"{stats['cic']['train']:,} / {stats['cic']['test']:,}"],
       ["Split", "temporal, last 30% of each class", "temporal, last 30% of each class"]],
      widths=[4.5, 5.8, 5.8], size=9)

caption("Supplementary Table S7", "**Scripts and output folders.** Every run is appended to a CSV as soon as it finishes, and all scripts skip "
        "finished runs when restarted.")
table(["Item", "Content"],
      [["poc_edge_iiot.py", "data loading, cleaning, class capping, temporal split, static label-flip / sign-flip grid, leakage experiment"],
       ["adaptive_attack.py", "adaptive attacker, aggregators (fedavg / median / trimmed / krum, *_clip, cosw, cosdrop), cost summary"],
       ["prepare_ciciot.py", "builds the class-capped CIC-IoT2023 sample from the per-capture CSV files"],
       ["cic_winners_ci.py", "bootstrap CI and winning strategies for the CIC-IoT2023 runs"],
       ["threshold_sensitivity.py", "threshold sensitivity of the adaptive attack cost (Supplementary Tables S8 and S9)"],
       ["mechanism_analysis.py", "spread of honest updates versus attack cost (Supplementary Tables S10 and S11)"],
       ["disp_edge/, disp_cic/", "clean runs with honest-update dispersion logs, 10 seeds (adaptive_attack.py --dispersion)"],
       ["run_*.ps1", "experiment launchers"],
       ["poc_full2/", f"static grid ({n_runs['static']:,} runs)"],
       ["adaptive_out/", f"Edge-IIoTset adaptive attacker, 10 seeds ({n_runs['adaptive_out']:,} runs)"],
       ["defense_out/", f"Edge-IIoTset norm clipping, 5 seeds ({n_runs['defense_out']:,} runs)"],
       ["cos_out/", f"Edge-IIoTset cosine defences, 5 seeds ({n_runs['cos_out']:,} runs)"],
       ["cic_adaptive/", f"CIC-IoT2023 adaptive attacker, 10 seeds ({n_runs['cic_adaptive']:,} runs)"],
       ["cic_defense/", f"CIC-IoT2023 norm clipping, 5 seeds ({n_runs['cic_defense']:,} runs)"],
       ["cic_cos/", f"CIC-IoT2023 cosine defences, 5 seeds ({n_runs['cic_cos']:,} runs)"],
       ["cic_hetdef/", f"CIC-IoT2023 centred clipping and bucketing, 5 seeds ({n_runs['cic_hetdef']:,} runs)"],
       ["paper/", "manuscript, figures and build scripts (make_figures.py, build_paper_srep.py)"]],
      widths=[4.0, 12.0], size=8.5)

def ts_cell(ds, d, p, a):
    v = TS_C[(TS_C.dataset == ds) & (TS_C["drop"] == d) & (TS_C.partition == p) & (TS_C.aggregator == a)]
    if ds == "Edge-IIoTset":
        v = v[v.floor == 0.8]
    r = v.iloc[0]
    return r.cost + ("*" if r.n_unresolved else "")


rows = [[PLAB[p], ALAB[a]] + [ts_cell(ds, d, p, a) for ds in ["Edge-IIoTset", "CIC-IoT2023"] for d in [0.05, 0.10, 0.15, 0.20]]
        for p in PARTS for a in AGGS]
caption("Supplementary Table S8", "**Adaptive attack cost under different break thresholds**, recomputed from the runs of the adaptive "
        "study (10 seeds; median [min..max]). d = required drop in macro-F1 and, on CIC-IoT2023, in balanced detection; the Edge-IIoTset "
        "recall floor is 0.8. The paper's setting is d = 0.10. * = at least one seed unresolved: not broken by the last m that was run, "
        "because runs stopped once both criteria were broken at the original thresholds; counted as “>5”. With d = 0.10 and recall floors "
        "of 0.7 and 0.9 the Edge-IIoTset costs are unchanged except for the median at Dir(0.1) (3 and 2 clients instead of 2.5).")
table(["Partition", "Aggregator", "Edge 0.05", "Edge 0.10", "Edge 0.15", "Edge 0.20", "CIC 0.05", "CIC 0.10", "CIC 0.15", "CIC 0.20"], rows,
      widths=[1.5, 1.8, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6], size=7.5)

rows = []
for ds in ["Edge-IIoTset", "CIC-IoT2023"]:
    for a in ["median", "trimmed"]:
        for q in ["dir0.5", "dir0.1"]:
            row = [ds, ALAB[a], f"IID vs {PLAB[q]}"]
            for d in [0.05, 0.10, 0.15, 0.20]:
                v = TS_P[(TS_P.dataset == ds) & (TS_P.aggregator == a) & (TS_P.comparison == f"iid vs {q}") & (TS_P["drop"] == d)]
                if ds == "Edge-IIoTset":
                    v = v[v.setting.str.contains("floor 0.8")]
                r = v.iloc[0]
                row.append(f"{r.cheaper}/{r.equal}/{r.costlier} ({r.p:.2g})" + ("*" if r.n_unresolved else ""))
            rows.append(row)
caption("Supplementary Table S9", "**Paired IID versus non-IID comparison under different break thresholds d** (10 seeds). Entries are "
        "the number of seeds in which the non-IID cost is lower / equal / higher than the IID cost, with the two-sided exact sign-test "
        "p-value. * = the comparison includes unresolved seeds (Supplementary Table S8).")
table(["Dataset", "Aggregator", "Comparison", "d = 0.05", "d = 0.10", "d = 0.15", "d = 0.20"], rows,
      widths=[2.4, 2.1, 2.6, 2.3, 2.3, 2.3, 2.3], size=7.5)

rows = []
for _, r in MECH_B.iterrows():
    rows.append([r.dataset, ALAB[r.aggregator], PLAB[r.partition], f"{r.clean_f1:.3f}", fmt(r.cost_median),
                 r.rel_spread_last10, r.pair_cos_last10, r.coord_snr_last10])
caption("Supplementary Table S10", "**Spread of honest updates in the clean (m = 0) runs** (10 seeds; median [min..max], averaged over the "
        "last 10 rounds). Relative spread = mean ‖hᵢ − h̄‖ / ‖h̄‖; pairwise cosine = mean cosine similarity between honest updates; "
        "coordinate spread = median over coordinates of s.d./|h̄|. Attack cost = median adaptive cost from main Table 3.")
table(["Dataset", "Aggregator", "Partition", "Clean F1", "Attack cost", "Relative spread", "Pairwise cosine", "Coordinate spread"], rows,
      widths=[2.3, 2.0, 1.6, 1.4, 1.4, 2.6, 2.8, 2.6], size=7.5)

MLAB = {"rel_spread_last10": "Relative spread", "pair_cos_last10": "Pairwise cosine", "coord_snr_last10": "Coordinate spread"}
rows = [[r.dataset, ALAB[r.aggregator], MLAB[r.metric], r.n, f"{r.spearman:+.2f}".replace("-", "−"), f"{max(r.perm_p, 1e-4):.4f}", r.within_partition.replace("-", "−")]
        for r in MECH_C.itertuples()]
caption("Supplementary Table S11", "**Rank correlation between the spread of honest updates and the adaptive attack cost.** Spearman ρ over "
        "all seed × partition cells, two-sided permutation p-value (10,000 permutations), and ρ within each partition (nan = cost constant "
        "within that partition). A negative ρ for spread (positive for cosine) means more dispersed honest updates go with a cheaper attack.")
table(["Dataset", "Aggregator", "Measure", "n", "ρ", "p", "Within partition"], rows,
      widths=[2.3, 2.0, 2.6, 0.9, 1.2, 1.4, 5.6], size=7.5)

out_sup = "paper/FL_IIoT_Attack_Cost_SciRep_Supplementary.docx"
doc.save(out_sup)
print("saved", out_main, out_sup, "refs:", len(CITED))
