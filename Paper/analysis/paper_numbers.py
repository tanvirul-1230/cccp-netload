"""
Every number, table cell and figure datum used in the paper.

Reads the protocol-v3 runs in Additional_Analyses/final3/{v3_tree, v3_mlp, v3_tree_clip}
(training / tune / development / calibration / test = 50/10/10/10/20 %; gains and
steps selected on development events in the development split) and the tables of
final_stats2.py for those runs. The unclipped-ACI and household-scale values are
recomputed here.

Writes
  ../numbers.tex                         LaTeX macros used in main.tex
  ../tables/table_results.tex            Table II body
  ../tables/table_ablation.tex           Table III body
  ../figures/data/*.csv                  data behind Figs. 2-4
  ../verification_code/paper_numbers_output.txt
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
AA = os.path.abspath(os.path.join(PAPER, "..", "Additional_Analyses"))
sys.path.insert(0, AA)
import her_shift as hs      # noqa: E402
import engine3 as e3        # noqa: E402
from scipy import stats     # noqa: E402

F2 = os.path.join(AA, "final3")
ZIP = os.environ.get("HEMS_ZIP") or next(p for p in (os.path.join(AA, "..", "data", "hems_data.zip"),
                            os.path.join(AA, "..", "Conference", "Conference",
                                         "HEMS_CP_Paper_COMPLETE_Day7_Backup.zip")) if os.path.exists(p))
HOUSES = ["H3", "H4", "H6"]
HOMES = {"H3": "residential3", "H4": "residential4", "H6": "residential6"}
STEPS = {"15min": 1, "30min": 2, "60min": 4}
ALPHA, NB = 0.10, 4000
for d in ["tables", os.path.join("figures", "data"), "verification_code"]:
    os.makedirs(os.path.join(PAPER, d), exist_ok=True)
LOG = open(os.path.join(PAPER, "verification_code", "paper_numbers_output.txt"), "w", encoding="utf-8")
MAC = {}
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def mac(name, value):
    assert name.isalpha(), name
    assert name not in MAC, name
    MAC[name] = value


def f3(x):
    return f"{x:.3f}"


def s3(x):
    return f"{x:+.3f}".replace("-", "$-$")


def pct(x, signed=True, nd=1):
    v = f"{100 * x:+.{nd}f}" if signed else f"{100 * x:.{nd}f}"
    return v.replace("-", "$-$") + "\\%"


def ci3(lo, hi):
    return f"[{s3(lo)}, {s3(hi)}]"


def cipct(lo, hi):
    return f"[{pct(lo)}, {pct(hi)}]"


# ----------------------------------------------------------------------
def load(tag, sel_tag=None):
    D = os.path.join(F2, tag)
    sel = json.load(open(os.path.join(F2, sel_tag or tag, "selection.json")))["selected"]
    E = pd.read_csv(os.path.join(D, "event_rows_test.csv.gz"))
    E["cl"] = E.Household + "|" + E.Day.astype(str)
    E["isr"] = E.IS_in / E.R
    CM = pd.read_csv(os.path.join(D, "clean_metrics_test.csv"))
    return dict(E=E, CM=CM, SEL=sel, D=D)


RUNS = {"tree": load("v3_tree"), "clip": load("v3_tree_clip", "v3_tree")}
if os.path.exists(os.path.join(F2, "v3_mlp", "event_rows_test.csv.gz")):
    RUNS["mlp"] = load("v3_mlp")
T = RUNS["tree"]
SEL = T["SEL"]
MAIN = {"SCP": "SCP", "LACP": "LACP", "ACItuned": "ACI|tuned", "ACI": SEL["ACI"], "CCCP": SEL["CC"],
        "CCLACP": SEL["CCLACP"], "ACICC": SEL["ACICC"], "WM": "WM-CCLACP",
        "ACIL": SEL["ACIL"], "ACICCL": SEL["ACICCL"], "ROLLSCP": "ROLL|SCP", "ROLLLACP": "ROLL|LACP",
        "ROLLCCLACP": SEL["ROLLCCL"]}
rng = np.random.default_rng(2026)


def shift(E, es="extended", types=None):
    X = E[(E.EventSet == es) & (E.Condition == "shift")]
    return X if types is None else X[X.ShiftType.isin(types)]


def csum(X, m):
    return X[X.Method == m].groupby("cl")[["isr", "N_in", "Cov_in", "W_in"]].sum()


def nis(X, m):
    s = csum(X, m)
    return s.isr.sum() / s.N_in.sum()


def picp(X, m):
    s = csum(X, m)
    return s.Cov_in.sum() / s.N_in.sum()


def rel(X, a, b):
    """Per-household relative change of inside-event nIS, a vs b; mean over
    households with a stratified cluster bootstrap (household-days)."""
    per, bs = {}, np.zeros(NB)
    for h in HOUSES:
        Xh = X[X.Household == h]
        A, B = csum(Xh, a), csum(Xh, b)
        idx = A.index.intersection(B.index)
        na, nb = A.loc[idx].isr.values, B.loc[idx].isr.values
        per[h] = na.sum() / nb.sum() - 1
        ix = rng.integers(0, len(idx), (NB, len(idx)))
        bs += na[ix].sum(1) / nb[ix].sum(1) - 1
    bs /= len(HOUSES)
    hl = {}
    for h in HOUSES:
        Xh = X[X.Household == h]
        A, B = csum(Xh, a), csum(Xh, b)
        idx = A.index.intersection(B.index)
        na, nb = A.loc[idx].isr.values, B.loc[idx].isr.values
        ix = rng.integers(0, len(idx), (NB, len(idx)))
        b_ = na[ix].sum(1) / nb[ix].sum(1) - 1
        hl[h] = (np.percentile(b_, 2.5), np.percentile(b_, 97.5))
    return per, float(np.mean(list(per.values()))), np.percentile(bs, 2.5), np.percentile(bs, 97.5), hl


def paired(X, a, b):
    A, B = csum(X, a), csum(X, b)
    idx = A.index.intersection(B.index)
    na, nb, n = A.loc[idx].isr.values, B.loc[idx].isr.values, A.loc[idx].N_in.values
    ix = rng.integers(0, len(idx), (NB, len(idx)))
    bs = (na[ix].sum(1) - nb[ix].sum(1)) / n[ix].sum(1)
    return (na.sum() - nb.sum()) / n.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def clean_house(CM, m):
    g = CM[CM.Method == m]
    return {h: (g[g.Household == h].nIS * g[g.Household == h].N).sum() / g[g.Household == h].N.sum()
            for h in HOUSES}


def clean_pooled(CM, m, col):
    g = CM[CM.Method == m]
    return (g[col] * g.N).sum() / g.N.sum()


def clean_rel(CM, a, b):
    A, B = clean_house(CM, a), clean_house(CM, b)
    per = {h: A[h] / B[h] - 1 for h in HOUSES}
    return per, float(np.mean(list(per.values())))


def logw(E, es="extended", st="PV"):
    X = E[(E.EventSet == es) & (E.ShiftType == st)]
    key = ["Household", "Horizon", "EventID", "cl", "Severity", "DurationMin", "Method"]
    P = X.pivot_table(index=key, columns="Condition", values="Worig", aggfunc="first").dropna()
    P["lr"] = np.log(P["shift"] / P["placebo"])
    return P.reset_index()


def boot_mean(cl, v):
    codes, u = pd.factorize(pd.Series(np.asarray(cl)))
    s = np.bincount(codes, weights=v, minlength=len(u))
    c = np.bincount(codes, minlength=len(u)).astype(float)
    ix = rng.integers(0, len(u), (NB, len(u)))
    b = s[ix].sum(1) / c[ix].sum(1)
    return s.sum() / c.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)


# ======================================================================
# 1. setup
# ======================================================================
E = T["E"]
X_all = shift(E)
n_ext = X_all.drop_duplicates(["Household", "EventID"]).shape[0]
n_frz = shift(E, "frozen").drop_duplicates(["Household", "EventID"]).shape[0]
n_dev = json.load(open(os.path.join(T["D"], "selection.json")))["n_dev_events"]
mac("nExt", f"{n_ext:,}".replace(",", "{,}"))
mac("nFrozen", str(n_frz))
mac("nDev", str(n_dev))
mac("nPerHouse", str(n_ext // 3))
say("events: extended", n_ext, "frozen", n_frz, "development", n_dev)

z = zipfile.ZipFile(ZIP)
split_rows, scale_rows = [], []
meta = pd.read_csv(os.path.join(T["D"], "task_meta_test.csv"))
for h in HOUSES:
    df = hs.regularize_15min(pd.read_csv(io.BytesIO(z.read(f"data/processed/{HOMES[h]}_clean.csv"))))
    df["net_load_kw"] = df["load_kw"] - df["pv_kw"]
    tr_end, tu_end, dv_end, ca_end = e3.boundaries3(df)
    ev = pd.read_csv(os.path.join(AA, "final", "events", f"{h}_extended_events.csv"))
    ev["start"] = pd.to_datetime(ev["start"], utc=True)
    loss = []
    for _, e in ev[ev.shift_type.isin(["PV", "COMBINED"])].iterrows():
        stop = hs.event_stop_time(e)
        p = df.loc[(df.index >= e["start"]) & (df.index < stop), "pv_kw"]
        loss.append(float(e["severity"]) * float(p.mean()))
    qs = meta[meta.Household == h].set_index("Horizon").q_SCP
    ratio = {hz: np.mean(loss) / qs[hz] for hz in STEPS}
    test = df[df.index > ca_end]
    scale_rows.append(dict(Household=h, loss=np.mean(loss), **{f"ratio_{k}": v for k, v in ratio.items()},
                           pv99_over_R=float(test.pv_kw.quantile(0.99) / (test.net_load_kw.max() - test.net_load_kw.min())),
                           net_min=float(test.net_load_kw.min()), pv_max=float(test.pv_kw.max())))
    split_rows.append(dict(Household=h, start=df.index.min(), train_end=tr_end, tune_end=tu_end, dev_end=dv_end,
                           cal_end=ca_end, end=df.index.max()))
SC = pd.DataFrame(scale_rows).set_index("Household")
SP = pd.DataFrame(split_rows).set_index("Household")
say(SC.round(3).to_string())
say(SP.to_string())
for h in HOUSES:
    L = "HThree HFour HSix".split()[HOUSES.index(h)]
    lo, hi = SC.loc[h, ["ratio_60min", "ratio_15min"]]
    mac(f"ratio{L}", f"{lo:.1f}--{hi:.1f}")
    mac(f"pvR{L}", f"{SC.loc[h, 'pv99_over_R']:.2f}")
fmt_m = "%b %Y"
for h in HOUSES:
    L = "HThree HFour HSix".split()[HOUSES.index(h)]
    mac(f"dev{L}", f"{SP.loc[h, 'tune_end'].strftime(fmt_m)}--{SP.loc[h, 'dev_end'].strftime(fmt_m)}")
    mac(f"cal{L}", f"{SP.loc[h, 'dev_end'].strftime(fmt_m)}--{SP.loc[h, 'cal_end'].strftime(fmt_m)}")
    mac(f"test{L}", f"{SP.loc[h, 'cal_end'].strftime(fmt_m)}--{SP.loc[h, 'end'].strftime(fmt_m)}")
    mac(f"data{L}", f"{SP.loc[h, 'start'].strftime(fmt_m)}--{SP.loc[h, 'end'].strftime(fmt_m)}")

# ======================================================================
# 2. Table II: normal operation and inside-event accuracy
# ======================================================================
CM = T["CM"]
rows = []
for key, m in MAIN.items():
    r = dict(key=key, method=m)
    if m in set(CM.Method):
        r["cPICP"] = clean_pooled(CM, m, "PICP")
        r["cNIS"] = clean_pooled(CM, m, "nIS")
        r["cRel"] = clean_rel(CM, m, "LACP")[1]
    r["iPICP"] = picp(X_all, m)
    for h in HOUSES:
        r[f"i{h}"] = nis(X_all[X_all.Household == h], m)
    r["iPool"] = nis(X_all, m)
    if m != "LACP":
        per, mean, lo, hi, _ = rel(X_all, m, "LACP")
        r.update(iRel=mean, iRelLo=lo, iRelHi=hi)
    else:
        r.update(iRel=0.0, iRelLo=np.nan, iRelHi=np.nan)
    # held-out stress sets: complete PV trip (s = 1) and 3.7 kW load step
    for es, col in [("trip", "trip"), ("ev", "ev")]:
        Xs = shift(E, es)
        r[f"{col}PICP"] = picp(Xs, m)
        if m != "LACP":
            _, mean_s, lo_s, hi_s, _ = rel(Xs, m, "LACP")
            r.update({col: mean_s, f"{col}Lo": lo_s, f"{col}Hi": hi_s})
        else:
            r.update({col: 0.0, f"{col}Lo": np.nan, f"{col}Hi": np.nan})
    rows.append(r)
TB = pd.DataFrame(rows).set_index("key")
say("\nTable II data\n", TB.round(4).to_string())


def gpar(m, p):
    return m.split(f"|{p}")[1].split("|")[0]


NAMES = {"SCP": "SCP", "LACP": "LACP", "ACItuned": "ACI, reference $\\gamma$ (tune coverage)",
         "ACI": f"ACI, $\\gamma={gpar(SEL['ACI'], 'g')}$",
         "CCCP": f"CC-CP, $\\beta={gpar(SEL['CC'], 'b')}$",
         "CCLACP": f"CC-LACP, $\\beta={gpar(SEL['CCLACP'], 'b')}$",
         "ACICC": (f"ACI on CC-CP, $\\beta={gpar(SEL['ACICC'], 'b')}$, $\\gamma={gpar(SEL['ACICC'], 'g')}$"
                   if "|b" in SEL["ACICC"] else f"ACI on CC-CP, $\\beta=0$, $\\gamma={gpar(SEL['ACICC'], 'g')}$"),
         "ACIL": f"ACI on LACP, $\\gamma={gpar(SEL['ACIL'], 'g')}$",
         "ACICCL": (f"ACI on CC-LACP, $\\beta={gpar(SEL['ACICCL'], 'b')}$, $\\gamma={gpar(SEL['ACICCL'], 'g')}$"
                    if "|b" in SEL["ACICCL"] else f"ACI on CC-LACP, $\\beta=0$, $\\gamma={gpar(SEL['ACICCL'], 'g')}$"),
         "ROLLSCP": "SCP", "ROLLLACP": "LACP",
         "ROLLCCLACP": (f"CC-LACP, $\\beta={gpar(SEL['ROLLCCL'], 'b')}$" if "|b" in SEL["ROLLCCL"]
                        else "CC-LACP, $\\beta=0$"),
         "WM": "Width-matched LACP$^{\\dagger}$"}
GROUPS = [("Static quantile, calibration split", ["SCP", "LACP", "CCCP", "CCLACP"]),
          ("Rolling window, latest 2000 scores", ["ROLLSCP", "ROLLLACP", "ROLLCCLACP"]),
          ("Adaptive level (ACI), latest 2000 scores",
           ["ACItuned", "ACI"] + ([] if SEL["ACICC"] == SEL["ACI"] else ["ACICC"]) + ["ACIL", "ACICCL"]),
          ("Control", ["WM"])]
cand = [k for _, ks in GROUPS for k in ks if k != "WM"]
best_in = {h: TB.loc[cand, f"i{h}"].min() for h in HOUSES}
lines = []
NCOL = 10
for gname, keys in GROUPS:
    lines.append(f"\\multicolumn{{{NCOL}}}{{|l|}}{{\\emph{{{gname}}}}}\\\\")
    for key in keys:
        r = TB.loc[key]
        c1 = "--" if pd.isna(r.get("cPICP")) else f3(r.cPICP)
        c2 = "--" if pd.isna(r.get("cNIS")) else f3(r.cNIS)
        hcells = []
        for h in HOUSES:
            v = f3(r[f"i{h}"])
            hcells.append(f"\\textbf{{{v}}}" if key != "WM" and v == f3(best_in[h]) else v)   # ties at 3 dp
        rl = "0" if key == "LACP" else pct(r.iRel)
        st = ["0" if key == "LACP" else pct(r[c]) for c in ("trip", "ev")]
        if key == "CCLACP":
            lines.append("\\rowcolor{black!6}")
        lines.append(f"\\quad {NAMES[key]} & {c1} & {c2} & {f3(r.iPICP)} & " + " & ".join(hcells)
                     + f" & {rl} & {st[0]} & {st[1]}\\\\")
    lines.append("\\hline")
lines = lines[:-1]
HEAD2 = [r"\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}", r"\hline",
         r"\multirow{2}{*}{Method} & \multicolumn{2}{c|}{Normal operation} & \multicolumn{5}{c|}{Inside disturbances}"
         r" & \multicolumn{2}{c|}{Held-out, vs.\ LACP}\\",
         r"\cline{2-10}",
         r" & Coverage & Score & Coverage & Score, H3 & Score, H4 & Score, H6 & vs.\ LACP & PV trip & 3.7~kW step\\", r"\hline"]
TAIL2 = [r"\hline", r"\end{tabular}"]
open(os.path.join(PAPER, "tables", "table_results.tex"), "w", encoding="utf-8").write(
    "\n".join(HEAD2 + lines + TAIL2) + "\n")

# text macros from Table II
for key in ["LACP", "CCLACP", "ACI", "CCCP", "SCP", "ACItuned", "ACIL", "ACICCL", "ROLLSCP", "ROLLLACP",
            "ROLLCCLACP", "ACICC"]:
    r = TB.loc[key]
    mac(f"clean{key}PICP", f3(r.cPICP))
    mac(f"clean{key}nIS", f3(r.cNIS))
    mac(f"in{key}PICP", f3(r.iPICP))
    for h in HOUSES:
        mac(f"in{key}{'HThree HFour HSix'.split()[HOUSES.index(h)]}", f3(r[f'i{h}']))
    mac(f"in{key}pool", f3(r.iPool))
    for col in ("trip", "ev"):
        mac(f"{col}{key}PICP", f3(r[f"{col}PICP"]))
        if key != "LACP":
            mac(f"{col}rel{key}", pct(r[col]))
            mac(f"{col}rel{key}CI", cipct(r[f"{col}Lo"], r[f"{col}Hi"]))
mac("inWMpool", f3(TB.loc["WM", "iPool"]))
for es, key in [("trip", "Trip"), ("ev", "EV")]:
    mac(f"n{key}", str(shift(E, es).drop_duplicates(["Household", "EventID"]).shape[0]))
mac("evabsCCLACP", pct(abs(TB.loc["CCLACP", "ev"]), signed=False))
mac("evabsACI", pct(abs(TB.loc["ACI", "ev"]), signed=False))
# the factor inside the recalibrated variants on the 3.7-kW steps
for a, b, tag in [("ROLLCCLACP", "ROLLLACP", "ROLL"), ("ACICCL", "ACIL", "ACIL")]:
    _, mean_f, lo_f, hi_f, _ = rel(shift(E, "ev"), MAIN[a], MAIN[b])
    mac(f"evfac{tag}", pct(mean_f))
    mac(f"evfac{tag}CI", cipct(lo_f, hi_f))
    say(f"  3.7-kW steps, {a} vs {b}: {pct(mean_f)} {cipct(lo_f, hi_f)}")

# ======================================================================
# 3. paired and relative comparisons (all disturbance types)
# ======================================================================
say("\nRelative inside-event nIS change (household mean) and pooled paired differences")
REL = {}
for a, b, tag in [("CCLACP", "LACP", "CCLACPvsLACP"), ("CCLACP", "ACI", "CCLACPvsACI"),
                  ("ACI", "LACP", "ACIvsLACP"), ("CCLACP", "WM", "CCLACPvsWM"),
                  ("CCCP", "LACP", "CCCPvsLACP"), ("CCCP", "ACI", "CCCPvsACI"),
                  ("ACICC", "ACI", "ACICCvsACI"), ("ACItuned", "ACI", "ACItunedvsACI"),
                  ("CCLACP", "ACIL", "CCLACPvsACIL"), ("CCLACP", "ROLLLACP", "CCLACPvsROLLLACP"),
                  ("ACICCL", "ACIL", "ACICCLvsACIL"), ("ROLLCCLACP", "ROLLLACP", "ROLLCCLACPvsROLLLACP"),
                  ("ACIL", "LACP", "ACILvsLACP"), ("ACICCL", "LACP", "ACICCLvsLACP"),
                  ("ROLLLACP", "LACP", "ROLLLACPvsLACP")]:
    per, mean, lo, hi, hl = rel(X_all, MAIN[a], MAIN[b])
    d, dlo, dhi = paired(X_all, MAIN[a], MAIN[b])
    REL[tag] = (per, mean, lo, hi, hl)
    say(f"  {tag:14s} " + "  ".join(f"{h} {pct(v)} {cipct(*hl[h])}" for h, v in per.items())
        + f" | mean {pct(mean)} {cipct(lo, hi)} | pooled diff {s3(d)} {ci3(dlo, dhi)}")
    mac(f"rel{tag}", pct(mean))
    mac(f"rel{tag}CI", cipct(lo, hi))
    mac(f"absrel{tag}", pct(abs(mean), signed=False))
    for h in HOUSES:
        L = "HThree HFour HSix".split()[HOUSES.index(h)]
        mac(f"rel{tag}{L}", pct(per[h]))
        mac(f"rel{tag}{L}CI", cipct(*hl[h]))
    mac(f"d{tag}", s3(d))
    mac(f"d{tag}CI", ci3(dlo, dhi))

# clean relative changes (household mean)
for a in ["CCLACP", "ACI", "CCCP", "ACItuned", "ACIL", "ACICCL", "ROLLLACP", "ROLLCCLACP", "ROLLSCP", "ACICC"]:
    per, mean = clean_rel(CM, MAIN[a], "LACP")
    say(f"  clean rel {a} vs LACP: " + "  ".join(f"{h} {pct(v)}" for h, v in per.items()) + f" mean {pct(mean)}")
    mac(f"cleanrel{a}", pct(mean))
    mac(f"abscleanrel{a}", pct(abs(mean), signed=False))
per, mean = clean_rel(CM, MAIN["CCLACP"], MAIN["ACI"])
mac("cleanrelCCLACPvsACI", pct(mean))

# coverage differences inside events
for a, b, tag in [("CCLACP", "LACP", "CCLACPvsLACP"), ("CCLACP", "ACI", "CCLACPvsACI")]:
    mac(f"dPICP{tag}", s3(TB.loc[a, "iPICP"] - TB.loc[b, "iPICP"]))

# Diebold-Mariano tests on the clean test period, per household and horizon: Newey-West
# variance with 24 lags (96 lags as a check), Holm over the nine tasks. Counts: tasks where
# the method has a lower / equal / higher interval score at the 5% level.
PFc = pd.read_csv(os.path.join(T["D"], "clean_per_forecast.csv.gz"))


def isc(g, m):
    lo, hi, y = g[f"{m}_lo"].values, g[f"{m}_hi"].values, g.y.values
    return (hi - lo) + (2 / ALPHA) * (lo - y) * (y < lo) + (2 / ALPHA) * (y - hi) * (y > hi)


def dm(a, b, lags):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    st = d.mean() / np.sqrt(var / n)
    return d.mean(), 2 * (1 - stats.norm.cdf(abs(st)))


def holm(p):
    p = np.asarray(p, float)
    o, m_, run, adj = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for r_, i in enumerate(o):
        run = max(run, (m_ - r_) * p[i])
        adj[i] = min(run, 1.0)
    return adj


for a, b in [("CCLACP", "LACP"), ("CCLACP", "ACI"), ("CCCP", "ACI"), ("CCCP", "LACP"), ("CCLACP", "ACIL"),
             ("CCLACP", "ROLLLACP"), ("ACICCL", "ACIL"), ("ROLLCCLACP", "ROLLLACP")]:
    for lags in (24, 96):
        res = []
        for (h, hz), g in PFc.groupby(["Household", "Horizon"], sort=False):
            g = g.sort_values("t")
            res.append(dm(isc(g, MAIN[b]), isc(g, MAIN[a]), lags))
        d_ = np.array([x[0] for x in res])
        s_ = np.where(holm([x[1] for x in res]) < 0.05, np.sign(d_), 0)
        cnt = f"{(s_ > 0).sum()}/{(s_ == 0).sum()}/{(s_ < 0).sum()}"
        say(f"  DM {a} vs {b}, {lags} lags: lower/equal/higher score {cnt}")
        mac(f"dm{a}vs{b}" + ("" if lags == 24 else "Long"), cnt)

# load steps: spread of inside-event nIS over the main methods
XL = shift(E, types=["LOAD"])
ld = {k: nis(XL, m) for k, m in MAIN.items() if k != "WM"}
mac("loadMin", f3(min(ld.values())))
mac("loadMax", f3(max(ld.values())))
say("  load-step inside nIS:", {k: round(v, 3) for k, v in ld.items()})

# ======================================================================
# 4. PV loss: width response and accuracy per household (Fig. 2)
# ======================================================================
LW = logw(E)
FIGM = {"LACP": "LACP", "CC-LACP": MAIN["CCLACP"], "CC-CP": MAIN["CCCP"], "ACI": MAIN["ACI"]}
fig2 = []
for lab, m in FIGM.items():
    Lm = LW[LW.Method == m]
    for h in HOUSES + ["Pooled"]:
        g = Lm if h == "Pooled" else Lm[Lm.Household == h]
        p, lo, hi = boot_mean(g.cl.values, g.lr.values)
        fig2.append(dict(panel="a", method=lab, household=h, value=p, lo=lo, hi=hi,
                         narrower=float((g.lr < 0).mean())))
XP = shift(E, types=["PV"])
for lab, m in FIGM.items():
    if lab == "LACP":
        continue
    per, mean, lo, hi, hl = rel(XP, m, "LACP")
    for h in HOUSES:
        fig2.append(dict(panel="b", method=lab, household=h, value=per[h], lo=hl[h][0], hi=hl[h][1]))
    fig2.append(dict(panel="b", method=lab, household="Mean", value=mean, lo=lo, hi=hi))
F2D = pd.DataFrame(fig2)
F2D.to_csv(os.path.join(PAPER, "figures", "data", "fig2_pv_loss_households.csv"), index=False)
say("\nFig. 2 data\n", F2D.round(4).to_string(index=False))
a = F2D[F2D.panel == "a"].set_index(["method", "household"])
for lab, key in [("LACP", "LACP"), ("CC-LACP", "CCLACP"), ("CC-CP", "CCCP"), ("ACI", "ACI")]:
    mac(f"lw{key}", s3(a.loc[(lab, "Pooled"), "value"]))
    mac(f"lw{key}CI", ci3(a.loc[(lab, "Pooled"), "lo"], a.loc[(lab, "Pooled"), "hi"]))
    for h in HOUSES:
        L = "HThree HFour HSix".split()[HOUSES.index(h)]
        mac(f"lw{key}{L}", s3(a.loc[(lab, h), "value"]))
mac("narrowLACP", f"{100 * a.loc[('LACP', 'Pooled'), 'narrower']:.0f}\\%")
b = F2D[F2D.panel == "b"].set_index(["method", "household"])
for lab, key in [("CC-LACP", "CCLACP"), ("CC-CP", "CCCP"), ("ACI", "ACI")]:
    mac(f"pvrel{key}", pct(b.loc[(lab, "Mean"), "value"]))
    for h in HOUSES:
        L = "HThree HFour HSix".split()[HOUSES.index(h)]
        mac(f"pvrel{key}{L}", pct(b.loc[(lab, h), "value"]))

# paired log-width difference CC-LACP - LACP under PV loss, per household and per severity
Pw = LW.pivot_table(index=["Household", "Horizon", "EventID", "cl", "Severity"], columns="Method",
                    values="lr").dropna(subset=[MAIN["CCLACP"], "LACP"]).reset_index()
Pw["d"] = Pw[MAIN["CCLACP"]] - Pw["LACP"]
for h in HOUSES:
    g = Pw[Pw.Household == h]
    p, lo, hi = boot_mean(g.cl.values, g.d.values)
    L = "HThree HFour HSix".split()[HOUSES.index(h)]
    mac(f"lwdiff{L}", s3(p))
    mac(f"lwdiff{L}CI", ci3(lo, hi))
    say(f"  log width CC-LACP - LACP, PV loss, {h}: {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")
for s_, g in Pw.groupby("Severity"):
    p, lo, hi = boot_mean(g.cl.values, g.d.values)
    say(f"  log width CC-LACP - LACP, PV loss, severity {s_}: {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    mac("lwdiffSev" + {0.25: "Low", 0.5: "Mid", 0.75: "High"}[s_], s3(p))
g = LW[(LW.Method == "LACP") & (LW.Severity == 0.75)]
p, lo, hi = boot_mean(g.cl.values, g.lr.values)
mac("lwLACPhigh", s3(p))
mac("lwLACPhighCI", ci3(lo, hi))

# net adaptation at the 1.25 threshold (placebo-corrected), PV loss
for es in ["extended", "frozen"]:
    R3 = pd.read_csv(os.path.join(T["D"], "tables", f"C3_{es}_response.csv"))
    r = R3[(R3.Outcome == "Ad125") & (R3.ShiftType == "PV")].set_index("Method")
    pre = "net" if es == "extended" else "frz"
    for key in ["LACP", "CCLACP", "CCCP", "ACI", "ACItuned"]:
        m = MAIN[key]
        mac(f"{pre}{key}", s3(r.loc[m, "Net"]))
        mac(f"{pre}{key}CI", ci3(r.loc[m, "CI_lo"], r.loc[m, "CI_hi"]))
    mac(f"{pre}Persist", s3(r.loc[SEL["Persist"], "Net"]))
    D3 = pd.read_csv(os.path.join(T["D"], "tables", f"C3_{es}_response_differences.csv"))
    aci_lab = "ACI (= ACI-CC, beta 0 selected)" if SEL["ACICC"] == SEL["ACI"] else "ACI"   # final_stats2 label
    x = D3[(D3.Outcome == "Ad125") & (D3.Method == "CC-CP") & (D3.Baseline == aci_lab) & (D3.ShiftType == "PV")].iloc[0]
    mac(f"{pre}dCCCPvsACI", s3(x.Diff))
    mac(f"{pre}dCCCPvsACICI", ci3(x.CI_lo, x.CI_hi))
    say(f"  {es}: PV net CC-CP {r.loc[MAIN['CCCP'], 'Net']:.3f}, ACI {r.loc[MAIN['ACI'], 'Net']:.3f}, "
        f"diff {x.Diff:+.3f} [{x.CI_lo:+.3f}, {x.CI_hi:+.3f}]")
r = R3  # placebo rates (extended) for the text
R3e = pd.read_csv(os.path.join(T["D"], "tables", "C3_extended_response.csv"))
pl = R3e[(R3e.Outcome == "Ad125") & (R3e.ShiftType == "PV")].set_index("Method").Placebo
mac("placeboACI", f"{pl[MAIN['ACI']]:.2f}")
mac("placeboACItuned", f"{pl[MAIN['ACItuned']]:.2f}")

# severity sweep (Fig. 4 data): mean log width ratio under PV loss by severity
fig4 = []
for lab, m in FIGM.items():
    Lm = LW[LW.Method == m]
    for s_, g in Lm.groupby("Severity"):
        p, lo, hi = boot_mean(g.cl.values, g.lr.values)
        fig4.append(dict(method=lab, severity=s_, value=p, lo=lo, hi=hi))
F4D = pd.DataFrame(fig4)
F4D.to_csv(os.path.join(PAPER, "figures", "data", "fig4_pv_severity.csv"), index=False)
say("\nFig. 4 data\n", F4D.round(4).to_string(index=False))

# ======================================================================
# 5. parameter sweep (Fig. 3): household-mean relative change vs LACP
# ======================================================================
fams = {"ACI": [m for m in sorted(set(CM.Method)) if m.startswith("ACI|g")],
        "CC-CP": [m for m in sorted(set(CM.Method)) if m.startswith("CC|b")],
        "CC-LACP": [m for m in sorted(set(CM.Method)) if m.startswith("CCLACP|b")],
        "ACI-CC": [m for m in sorted(set(CM.Method)) if m.startswith("ACICC|b0.25|")]}
fig3 = []
for fam, cands in fams.items():
    for m in cands:
        par = float(m.split("|")[-1][1:])
        _, cr = clean_rel(CM, m, "LACP")
        per, ir, _, _, _ = rel(X_all, m, "LACP")
        fig3.append(dict(family=fam, method=m, param=par, clean_rel=cr, inside_rel=ir,
                         selected=m in SEL.values() and (fam != "ACI" or m == SEL["ACI"])))
for m in ["SCP", "ACI|tuned"]:
    _, cr = clean_rel(CM, m, "LACP")
    per, ir, _, _, _ = rel(X_all, m, "LACP")
    fig3.append(dict(family=m, method=m, param=np.nan, clean_rel=cr, inside_rel=ir, selected=False))
F3D = pd.DataFrame(fig3).sort_values(["family", "param"])
F3D.to_csv(os.path.join(PAPER, "figures", "data", "fig3_parameter_sweep.csv"), index=False)
say("\nFig. 3 data\n", F3D.round(4).to_string(index=False))

# ======================================================================
# 6. ablation of the consistency signal (Table III)
# ======================================================================
R3 = pd.read_csv(os.path.join(T["D"], "tables", "C3_extended_response.csv"))
ABL = [(r"Scale $1+\beta\tilde\rho_t$ (CC-CP)",
        [("None (SCP)", None), ("PV and load", "CC"), ("PV only", "PVonly"), ("Load only", "LoadOnly"),
         ("Load model, day lag only", "LF-load"), ("Both, day lag only", "LF-both"),
         ("Persistence residual", "Persist")]),
       (r"Scale $s(x_t)(1+\beta\tilde\rho_t)$ (CC-LACP)",
        [("None (LACP)", None), ("PV and load", "CCLACP"), ("PV only", "PVonlyL"), ("Persistence residual", "PersistL"),
         (r"Indicator $\mathbf{1}\{\tilde\rho_t>\tau\}$", "THL")])]
lines, abl = [], []
for gname, items in ABL:
    lines.append(f"\\multicolumn{{5}}{{|l|}}{{\\emph{{{gname}}}}}\\\\")
    for name, fam in items:
        m = SEL[fam] if fam else ("SCP" if "SCP" in name else "LACP")

        def net(st):
            r = R3[(R3.Outcome == "Ad125") & (R3.ShiftType == st) & (R3.Method == m)].iloc[0]
            star = "" if r.CI_lo <= 0 <= r.CI_hi else "$^{*}$"
            return r.Net, f"{s3(r.Net)}{star}"
        _, pvs = net("PV")
        _, lds = net("LOAD")
        if m == "LACP":
            ir_s = "0"
        else:
            _, ir, _, _, _ = rel(X_all, m, "LACP")
            ir_s = pct(ir)
        b_ = gpar(m, "b") if fam else "--"
        if m.startswith("THL|"):
            b_ = f"{b_}, $\\tau=q_{{{m.split('|')[1][1:]}}}$"
        lines.append(f"\\quad {name} & {b_} & {pvs} & {lds} & {ir_s}\\\\")
        if fam:
            abl.append(dict(signal=name, method=m, inside_rel=ir))
            mac("abl" + "".join(ch for ch in fam if ch.isalpha()), pct(ir))
    lines.append("\\hline")
lines = lines[:-1]
HEAD3 = [r"\begin{tabular}{|l|c|c|c|c|}", r"\hline",
         r"\multirow{2}{*}{Signal} & \multirow{2}{*}{$\beta$} & \multicolumn{2}{c|}{Net adaptation} & Score vs.\\",
         r"\cline{3-4}", r" & & PV loss & Load step & LACP\\", r"\hline"]
open(os.path.join(PAPER, "tables", "table_ablation.tex"), "w", encoding="utf-8").write(
    "\n".join(HEAD3 + lines + [r"\hline", r"\end{tabular}"]) + "\n")
say("\nTable III\n" + "\n".join(lines))

# ======================================================================
# 7. robustness: neural backbone, clipped load, frozen set
# ======================================================================
for tag in ["mlp", "clip"]:
    if tag not in RUNS:          # run still pending: placeholders, replaced on the next run
        for k in ["CCLACPvsLACP", "CCLACPvsLACPCI", "CCLACPvsACI", "CCLACPvsACICI", "narrow"]:
            mac(f"rob{tag}{k}", r"\textbf{??}")
        say(f"  robustness {tag}: run not available")
        continue
    R = RUNS[tag]
    Xr = shift(R["E"])
    sel = R["SEL"]
    per, mean, lo, hi, _ = rel(Xr, sel["CCLACP"], "LACP")
    per2, mean2, lo2, hi2, _ = rel(Xr, sel["CCLACP"], sel["ACI"])
    LWr = logw(R["E"])
    g = LWr[LWr.Method == "LACP"]
    p, plo, phi = boot_mean(g.cl.values, g.lr.values)
    mac(f"rob{tag}CCLACPvsLACP", pct(mean))
    mac(f"rob{tag}CCLACPvsLACPCI", cipct(lo, hi))
    mac(f"rob{tag}CCLACPvsACI", pct(mean2))
    mac(f"rob{tag}CCLACPvsACICI", cipct(lo2, hi2))
    mac(f"rob{tag}narrow", f"{100 * (g.lr < 0).mean():.0f}\\%")
    say(f"  robustness {tag}: CC-LACP vs LACP {pct(mean)} {cipct(lo, hi)} per {per}; vs ACI {pct(mean2)} "
        f"{cipct(lo2, hi2)} per {per2}; LACP PV log width {p:+.3f} [{plo:+.3f},{phi:+.3f}], narrower "
        f"{(g.lr < 0).mean():.2f}")
XF = shift(E, "frozen")
per, mean, lo, hi, _ = rel(XF, MAIN["CCLACP"], "LACP")
mac("robfrzCCLACPvsLACP", pct(mean))
mac("robfrzCCLACPvsLACPCI", cipct(lo, hi))
say(f"  frozen: CC-LACP vs LACP {pct(mean)} {cipct(lo, hi)} per {per}")

# ======================================================================
# 8. ACI at the selected step without the clip (clean test period)
# ======================================================================
def cq(scores, a):
    s = np.asarray(scores, float)
    k = min(max(int(np.ceil((len(s) + 1) * (1 - a))), 1), len(s))
    return float(np.partition(s, k - 1)[k - 1])


def aci(y, pred, init, g, h, clip):
    scores, a = list(np.asarray(init, float)), ALPHA
    lo, hi = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for t in range(len(y)):
        j = t - h
        if j >= 0:
            miss = int((y[j] < lo[j]) or (y[j] > hi[j]))
            a = a + g * (ALPHA - miss)
            if clip:
                a = float(np.clip(a, 0.001, 0.50))
            scores.append(abs(y[j] - pred[j]))
            scores = scores[-2000:]
        q = cq(scores, a)
        lo[t], hi[t] = pred[t] - q, pred[t] + q
    return lo, hi


PF = pd.read_csv(os.path.join(T["D"], "clean_per_forecast.csv.gz"))
INF = pd.read_csv(os.path.join(T["D"], "info_inputs.csv.gz"))
g_sel = float(SEL["ACI"].split("|g")[1])
acc = {True: [0, 0, 0], False: [0, 0, 0]}
err = 0.0
for (h, hz), g in PF.groupby(["Household", "Horizon"], sort=False):
    g = g.sort_values("t")
    init = INF[(INF.Household == h) & (INF.Horizon == hz) & (INF.split == "cal")].sort_values("t").abs_err.values
    y, p, Rr = g.y.values, g.pred.values, g.R.iloc[0]
    for clip in (True, False):
        lo, hi = aci(y, p, init, g_sel, STEPS[hz], clip)
        if clip:
            err = max(err, np.nanmax(np.abs(lo - g[f"{SEL['ACI']}_lo"].values)))
        w = hi - lo
        sc = w + (2 / ALPHA) * (lo - y) * (y < lo) + (2 / ALPHA) * (y - hi) * (y > hi)
        acc[clip][0] += ((y >= lo) & (y <= hi)).sum()
        acc[clip][1] += sc.sum() / Rr
        acc[clip][2] += len(y)
for clip in (True, False):
    c, s, n = acc[clip]
    say(f"  ACI gamma={g_sel} {'clipped' if clip else 'unclipped'}: PICP {c / n:.4f}  nIS {s / n:.4f}")
say(f"  reproduction error of stored clipped ACI: {err:.1e}")
assert err < 1e-5          # GPU (float32) predictions of the same model differ by ~1e-7 between calls
mac("unclipPICP", f3(acc[False][0] / acc[False][2]))
mac("unclipnIS", f3(acc[False][1] / acc[False][2]))

# ======================================================================
# 9. per-household selection (sensitivity) and misc
# ======================================================================
Cd = pd.read_csv(os.path.join(T["D"], "clean_metrics_dev.csv"))
Ed = pd.read_csv(os.path.join(T["D"], "event_rows_dev.csv.gz"))
Ed = Ed[Ed.Condition == "shift"]
import engine2 as e2        # noqa: E402
FAM = {"ACI": [f"ACI|g{x:g}" for x in e2.GAMMAS], "CCLACP": [f"CCLACP|b{x:g}" for x in e2.BETAS_LACP]}
for h in HOUSES:
    c = Cd[Cd.Household == h].groupby("Method").apply(lambda g: (g.nIS * g.N).sum() / g.N.sum())
    d = Ed[Ed.Household == h].groupby("Method").apply(lambda g: (g.IS_in / g.R).sum() / g.N_in.sum())
    for fam, cands in FAM.items():
        ok = [x for x in cands if c[x] <= 1.02 * c.loc[cands].min()]
        pick = min(ok, key=lambda x: (round(d[x], 10), cands.index(x)))
        L = "HThree HFour HSix".split()[HOUSES.index(h)]
        mac(f"hsel{fam}{L}", pick.split("|")[1][1:])
        say(f"  per-household selection {h} {fam}: {pick}")
mac("selACI", gpar(SEL["ACI"], "g"))
mac("selCC", gpar(SEL["CC"], "b"))
mac("selCCLACP", gpar(SEL["CCLACP"], "b"))
mac("selACICCb", gpar(SEL["ACICC"], "b") if "|b" in SEL["ACICC"] else "0")
mac("selACICCg", gpar(SEL["ACICC"], "g"))
mac("selACIL", gpar(SEL["ACIL"], "g"))
mac("selACICCLb", gpar(SEL["ACICCL"], "b") if "|b" in SEL["ACICCL"] else "0")
mac("selACICCLg", gpar(SEL["ACICCL"], "g"))
mac("selROLLCCL", gpar(SEL["ROLLCCL"], "b"))
# development normal-operation nIS of each ACI step relative to the best step (2% tolerance)
ST = pd.read_csv(os.path.join(T["D"], "selection_table.csv"))
ST = ST[ST.Family == "ACI"].copy()
ST["g"] = ST.Candidate.str.split("|g", regex=False).str[1].astype(float)
adm = sorted(ST[ST.Admissible.astype(str) == "True"].g)
assert len(adm) == 2 and gpar(SEL["ACI"], "g") in [f"{g:g}" for g in adm]
low = ST[ST.g < min(adm)].CleanRatio - 1
mac("devACIadm", " and ".join(f"{g:g}" for g in adm))
mac("devACIlowMin", pct(low.min(), signed=False))
mac("devACIlowMax", pct(low.max(), signed=False))
say(f"  development ACI: admissible steps {adm}; smaller steps {100 * low.min():.1f}-{100 * low.max():.1f}% "
    f"above the best without injection")
mac("selTHLb", gpar(SEL["THL"], "b"))
mac("selTHLq", SEL["THL"].split("|")[1][1:] if SEL["THL"].startswith("THL|") else "--")
say("selection:", SEL)

# leave-one-household-out: the rule applied to the development events of the other two
# households, the result evaluated on the held-out household
SJ = json.load(open(os.path.join(T["D"], "selection.json")))
LO = SJ["leave_one_household_out"]
lo_rel = {}
for h in HOUSES:
    m = LO[h]["CCLACP"]
    Xh = X_all[X_all.Household == h]
    A, B = csum(Xh, m), csum(Xh, "LACP")
    idx = A.index.intersection(B.index)
    lo_rel[h] = A.loc[idx].isr.sum() / B.loc[idx].isr.sum() - 1
    L = "HThree HFour HSix".split()[HOUSES.index(h)]
    mac(f"loho{L}", gpar(m, "b"))
    mac(f"lohoACI{L}", gpar(LO[h]["ACI"], "g"))
    mac(f"lohorel{L}", pct(lo_rel[h]))
    say(f"  LOHO {h}: CC-LACP {m}, ACI {LO[h]['ACI']}; inside nIS vs LACP {pct(lo_rel[h])}")
mac("lohorel", pct(np.mean(list(lo_rel.values()))))
for t_, d_ in SJ["sensitivity"].items():
    say(f"  selection with {t_}: CC-LACP {d_['CCLACP']}, ACI {d_['ACI']}, CC {d_['CC']}")

# ======================================================================
# 10. recovery step, every target inside the disturbance, upper-bound exceedance, placebo rates
# ======================================================================


def as_part(X, part):
    Y = X.copy()
    Y["isr"] = Y[f"IS_{part}"] / Y.R
    for c in ("N", "Cov", "W"):
        Y[f"{c}_in"] = Y[f"{c}_{part}"]
    return Y


XR, XT_ = as_part(X_all, "rec"), as_part(X_all, "tgt")
for key in ["CCLACP", "CCCP", "ACI", "ACIL", "ACICCL", "ROLLLACP", "ROLLCCLACP"]:
    for Y, tag in [(XR, "rec"), (XT_, "tgt")]:
        _, mean, lo, hi, _ = rel(Y, MAIN[key], "LACP")
        mac(f"{tag}rel{key}", pct(mean))
        mac(f"{tag}rel{key}CI", cipct(lo, hi))
        say(f"  {tag}: {key} vs LACP {pct(mean)} {cipct(lo, hi)}")
for key in ["LACP", "CCLACP", "CCCP", "ACI", "ACIL", "ACICCL", "ROLLLACP", "ROLLCCLACP", "SCP"]:
    mac(f"rec{key}PICP", f3(picp(XR, MAIN[key])))
    mac(f"tgt{key}PICP", f3(picp(XT_, MAIN[key])))
    g = X_all[X_all.Method == MAIN[key]]
    mac(f"exc{key}", f"{1000 * g.Exc_tgt.sum() / g.N_tgt.sum():.0f}")
    say(f"  {key}: recovery PICP {picp(XR, MAIN[key]):.3f}, all-target PICP {picp(XT_, MAIN[key]):.3f}, "
        f"mean upper exceedance {1000 * g.Exc_tgt.sum() / g.N_tgt.sum():.0f} W")
PL = R3e[(R3e.Outcome == "Ad125") & (R3e.ShiftType == "ALL")].set_index("Method").Placebo
for key in ["LACP", "CCLACP", "CCCP", "ACI", "ACItuned", "ACIL", "ACICCL", "ROLLLACP", "ROLLCCLACP", "SCP"]:
    mac(f"fa{key}", f"{PL[MAIN[key]]:.2f}")
    say(f"  placebo widening rate (1.25 x pre-event width), {key}: {PL[MAIN[key]]:.3f}")

# normal-operation coverage range of the static intervals
st = [float(MAC[f"clean{k}PICP"]) for k in ("SCP", "LACP", "CCCP", "CCLACP")]
mac("cleanStaticMin", f3(min(st)))
mac("cleanStaticMax", f3(max(st)))
say(f"  static normal-operation PICP {min(st):.3f}-{max(st):.3f}")

# ======================================================================
with open(os.path.join(PAPER, "numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% generated by analysis/paper_numbers.py -- do not edit\n")
    for k, v in MAC.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
say(f"\n{len(MAC)} macros written")
LOG.close()
