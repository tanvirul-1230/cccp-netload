"""
Data and macros for the added figures (tree/linear backbone, protocol v3, extended events):
 1  horizon: household-mean relative change of inside-event nIS against LACP, by
    forecast horizon, with stratified cluster-bootstrap intervals, on a common
    event basis: 120-min events, issue times 0-45 min after the onset (every
    target of every horizon then lies inside the disturbance)
 2  normal operation: coverage and interval score by consistency band
 3  event time (needs figures/data/event_trace.csv.gz from event_trace.py):
    log width ratio injected/placebo by issue time and coverage by target time,
    120-min PV losses
Writes ../figures/data/fig_*.csv, ../numbers_extra.tex and
../verification_code/paper_extra_output.txt
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
AA = os.path.abspath(os.path.join(PAPER, "..", "Additional_Analyses"))
D = os.path.join(AA, "final3", "v3_tree")
DATA = os.path.join(PAPER, "figures", "data")
SEL = json.load(open(os.path.join(D, "selection.json")))["selected"]
M = {"LACP": "LACP", "CC-LACP": SEL["CCLACP"], "CC-CP": SEL["CC"], "ACI": SEL["ACI"], "ACI-CC": SEL["ACICC"],
     "ACI-tuned": "ACI|tuned", "ACI-LACP": SEL["ACIL"], "ACI-CC-LACP": SEL["ACICCL"]}
HOUSES = ["H3", "H4", "H6"]
NB = 4000
rng = np.random.default_rng(29)
LOG = open(os.path.join(PAPER, "verification_code", "paper_extra_output.txt"), "w", encoding="utf-8")
MAC = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def mac(k, v):
    assert k.isalpha() and k not in MAC, k
    MAC[k] = v


def pct(x):
    return f"{100 * x:+.1f}".replace("-", "$-$") + "\\%"


E = pd.read_csv(os.path.join(D, "event_rows_test.csv.gz"))
E = E[E.EventSet == "extended"]
E["cl"] = E.Household + "|" + E.Day.astype(str)
E["isr"] = E.IS_in / E.R
S = E[E.Condition == "shift"]


def rel(X, a, b):
    per, bs = {}, np.zeros(NB)
    for h in HOUSES:
        Xh = X[X.Household == h]
        A = Xh[Xh.Method == a].groupby("cl").isr.sum()
        B = Xh[Xh.Method == b].groupby("cl").isr.sum()
        idx = A.index.intersection(B.index)
        na, nb = A.loc[idx].values, B.loc[idx].values
        per[h] = na.sum() / nb.sum() - 1
        ix = rng.integers(0, len(idx), (NB, len(idx)))
        bs += na[ix].sum(1) / nb[ix].sum(1) - 1
    bs /= len(HOUSES)
    return float(np.mean(list(per.values()))), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ---------------------------------------------------------------- 1 horizon (common event basis)
TP = os.path.join(DATA, "event_trace.csv.gz")
TR = pd.read_csv(TP)
TR["cl"] = TR.Household + "|" + TR.Day.astype(str)
inv_trace = {"LACP": "LACP", "CC-LACP": "CC-LACP", "CC-CP": "CC-CP", "ACI": "ACI", "ACI-CC": "ACI-CC",
             "ACI-tuned": "ACI-tuned", "ACI-LACP": "ACI-LACP", "ACI-CC-LACP": "ACI-CC-LACP"}
CB = TR[(TR.Condition == "shift") & (TR.dur == 120) & (TR.off >= 0) & (TR.off <= 45) & (TR.toff < TR.dur)].copy()
CB["isr"] = CB.isr
CB = CB.groupby(["Household", "Horizon", "EventID", "ShiftType", "cl", "Method"], as_index=False).isr.sum()
say("common basis: 120-min events, issue offsets 0-45 min; target counts per event and horizon:",
    TR[(TR.Condition == "shift") & (TR.dur == 120) & (TR.off >= 0) & (TR.off <= 45) & (TR.toff < TR.dur)
       & (TR.Method == "LACP")].groupby(["Horizon", "EventID", "Household"]).size().groupby("Horizon").describe()
    .to_string())
rows = []
for types, tag in [(None, "all"), (["PV"], "pv")]:
    X = CB if types is None else CB[CB.ShiftType.isin(types)]
    for hz in ["15min", "30min", "60min"]:
        Xh = X[X.Horizon == hz]
        for lab in ["CC-LACP", "CC-CP", "ACI", "ACI-CC", "ACI-tuned", "ACI-LACP", "ACI-CC-LACP"]:
            m, lo, hi = rel(Xh, inv_trace[lab], "LACP")
            rows.append(dict(types=tag, horizon=int(hz[:-3]), method=lab, value=m, lo=lo, hi=hi))
H = pd.DataFrame(rows)
H.to_csv(os.path.join(DATA, "fig_horizon.csv"), index=False)
say("Inside-event nIS vs LACP by horizon (household mean)\n",
    H.assign(v=[f"{100 * v:+.1f} [{100 * a:+.1f},{100 * b:+.1f}]" for v, a, b in zip(H.value, H.lo, H.hi)])
    .pivot_table(index=["types", "horizon"], columns="method", values="v", aggfunc="first").to_string())
g = H[H.types == "all"].set_index(["method", "horizon"])
for lab, key in [("ACI", "ACI"), ("CC-LACP", "CCLACP"), ("ACI-CC", "ACICC"), ("CC-CP", "CCCP"),
                 ("ACI-LACP", "ACIL"), ("ACI-CC-LACP", "ACICCL")]:
    for hz, w in [(15, "Short"), (30, "Mid"), (60, "Long")]:
        mac(f"hz{key}{w}", pct(g.loc[(lab, hz), "value"]))

# ---------------------------------------------------------------- 2 bands (normal operation)
PF = pd.read_csv(os.path.join(D, "clean_per_forecast.csv.gz"))
PF["day"] = pd.to_datetime(PF.t, utc=True).dt.tz_convert("Europe/Berlin").dt.date.astype(str)
PF["cl"] = PF.Household + "|" + PF.day
parts = []
for (h, hz), gg in PF.groupby(["Household", "Horizon"], sort=False):
    gg = gg.sort_values("t").copy()
    k = np.mean(gg[f"{SEL['CC']}_hi"] - gg[f"{SEL['CC']}_lo"]) / np.mean(gg.SCP_hi - gg.SCP_lo)
    gg["WMS_lo"], gg["WMS_hi"] = gg.pred - k * (gg.SCP_hi - gg.pred), gg.pred + k * (gg.SCP_hi - gg.pred)
    parts.append(gg)
PF = pd.concat(parts)
PF["band"] = np.where(PF.rho <= 0, "zero", np.where(PF.rho <= PF.band_threshold, "moderate", "high"))
BM = {"SCP": "SCP", "LACP": "LACP", "ACI": SEL["ACI"], "CC-CP": SEL["CC"], "CC-LACP": SEL["CCLACP"],
      "Width-matched SCP": "WMS"}
brow = []
for lab, m in BM.items():
    cov = ((PF.y >= PF[f"{m}_lo"]) & (PF.y <= PF[f"{m}_hi"])).astype(float)
    for b in ["zero", "moderate", "high"]:
        sel = (PF.band == b).values
        c = cov[sel].values
        codes, u = pd.factorize(PF.cl[sel])
        s_ = np.bincount(codes, weights=c, minlength=len(u))
        n_ = np.bincount(codes, minlength=len(u)).astype(float)
        ix = rng.integers(0, len(u), (NB, len(u)))
        bb = s_[ix].sum(1) / n_[ix].sum(1)
        brow.append(dict(method=lab, band=b, share=float(sel.mean()), picp=float(c.mean()),
                         lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5)))
BD = pd.DataFrame(brow)
BD.to_csv(os.path.join(DATA, "fig_bands.csv"), index=False)
say("\nNormal operation: coverage by consistency band\n",
    BD.assign(v=[f"{p:.3f} [{a:.3f},{b:.3f}]" for p, a, b in zip(BD.picp, BD.lo, BD.hi)])
    .pivot_table(index="method", columns="band", values="v", aggfunc="first").to_string())
bd = BD.set_index(["method", "band"])
mac("bandHighShare", f"{100 * bd.loc[('SCP', 'high'), 'share']:.0f}\\%")
for lab, key in [("SCP", "SCP"), ("LACP", "LACP"), ("CC-CP", "CCCP"), ("CC-LACP", "CCLACP"), ("ACI", "ACI"),
                 ("Width-matched SCP", "WMS")]:
    mac(f"bandHigh{key}", f"{bd.loc[(lab, 'high'), 'picp']:.3f}")
    mac(f"bandZero{key}", f"{bd.loc[(lab, 'zero'), 'picp']:.3f}")

# ---------------------------------------------------------------- 3 event time
TP = os.path.join(DATA, "event_trace.csv.gz")
if os.path.exists(TP):
    T = pd.read_csv(TP)
    T = T[(T.ShiftType == "PV") & (T.dur == 120)]
    T["cl"] = T.Household + "|" + T.Day.astype(str)
    key = ["Household", "Horizon", "EventID", "Method", "off"]
    W = T.pivot_table(index=key + ["cl"], columns="Condition", values="w", aggfunc="first").dropna().reset_index()
    W["lr"] = np.log(W["shift"] / W["placebo"])
    erow = []
    for lab in ["LACP", "CC-LACP", "CC-CP", "ACI"]:
        for off, gg in W[W.Method == lab].groupby("off"):
            if off < -60 or off > 180:
                continue
            codes, u = pd.factorize(gg.cl)
            s_ = np.bincount(codes, weights=gg.lr.values, minlength=len(u))
            n_ = np.bincount(codes, minlength=len(u)).astype(float)
            ix = rng.integers(0, len(u), (NB, len(u)))
            bb = s_[ix].sum(1) / n_[ix].sum(1)
            erow.append(dict(panel="width", method=lab, x=off, value=float(gg.lr.mean()),
                             lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5)))
    C = T.copy()
    C["tbin"] = C.toff
    for lab in ["LACP", "CC-LACP", "CC-CP", "ACI"]:
        for cond in ["shift", "placebo"]:
            gsub = C[(C.Method == lab) & (C.Condition == cond)]
            for tb, gg in gsub.groupby("tbin"):
                if tb < -60 or tb > 180:
                    continue
                codes, u = pd.factorize(gg.cl)
                s_ = np.bincount(codes, weights=gg["cov"].values, minlength=len(u))
                n_ = np.bincount(codes, minlength=len(u)).astype(float)
                ix = rng.integers(0, len(u), (NB, len(u)))
                bb = s_[ix].sum(1) / n_[ix].sum(1)
                erow.append(dict(panel="coverage" if cond == "shift" else "coverage_placebo", method=lab, x=tb,
                                 value=float(gg["cov"].mean()), lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5)))
    ED = pd.DataFrame(erow)
    ED.to_csv(os.path.join(DATA, "fig_eventtime.csv"), index=False)
    say("\nEvent time, 120-min PV losses: mean log width ratio by issue time\n",
        ED[ED.panel == "width"].pivot_table(index="x", columns="method", values="value").round(3).to_string())
    say("\ncoverage by target time (injected)\n",
        ED[ED.panel == "coverage"].pivot_table(index="x", columns="method", values="value").round(3).to_string())
    ew = ED[ED.panel == "width"].set_index(["method", "x"])
    ec = ED[ED.panel == "coverage"].set_index(["method", "x"])
    for lab, key in [("LACP", "LACP"), ("CC-LACP", "CCLACP"), ("CC-CP", "CCCP"), ("ACI", "ACI")]:
        mac(f"etOnset{key}", f"{ew.loc[(lab, 0), 'value']:+.3f}".replace("-", "$-$"))
        mac(f"etEnd{key}", f"{ew.loc[(lab, 105), 'value']:+.3f}".replace("-", "$-$"))
        mac(f"etRestore{key}", f"{ew.loc[(lab, 120), 'value']:+.3f}".replace("-", "$-$"))
        fh = C[(C.Method == lab) & (C.Condition == "shift") & (C.tbin >= 0) & (C.tbin < 60)]
        first = fh["cov"].mean()
        codes, u = pd.factorize(fh.cl)
        s_ = np.bincount(codes, weights=fh["cov"].values, minlength=len(u))
        n_ = np.bincount(codes, minlength=len(u)).astype(float)
        ix = rng.integers(0, len(u), (NB, len(u)))
        bb = s_[ix].sum(1) / n_[ix].sum(1)
        mac(f"covHour{key}", f"{first:.3f}")
        mac(f"covHour{key}CI", f"[{np.percentile(bb, 2.5):.3f}, {np.percentile(bb, 97.5):.3f}]")
        say(f"  {lab}: coverage of targets in [0,60) min {first:.3f}; width at onset {ew.loc[(lab, 0), 'value']:+.3f},"
            f" at 105 min {ew.loc[(lab, 105), 'value']:+.3f}, at restoration {ew.loc[(lab, 120), 'value']:+.3f}")
    # paired first-hour coverage differences (same targets), cluster bootstrap over household-days
    FH = C[(C.Condition == "shift") & (C.tbin >= 0) & (C.tbin < 60)]
    PFH = FH.pivot_table(index=["Household", "Horizon", "EventID", "off", "cl"], columns="Method", values="cov",
                         aggfunc="first").dropna().reset_index()
    for a, b, key in [("CC-LACP", "LACP", "CCLACPvsLACP"), ("CC-LACP", "ACI", "CCLACPvsACI"),
                      ("CC-CP", "ACI", "CCCPvsACI")]:
        d = (PFH[a] - PFH[b]).values
        codes, u = pd.factorize(PFH.cl)
        s_ = np.bincount(codes, weights=d, minlength=len(u))
        n_ = np.bincount(codes, minlength=len(u)).astype(float)
        ix = rng.integers(0, len(u), (NB, len(u)))
        bb = s_[ix].sum(1) / n_[ix].sum(1)
        mac(f"covHourD{key}", f"{d.mean():+.3f}".replace("-", "$-$"))
        mac(f"covHourD{key}CI", "[{}, {}]".format(*[f"{v:+.3f}".replace("-", "$-$")
                                                     for v in np.percentile(bb, [2.5, 97.5])]))
        say(f"  first-hour coverage {a} - {b}: {d.mean():+.3f} [{np.percentile(bb, 2.5):+.3f}, "
            f"{np.percentile(bb, 97.5):+.3f}]")
    # the factor inside the recalibrated variants: onset width and first-hour coverage (paired)
    WON = T[T.off == 0].pivot_table(index=["Household", "Horizon", "EventID", "Method", "cl"], columns="Condition",
                                    values="w", aggfunc="first").dropna().reset_index()
    WON["lr"] = np.log(WON["shift"] / WON["placebo"])
    WON = WON.pivot_table(index=["Household", "Horizon", "EventID", "cl"], columns="Method", values="lr").dropna()
    WON = WON.reset_index()
    for a, b, key in [("Rolling CC-LACP", "Rolling LACP", "ROLL"), ("ACI-CC-LACP", "ACI-LACP", "ACIL"),
                      ("CC-LACP", "LACP", "LACP")]:
        for frame, name in [(PFH, "covHourF"), (WON, "onsetF")]:
            d = (frame[a] - frame[b]).values
            codes, u = pd.factorize(frame.cl)
            s_ = np.bincount(codes, weights=d, minlength=len(u))
            n_ = np.bincount(codes, minlength=len(u)).astype(float)
            ix = rng.integers(0, len(u), (NB, len(u)))
            bb = s_[ix].sum(1) / n_[ix].sum(1)
            mac(f"{name}{key}", f"{d.mean():+.3f}".replace("-", "$-$"))
            mac(f"{name}{key}CI", "[{}, {}]".format(*[f"{v:+.3f}".replace("-", "$-$")
                                                      for v in np.percentile(bb, [2.5, 97.5])]))
            say(f"  {name} {a} - {b}: {d.mean():+.3f} [{np.percentile(bb, 2.5):+.3f}, {np.percentile(bb, 97.5):+.3f}]")
    on = [ec.loc[(lab, 0), "value"] for lab in ["LACP", "CC-LACP", "CC-CP", "ACI"]]
    mac("covOnsetMin", f"{min(on):.2f}")
    mac("covOnsetMax", f"{max(on):.2f}")
    mac("etPeakACI", f"{ew.loc['ACI'].loc[0:105, 'value'].max():+.3f}")
    say("event rows:", len(T))
# ---------------------------------------------------------------- 4 off-scale points of the trade-off figure
SW = pd.read_csv(os.path.join(DATA, "fig3_parameter_sweep.csv"))
for meth, key in [("CC|b4", "offCCCP"), ("CCLACP|b2", "offCCLACP")]:
    r = SW[SW.method == meth].iloc[0]
    mac(key, f"({100 * r.clean_rel:.0f}, {100 * r.inside_rel:.0f})")
    say(f"off-scale {meth}: ({100 * r.clean_rel:.1f}, {100 * r.inside_rel:.1f})")

with open(os.path.join(PAPER, "numbers_extra.tex"), "w", encoding="utf-8") as f:
    f.write("% generated by analysis/paper_extra.py -- do not edit\n")
    for k, v in MAC.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
say(f"{len(MAC)} macros")
LOG.close()
