"""
Tables and tests for a corrected run (engine2):  python final_stats2.py v2_tree [selection_tag]

A  Clean test period: pooled and per-task PICP / PINAW / nIS; Diebold-Mariano
   tests (Newey-West, 24 lags, Holm over tasks); coverage by consistency band.
B  Incremental information of the consistency score in normal operation.
C  Injected events (per event set):
   C1 accuracy on inside-event targets (target before the event ends) and,
      separately, on the recovery step (first target after the event);
   C2 paired differences, cluster bootstrap over household-days;
   C3 net adaptation (shift - placebo), crossing only while the disturbance is
      present, thresholds 1.10 / 1.25 / 1.50 and within 15 min;
   C4 continuous response: mean log ratio of the mean width over issue times
      inside the event, shift / placebo; share of windows that narrow;
   C5 severity and duration; C6 per household; C7 trade-off over all gains and
      ACI steps.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("FINAL2_ROOT", os.path.join(HERE, "final2"))
TAG = sys.argv[1]
STAG = sys.argv[2] if len(sys.argv) > 2 else TAG
D = os.path.join(ROOT, TAG)
OUT = os.path.join(D, "tables")
os.makedirs(OUT, exist_ok=True)
ALPHA, HAC_LAGS, EWMA_HALFLIFE, TAU, NB = 0.10, 24, 4, 0.90, 4000
rng = np.random.default_rng(42)
SUMMARY = open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
SEL = json.load(open(os.path.join(ROOT, STAG, "selection.json")))["selected"]
SAME_ACICC = SEL["ACICC"] == SEL["ACI"]      # rule selected beta = 0: ACI-CC reduces to ACI
LABEL = {"SCP": "SCP", "LACP": "LACP", "ACI|tuned": "ACI (clean-tuned step)",
         SEL["ACICC"]: "ACI-CC", "CC|tuned": "CC-CP (clean-tuned gain)", SEL["CC"]: "CC-CP",
         SEL["CCLACP"]: "CC-LACP", SEL["Persist"]: "Persistence signal",
         "WM-CC": "Width-matched static (CC-CP)", "WM-CCLACP": "Width-matched LACP (CC-LACP)"}
LABEL[SEL["ACI"]] = "ACI (= ACI-CC, beta 0 selected)" if SAME_ACICC else "ACI"
MAIN = ["SCP", "LACP", "ACI|tuned", SEL["ACI"], "CC|tuned", SEL["CC"], SEL["CCLACP"],
        SEL["ACICC"], SEL["Persist"]]
MAIN = list(dict.fromkeys(MAIN))
CTRL = ["WM-CC", "WM-CCLACP"]


def lab(m):
    return LABEL.get(m, m)


def say(title, obj=None):
    txt = f"\n== {title} ==\n" + ("" if obj is None else
                                   (obj.to_string() if hasattr(obj, "to_string") else str(obj)))
    print(txt)
    SUMMARY.write(txt + "\n")


def iarr(y, lo, hi):
    w = hi - lo
    return w, (y >= lo) & (y <= hi), w + (2 / ALPHA) * (lo - y) * (y < lo) + (2 / ALPHA) * (y - hi) * (y > hi)


def dm_test(a, b, lags=HAC_LAGS):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    n = len(d)
    dc = d - d.mean()
    var = np.dot(dc, dc) / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * np.dot(dc[k:], dc[:-k]) / n
    se = np.sqrt(var / n)
    st = d.mean() / se if se > 0 else np.nan
    return d.mean(), st, 2 * (1 - stats.norm.cdf(abs(st)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m, run, adj = len(p), 0.0, np.empty(len(p))
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def boot(cl, num, den=None):
    """Cluster bootstrap of sum(num)/sum(den) (den = 1 per row if omitted)."""
    num = np.asarray(num, float)
    den = np.ones(len(num)) if den is None else np.asarray(den, float)
    codes, u = pd.factorize(pd.Series(np.asarray(cl)))
    sn = np.bincount(codes, weights=num, minlength=len(u))
    sd = np.bincount(codes, weights=den, minlength=len(u))
    ix = rng.integers(0, len(u), (NB, len(u)))
    b = sn[ix].sum(1) / sd[ix].sum(1)
    return sn.sum() / sd.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(p, lo, hi, signed=True):
    s = f"{p:+.3f}" if signed else f"{p:.3f}"
    return s + ("" if lo <= 0 <= hi else "*") + f" [{lo:+.3f},{hi:+.3f}]"


# ======================================================================
# A. clean test period
# ======================================================================
meta = pd.read_csv(os.path.join(D, "task_meta_test.csv"))
CM = pd.read_csv(os.path.join(D, "clean_metrics_test.csv")).merge(
    meta[["Household", "Horizon", "R"]], on=["Household", "Horizon"])
CM["PINAW"] = CM.MPIW / CM.R


def pool_clean(g):
    n = g.N.sum()
    return pd.Series(dict(PICP=(g.PICP * g.N).sum() / n, PINAW=(g.PINAW * g.N).sum() / n,
                          nIS=(g.nIS * g.N).sum() / n))


CP = CM.groupby("Method").apply(pool_clean)
A1 = CP.loc[[m for m in MAIN if m in CP.index]].rename(index=lab)
A1.to_csv(os.path.join(OUT, "A1_clean_pooled.csv"))
say("A1 clean test period, pooled", A1.round(4))
A2 = CM[CM.Method.isin(MAIN)].copy()
A2["Method"] = A2.Method.map(lab)
A2.to_csv(os.path.join(OUT, "A2_clean_per_task.csv"), index=False)
say("A2 clean coverage per task", A2.pivot_table(index=["Household", "Horizon"], columns="Method",
                                                  values="PICP").round(3))

PF = pd.read_csv(os.path.join(D, "clean_per_forecast.csv.gz"))
parts = []
for (h, hz), g in PF.groupby(["Household", "Horizon"], sort=False):
    g = g.sort_values("t").copy()
    k = np.mean(g[f"{SEL['CC']}_hi"] - g[f"{SEL['CC']}_lo"]) / np.mean(g.SCP_hi - g.SCP_lo)
    g["C'_lo"], g["C'_hi"] = g.pred - k * (g.SCP_hi - g.pred), g.pred + k * (g.SCP_hi - g.pred)
    parts.append(g)
PF = pd.concat(parts)
cols = [m for m in MAIN if f"{m}_lo" in PF.columns] + ["C'"]
for m in cols:
    w, c, s = iarr(PF.y.values, PF[f"{m}_lo"].values, PF[f"{m}_hi"].values)
    PF[f"{m}|cov"], PF[f"{m}|is"], PF[f"{m}|nw"] = c, s, w / PF.R.values
pairs = [(SEL["CC"], b) for b in ["SCP", "LACP", SEL["ACI"], "C'"]] + \
        [(SEL["CCLACP"], b) for b in ["LACP", SEL["ACI"]]] + \
        [(SEL["ACICC"], b) for b in [SEL["ACI"], "LACP", SEL["CCLACP"]]]
pairs = [p for p in pairs if p[0] != p[1]]
rows = []
for a, b in pairs:
    blk = []
    for (h, hz), g in PF.groupby(["Household", "Horizon"], sort=False):
        d, st, p = dm_test(g[f"{b}|is"].values, g[f"{a}|is"].values)
        blk.append(dict(Method=lab(a), Baseline=lab(b), Household=h, Horizon=hz, d=d, p=p))
    blk = pd.DataFrame(blk)
    blk["p_holm"] = holm(blk.p)
    rows.append(blk)
DM = pd.concat(rows)
DM.to_csv(os.path.join(OUT, "A3_clean_dm.csv"), index=False)
DM["s"] = np.where(DM.p_holm < 0.05, np.sign(DM.d), 0)
say("A3 clean DM, tasks where method is better / tie / worse (Holm 5%)",
    DM.groupby(["Method", "Baseline"])["s"].agg(lambda s: f"{(s > 0).sum()}/{(s == 0).sum()}/{(s < 0).sum()}"))
PF["band"] = np.where(PF.rho <= 0, "B0 zero", np.where(PF.rho <= PF.band_threshold, "B1 moderate", "B2 high"))
bands = pd.DataFrame([dict(Band=b, Share=len(g) / len(PF), Method=lab(m), PICP=g[f"{m}|cov"].mean(),
                           nIS=g[f"{m}|is"].mean())
                      for b, g in PF.groupby("band") for m in cols])
bands.to_csv(os.path.join(OUT, "A4_clean_bands.csv"), index=False)
say("A4 coverage by consistency band", bands.pivot_table(index="Band", columns="Method", values="PICP").round(3))

# ======================================================================
# B. incremental information
# ======================================================================
I = pd.read_csv(os.path.join(D, "info_inputs.csv.gz")).rename(columns={"timestamp_utc": "t", "index": "t"})
I["t"] = pd.to_datetime(I.t, utc=True)
irows = []
for (h, hz), g in I.groupby(["Household", "Horizon"], sort=False):
    st_ = int(g.steps.iloc[0])
    g = g.set_index("t").sort_index()
    g = g[~g.index.duplicated()]
    a = g.abs_err
    lag = pd.Timedelta(minutes=15 * st_)
    lag1 = a.reindex(a.index - lag)
    lag1.index = a.index
    ewm = pd.Series(a.values, index=a.index + lag).ewm(halflife=EWMA_HALFLIFE, adjust=False).mean() \
        .reindex(a.index, method="ffill")

    def build(split):
        s = g[g.split == split]
        X = pd.DataFrame(dict(lag_err=lag1.reindex(s.index).values, ewm_err=ewm.reindex(s.index).values,
                              hour_sin=s.hour_sin.values, hour_cos=s.hour_cos.values,
                              lacp_scale=s.lacp_scale.values, rho=s.rho.values), index=s.index)
        yv = s.abs_err.values
        ok = np.isfinite(X.values).all(1) & np.isfinite(yv)
        return X[ok], yv[ok]

    Xc, ac = build("cal")
    Xt, at = build("test")
    mu, sd = Xc.mean(), Xc.std().replace(0, 1)
    Xc, Xt = (Xc - mu) / sd, (Xt - mu) / sd
    base = ["lag_err", "ewm_err", "hour_sin", "hour_cos"]
    oos, coef = {}, {}
    for mn, cs in {"M0": base, "M1": base + ["rho"], "M2": base + ["lacp_scale"],
                   "M3": base + ["lacp_scale", "rho"]}.items():
        Xcm, Xtm = sm.add_constant(Xc[cs], has_constant="add"), sm.add_constant(Xt[cs], has_constant="add")
        if "rho" in cs:
            o = sm.OLS(ac, Xcm).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            coef[mn] = (o.params["rho"], o.pvalues["rho"])
        qr = sm.QuantReg(ac, Xcm).fit(q=TAU, max_iter=5000)
        u = at - qr.predict(Xtm)
        oos[mn] = np.maximum(TAU * u, (TAU - 1) * u)
    irows.append(dict(Household=h, Horizon=hz, rho_coef=coef["M1"][0], rho_p=coef["M1"][1],
                      gain_M1_pct=100 * (oos["M0"].mean() - oos["M1"].mean()) / oos["M0"].mean(),
                      p_M1=dm_test(oos["M0"], oos["M1"])[2],
                      gain_M3_pct=100 * (oos["M2"].mean() - oos["M3"].mean()) / oos["M2"].mean(),
                      p_M3=dm_test(oos["M2"], oos["M3"])[2]))
INFO = pd.DataFrame(irows)
for c in ["rho_p", "p_M1", "p_M3"]:
    INFO[c + "_holm"] = holm(INFO[c])
INFO.to_csv(os.path.join(OUT, "B1_incremental_information.csv"), index=False)
say("B1 incremental information", INFO.round(4))

# ======================================================================
# C. events
# ======================================================================
E = pd.read_csv(os.path.join(D, "event_rows_test.csv.gz"))
E["cl"] = E.Household + "_" + E.Day.astype(str)
EKEY = ["Household", "Horizon", "EventID", "ShiftType", "Severity", "DurationMin", "cl"]
SHOWM = [m for m in MAIN + CTRL if m in set(E.Method)]
KEYPAIRS = [(SEL["CC"], "LACP"), (SEL["CC"], SEL["ACI"]), (SEL["CC"], "ACI|tuned"),
            (SEL["CC"], SEL["Persist"]), (SEL["CC"], "WM-CC"), (SEL["CCLACP"], "LACP"),
            (SEL["CCLACP"], SEL["ACI"]), (SEL["CCLACP"], "WM-CCLACP"), (SEL["ACICC"], SEL["ACI"]),
            (SEL["ACICC"], "LACP"), (SEL["ACICC"], SEL["CCLACP"])]
KEYPAIRS = [p for p in dict.fromkeys(KEYPAIRS) if p[0] != p[1]]
TYPES = ["ALL", "PV", "COMBINED", "LOAD"]

for es, Es in E.groupby("EventSet"):
    t = f"C[{es}]"
    S = Es[Es.Condition == "shift"]
    nev = S.drop_duplicates(["Household", "EventID"]).shape[0]
    P = S[S.Method == SEL["CC"]]
    say(f"{t} events with at least one eligible forecast: {nev}; event-horizon pairs: {len(P)}; "
        f"inside-event targets: {int(P.N_in.sum())}; recovery targets: {int(P.N_rec.sum())}; "
        f"pairs with no inside-event target: {int((P.N_in == 0).sum())}; clusters: {S.cl.nunique()}")

    def pool(g, part):
        n = g[f"N_{part}"].sum()
        return pd.Series(dict(N=n, PICP=g[f"Cov_{part}"].sum() / n, PINAW=(g[f"W_{part}"] / g.R).sum() / n,
                              nIS=(g[f"IS_{part}"] / g.R).sum() / n))

    for part in ["in", "rec", "leg"]:
        X = S[S.Method.isin(SHOWM)].groupby("Method").apply(lambda g: pool(g, part)).loc[
            [m for m in SHOWM if m in set(S.Method)]]
        X.index = [lab(m) for m in X.index]
        X.to_csv(os.path.join(OUT, f"C1_{es}_{part}_pooled.csv"))
        name = {"in": "inside-event targets", "rec": "recovery step", "leg": "original rule (target <= end)"}[part]
        say(f"{t} C1 accuracy under injection, {name}", X.round(4))
    XT = S[S.Method.isin(SHOWM)].groupby(["ShiftType", "Method"]).apply(lambda g: pool(g, "in")).reset_index()
    XT["Method"] = XT.Method.map(lab)
    XT.to_csv(os.path.join(OUT, f"C1_{es}_in_by_type.csv"), index=False)
    say(f"{t} C1 inside-event nIS by type", XT.pivot_table(index="Method", columns="ShiftType", values="nIS").round(4))
    say(f"{t} C1 inside-event PICP by type", XT.pivot_table(index="Method", columns="ShiftType", values="PICP").round(3))

    # C2 paired differences on inside-event targets
    W = S.pivot_table(index=EKEY, columns="Method", values=["N_in", "Cov_in", "W_in", "IS_in", "R"],
                      aggfunc="first")
    prow = []
    for st in TYPES:
        sub = W if st == "ALL" else W[W.index.get_level_values("ShiftType") == st]
        sub = sub[sub[("N_in", SEL["CC"])] > 0]
        cl = sub.index.get_level_values("cl")
        R_ = sub[("R", SEL["CC"])].values
        n = sub[("N_in", SEL["CC"])].values
        for a, b in KEYPAIRS:
            for stat, col, norm in [("nIS", "IS_in", True), ("PICP", "Cov_in", False), ("PINAW", "W_in", True)]:
                num = sub[(col, a)].values - sub[(col, b)].values
                num = num / R_ if norm else num
                p, lo, hi = boot(cl, num, n)
                prow.append(dict(ShiftType=st, Method=lab(a), Baseline=lab(b), Stat=stat, Diff=p,
                                 CI_lo=lo, CI_hi=hi))
    PD = pd.DataFrame(prow)
    PD.to_csv(os.path.join(OUT, f"C2_{es}_paired_inside.csv"), index=False)
    say(f"{t} C2 paired differences, inside-event targets (method minus baseline)",
        PD[PD.ShiftType == "ALL"].assign(v=[fmt(a, b, c) for a, b, c in zip(PD[PD.ShiftType == "ALL"].Diff,
                                                                             PD[PD.ShiftType == "ALL"].CI_lo,
                                                                             PD[PD.ShiftType == "ALL"].CI_hi)])
        .pivot_table(index=["Method", "Baseline"], columns="Stat", values="v", aggfunc="first"))

    # C3 net adaptation and C4 continuous response
    arows, drows = [], []
    AD = Es.pivot_table(index=EKEY + ["Method"], columns="Condition",
                        values=["Ad110", "Ad125", "Ad150", "Onset15", "Worig"], aggfunc="first")
    for out in ["Ad125", "Onset15", "Ad110", "Ad150", "LogW"]:
        if out == "LogW":
            v = np.log(AD[("Worig", "shift")] / AD[("Worig", "placebo")])
            pl = pd.Series(np.nan, index=v.index)
        else:
            v = AD[(out, "shift")] - AD[(out, "placebo")]
            pl = AD[(out, "placebo")]
        v = v.dropna()
        for m in set(v.index.get_level_values("Method")):
            vm = v.xs(m, level="Method")
            for st in TYPES:
                g = vm if st == "ALL" else vm[vm.index.get_level_values("ShiftType") == st]
                if len(g) == 0:
                    continue
                p, lo, hi = boot(g.index.get_level_values("cl"), g.values)
                arows.append(dict(Outcome=out, Method=m, ShiftType=st, N=len(g), Net=p, CI_lo=lo, CI_hi=hi,
                                  Placebo=float(pl.xs(m, level="Method").mean()) if out != "LogW" else np.nan,
                                  Narrower=float((g < 0).mean()) if out == "LogW" else np.nan))
        vv = v.unstack("Method")
        for a, b in KEYPAIRS:
            if a not in vv or b not in vv:
                continue
            for st in TYPES:
                g = vv[[a, b]].dropna()
                g = g if st == "ALL" else g[g.index.get_level_values("ShiftType") == st]
                p, lo, hi = boot(g.index.get_level_values("cl"), (g[a] - g[b]).values)
                drows.append(dict(Outcome=out, Method=lab(a), Baseline=lab(b), ShiftType=st, Diff=p,
                                  CI_lo=lo, CI_hi=hi))
    AR = pd.DataFrame(arows)
    AR.to_csv(os.path.join(OUT, f"C3_{es}_response.csv"), index=False)
    DR = pd.DataFrame(drows)
    DR.to_csv(os.path.join(OUT, f"C3_{es}_response_differences.csv"), index=False)
    for out in ["Ad125", "Onset15", "LogW", "Ad110", "Ad150"]:
        g = AR[(AR.Outcome == out) & AR.Method.isin(SHOWM)].copy()
        g["v"] = [fmt(a, b, c) for a, b, c in zip(g.Net, g.CI_lo, g.CI_hi)]
        g["Method"] = g.Method.map(lab)
        tab = g.pivot_table(index="Method", columns="ShiftType", values="v", aggfunc="first")
        if out != "LogW":
            tab["placebo rate"] = g.groupby("Method").Placebo.first().round(3)
        else:
            tab["share narrower (PV)"] = g[g.ShiftType == "PV"].set_index("Method").Narrower.round(3)
        say(f"{t} C3 response: {out} (net = shift - placebo; LogW = mean log width ratio)", tab)
    for out in ["Ad125", "Onset15", "LogW"]:
        g = DR[DR.Outcome == out].copy()
        g["v"] = [fmt(a, b, c) for a, b, c in zip(g.Diff, g.CI_lo, g.CI_hi)]
        say(f"{t} C3 differences in {out}",
            g.pivot_table(index=["Method", "Baseline"], columns="ShiftType", values="v", aggfunc="first"))

    # C5 severity / duration (PV), C6 per household
    v = (AD[("Ad125", "shift")] - AD[("Ad125", "placebo")]).rename("net").to_frame()
    v["lr"] = np.log(AD[("Worig", "shift")] / AD[("Worig", "placebo")])
    v = v.reset_index()
    v = v[v.Method.isin(SHOWM)]
    v["Method"] = v.Method.map(lab)
    sv = pd.concat([v[v.ShiftType == "PV"].groupby(["Method", f])[["net", "lr"]].mean().reset_index()
                    .rename(columns={f: "Level"}).assign(Factor=f) for f in ["Severity", "DurationMin"]])
    sv.to_csv(os.path.join(OUT, f"C5_{es}_pv_severity_duration.csv"), index=False)
    say(f"{t} C5 PV loss: net adaptation (1.25) by severity / duration",
        sv.pivot_table(index=["Factor", "Level"], columns="Method", values="net").round(3))
    say(f"{t} C5 PV loss: mean log width ratio by severity / duration",
        sv.pivot_table(index=["Factor", "Level"], columns="Method", values="lr").round(3))
    hh = v.groupby(["Household", "ShiftType", "Method"])[["net", "lr"]].mean().reset_index()
    acc = S[S.Method.isin(SHOWM)].groupby(["Household", "Method"]).apply(lambda g: pool(g, "in")).reset_index()
    acc["Method"] = acc.Method.map(lab)
    hh.to_csv(os.path.join(OUT, f"C6_{es}_per_household_response.csv"), index=False)
    acc.to_csv(os.path.join(OUT, f"C6_{es}_per_household_accuracy.csv"), index=False)
    say(f"{t} C6 per household: PV net adaptation (1.25)",
        hh[hh.ShiftType == "PV"].pivot_table(index="Household", columns="Method", values="net").round(3))
    say(f"{t} C6 per household: inside-event nIS",
        acc.pivot_table(index="Household", columns="Method", values="nIS").round(3))

    # C7 trade-off over every gain and step
    fam = [m for m in set(Es.Method) if m.startswith(("ACI|g", "CC|b", "CCLACP|b", "ACICC|b", "Persist|b"))]
    fam += ["LACP", "SCP", "ACI|tuned", "CC|tuned"]
    tr = []
    ADm = AD.reset_index()
    for m in fam:
        acc_m = pool(S[S.Method == m], "in")
        rec_m = pool(S[S.Method == m], "rec")
        am = ADm[(ADm.Method == m) & (ADm.ShiftType == "PV")]
        tr.append(dict(Method=m, CleanPICP=CP.loc[m, "PICP"] if m in CP.index else np.nan,
                       CleanNIS=CP.loc[m, "nIS"] if m in CP.index else np.nan,
                       InsideNIS=acc_m.nIS, InsidePICP=acc_m.PICP, RecoveryNIS=rec_m.nIS,
                       NetAd125_PV=float((am[("Ad125", "shift")] - am[("Ad125", "placebo")]).mean()),
                       Onset15_PV=float((am[("Onset15", "shift")] - am[("Onset15", "placebo")]).mean()),
                       LogW_PV=float(np.log(am[("Worig", "shift")] / am[("Worig", "placebo")]).mean())))
    TR = pd.DataFrame(tr).sort_values("Method")
    TR["Selected"] = TR.Method.isin(SEL.values())
    TR.to_csv(os.path.join(OUT, f"C7_{es}_tradeoff.csv"), index=False)
    say(f"{t} C7 trade-off over gains and steps", TR.round(4).to_string(index=False))

say("selection used", SEL)
SUMMARY.close()
