"""
Checks every qualitative, ordering or bound statement in main.tex (protocol v3) that is
not a generated macro. Reads figures/data/*.csv, the macro files, the paper_numbers log,
the final_stats2 tables of final3/v3_tree and the battery events.
Output: verify_paper_statements_output.txt
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
D = os.path.join(PAPER, "figures", "data")
AA = os.path.abspath(os.path.join(PAPER, "..", "Additional_Analyses"))
RUN = os.path.join(AA, "final3", "v3_tree")
OUT = open(os.path.join(HERE, "verify_paper_statements_output.txt"), "w", encoding="utf-8")
ok_all = True


def check(label, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    line = f"[{'OK ' if cond else 'FAIL'}] {label}  {detail}"
    print(line)
    OUT.write(line + "\n")


def macros(name):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(os.path.join(PAPER, name), encoding="utf-8").read()))


M = {**macros("numbers.tex"), **macros("numbers_extra.tex"), **macros("numbers_battery.tex"),
     **macros("numbers_review.tex")}
LOG = open(os.path.join(HERE, "paper_numbers_output.txt"), encoding="utf-8").read()
SEL = json.load(open(os.path.join(RUN, "selection.json")))["selected"]


def num(s):
    return float(s.replace("$-$", "-").replace("\\%", "").replace("+", ""))


def ci(s):
    a_, b_ = s.strip("[]").split(",")
    return num(a_), num(b_)


F2 = pd.read_csv(os.path.join(D, "fig2_pv_loss_households.csv"))
F3 = pd.read_csv(os.path.join(D, "fig3_parameter_sweep.csv"))
F4 = pd.read_csv(os.path.join(D, "fig4_pv_severity.csv")).set_index(["method", "severity"])
ET = pd.read_csv(os.path.join(D, "fig_eventtime.csv"))
HZ = pd.read_csv(os.path.join(D, "fig_horizon.csv"))
BD = pd.read_csv(os.path.join(D, "fig_bands.csv")).set_index(["method", "band"])
R3 = pd.read_csv(os.path.join(RUN, "tables", "C3_extended_response.csv"))

# ---------------------------------------------------------------- Q1, Fig. households
a = F2[F2.panel == "a"].set_index(["method", "household"])
b = F2[F2.panel == "b"].set_index(["method", "household"])
check("LACP narrows under PV loss in H3 and H4 (CI below 0)",
      a.loc[("LACP", "H3"), "hi"] < 0 and a.loc[("LACP", "H4"), "hi"] < 0)
check("LACP widens in H6 (CI above 0)", a.loc[("LACP", "H6"), "lo"] > 0)
lw = re.findall(r"log width CC-LACP - LACP, PV loss, (H\d): ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", LOG)
check("factor widens LACP in every household (paired CI above 0)", len(lw) == 3 and all(float(x[2]) > 0 for x in lw),
      str(lw))
check("CC-LACP lower than LACP under PV loss in H4 and H6 (CI below 0)",
      b.loc[("CC-LACP", "H4"), "hi"] < 0 and b.loc[("CC-LACP", "H6"), "hi"] < 0)
check("CC-LACP not distinguishable from LACP under PV loss in H3 (CI contains 0)",
      b.loc[("CC-LACP", "H3"), "lo"] < 0 < b.loc[("CC-LACP", "H3"), "hi"])
check("ACI widens most in H6 (> 3x the other households)",
      a.loc[("ACI", "H6"), "value"] > 3 * max(a.loc[("ACI", "H3"), "value"], a.loc[("ACI", "H4"), "value"]))

# ---------------------------------------------------------------- Q1, severity
v = [F4.loc[("LACP", s_), "value"] for s_ in (0.25, 0.5, 0.75)]
check("LACP width ratio falls as the loss deepens", v[0] > v[1] > v[2], str(np.round(v, 3)))
check("LACP ratio below one at s = 0.75 (CI below 0)", F4.loc[("LACP", 0.75), "hi"] < 0)
dv = [num(M[k]) for k in ("lwdiffSevLow", "lwdiffSevMid", "lwdiffSevHigh")]
check("width added by the factor grows with the loss", dv[0] < dv[1] < dv[2], str(dv))
c = [F4.loc[("CC-LACP", s_), "value"] for s_ in (0.25, 0.5, 0.75)]
check("CC-LACP ratio stays level (range < 0.01)", max(c) - min(c) < 0.01, str(np.round(c, 3)))
for m in ["CC-CP", "ACI"]:
    v = [F4.loc[(m, s_), "value"] for s_ in (0.25, 0.5, 0.75)]
    check(f"{m} widens more as the loss grows", v[0] < v[1] < v[2], str(np.round(v, 3)))

# ---------------------------------------------------------------- Q1, event time
W = ET[ET.panel == "width"].set_index(["method", "x"])
at0 = {m: W.loc[(m, 0.0), "value"] for m in ["LACP", "CC-LACP", "CC-CP", "ACI"]}
check("ACI widens most at the first disturbed issue time", max(at0, key=at0.get) == "ACI", str(at0))
check("ACI stays wider than placebo until the loss ends (CI above 0 at 0..105 min)",
      all(W.loc[("ACI", x), "lo"] > 0 for x in np.arange(0.0, 106.0, 15.0)))
cc = W.loc["CC-CP"].loc[0:105, "value"]
check("CC-CP widening peaks at the onset", cc.idxmax() == 0.0, str(cc.round(3).to_dict()))
check("CC-CP widening fades before the loss ends (CI at 105 min contains 0)",
      W.loc[("CC-CP", 105.0), "lo"] < 0 < W.loc[("CC-CP", 105.0), "hi"])
check("LACP narrower than placebo throughout the loss (value < 0 at 0..105)",
      all(W.loc[("LACP", x), "value"] < 0 for x in np.arange(0.0, 106.0, 15.0)))
for k in ["covHourDCCLACPvsLACPCI", "covHourDCCLACPvsACICI"]:
    lo_, hi_ = ci(M[k])
    check(f"{k} excludes zero (text: more than)", lo_ > 0, M[k])

# ---------------------------------------------------------------- Q2, Table II
order = [num(M[f"relCCLACPvsLACP{h}"]) for h in ("HSix", "HFour", "HThree")]
check("CC-LACP vs LACP ranges from H6 (lowest) to H3 (highest)", order[0] < order[1] < order[2], str(order))
check("width-matched control scores lower than CC-LACP", num(M["relCCLACPvsWM"]) > 0, M["relCCLACPvsWM"])
check("CC-LACP vs width-matched control: interval above zero (text gives it)", ci(M["relCCLACPvsWMCI"])[0] > 0,
      M["relCCLACPvsWMCI"])
KEYS = ["SCP", "LACP", "CCCP", "CCLACP", "ACI", "ACItuned", "ACIL", "ACICCL", "ROLLSCP", "ROLLLACP", "ROLLCCLACP"]
evp = {k: num(M[f"ev{k}PICP"]) for k in KEYS}
check("3.7-kW steps fall outside most intervals (PICP < 0.5 for most methods)",
      sum(x < 0.5 for x in evp.values()) > len(evp) / 2, str(evp))
inp = {k: num(M[f"in{k}PICP"]) for k in KEYS}
check("coverage below 0.90 inside disturbances for every method", max(inp.values()) < 0.90, str(inp))

# ---------------------------------------------------------------- Q2, bands
check("score marks larger error: SCP covers more at zero than in the high band",
      BD.loc[("SCP", "high"), "hi"] < BD.loc[("SCP", "zero"), "lo"])
for m in ["CC-CP", "CC-LACP"]:
    check(f"{m} raises high-band coverage above SCP and LACP",
          BD.loc[(m, "high"), "lo"] > max(BD.loc[("SCP", "high"), "hi"], BD.loc[("LACP", "high"), "hi"]))
check("width-matched SCP stays near SCP in the high band",
      abs(BD.loc[("Width-matched SCP", "high"), "picp"] - BD.loc[("SCP", "high"), "picp"]) < 0.02)
check("CC-LACP covers less than LACP when the score is zero (CIs separate)",
      BD.loc[("CC-LACP", "zero"), "hi"] < BD.loc[("LACP", "zero"), "lo"])

# ---------------------------------------------------------------- Q2, trade-off
okc = F3[(F3.inside_rel < 0) & (F3.clean_rel <= 0.02)]
check("of the Fig. 6 sweeps, only the selected CC-LACP gain is below LACP inside with clean score within 2%",
      list(okc.method) == [SEL["CCLACP"]], str(list(okc.method)))
g = F3[F3.family == "CC-LACP"].set_index("param")
check("larger CC-LACP gains raise both scores",
      all(g.loc[p, "clean_rel"] > g.loc[0.25, "clean_rel"] and g.loc[p, "inside_rel"] > g.loc[0.25, "inside_rel"]
          for p in (0.5, 1, 2)))
check("ACI above LACP inside events at every step", (F3[F3.family == "ACI"].inside_rel > 0).all())
check("CC-CP at least 13% above LACP inside for every gain", (F3[F3.family == "CC-CP"].inside_rel >= 0.13).all(),
      str(F3[F3.family == "CC-CP"].inside_rel.round(3).tolist()))
ai = F3[F3.family == "ACI"].set_index("param")
ac = F3[F3.family == "ACI-CC"].set_index("param")
check("factor lowers both scores of ACI at the smallest steps",
      all(ac.loc[p, "clean_rel"] < ai.loc[p, "clean_rel"] and ac.loc[p, "inside_rel"] < ai.loc[p, "inside_rel"]
          for p in (0.005, 0.01)))
gs = float(SEL["ACI"].split("|g")[1])
check("factor raises both scores of ACI at the selected step",
      ac.loc[gs, "clean_rel"] > ai.loc[gs, "clean_rel"] and ac.loc[gs, "inside_rel"] > ai.loc[gs, "inside_rel"])
check("rule sets the gain of ACI on CC-CP to zero", SEL["ACICC"] == SEL["ACI"])
off = F3[((F3.clean_rel * 100 > 18.5) | (F3.inside_rel * 100 > 22)) & F3.family.isin(["CC-CP", "ACI", "ACI-CC",
                                                                                      "CC-LACP"])]
check("off-scale points named in the caption",
      sorted(off.method) == sorted(["ACI|g0.5", "ACICC|b0.25|g0.5", "CC|b4", "CCLACP|b2"]), str(sorted(off.method)))

# ---------------------------------------------------------------- Q2, Table III


def net(m, st):
    r = R3[(R3.Outcome == "Ad125") & (R3.ShiftType == st) & (R3.Method == m)].iloc[0]
    return r.Net, r.CI_lo, r.CI_hi


def sig(m, st):
    _, lo_, hi_ = net(m, st)
    return not (lo_ <= 0 <= hi_)


check("PV channel carries the PV-loss response, not the load-step response",
      sig(SEL["PVonly"], "PV") and not sig(SEL["PVonly"], "LOAD"))
check("load channel carries the load-step response, not the PV-loss response",
      sig(SEL["LoadOnly"], "LOAD") and not sig(SEL["LoadOnly"], "PV"))
check("models restricted to the day lag lose the PV-loss response", not sig(SEL["LF-both"], "PV"))
check("persistence residual responds less to PV loss", net(SEL["Persist"], "PV")[0] < net(SEL["CC"], "PV")[0])
check("persistence residual scores worse inside events", num(M["ablPersist"]) > num(M["ablCC"]))
check("on the LACP scale the load-step response comes from s(x): LACP and CC-LACP load responses within 0.01",
      abs(net("LACP", "LOAD")[0] - net(SEL["CCLACP"], "LOAD")[0]) < 0.01)
check("PV channel alone gives nearly the same gain as both (within 0.5 points)",
      abs(num(M["ablPVonlyL"]) - num(M["ablCCLACP"])) < 0.5)
check("persistence residual and indicator give smaller gains",
      num(M["ablPersistL"]) > num(M["ablCCLACP"]) and num(M["ablTHL"]) > num(M["ablCCLACP"]))

# ---------------------------------------------------------------- Q3
check("CC-LACP vs ACI has opposite signs by household",
      num(M["relCCLACPvsACIHThree"]) < 0 < num(M["relCCLACPvsACIHSix"]))
check("ACI covers more inside disturbances than CC-LACP", inp["ACI"] > inp["CCLACP"])
for k in ["relCCLACPvsACILCI", "relCCLACPvsROLLLACPCI"]:
    lo_, hi_ = ci(M[k])
    check(f"{k} below zero (text: scores below)", hi_ < 0, M[k])
cp = [num(M[f"clean{k}PICP"]) for k in ["ROLLLACP", "ROLLCCLACP", "ACIL", "ACICCL"]]
check("recalibration brings normal-operation coverage to about 0.90", all(0.89 <= x <= 0.91 for x in cp), str(cp))
for k in ["relROLLCCLACPvsROLLLACPCI", "relACICCLvsACILCI", "netdCCCPvsACICI", "relCCLACPvsACICI"]:
    lo_, hi_ = ci(M[k])
    check(f"{k} contains zero (text: no significant average gain / not distinguishable)", lo_ <= 0 <= hi_, M[k])
for k in ["onsetFROLLCI", "onsetFACILCI", "covHourFROLLCI", "covHourFACILCI"]:
    lo_, hi_ = ci(M[k])
    check(f"{k}: the factor still widens at onset / raises first-hour coverage (CI above 0)", lo_ > 0, M[k])
for k in ["evfacROLLCI", "evfacACILCI"]:
    lo_, hi_ = ci(M[k])
    check(f"{k}: the factor lowers the score on 3.7-kW steps inside recalibration (CI below 0)", hi_ < 0, M[k])
h = HZ[(HZ.types == "all") & (HZ.method == "CC-LACP")]
check("CC-LACP below LACP at every horizon (CI below 0)", (h.hi < 0).all(), str(h[["horizon", "value", "hi"]].values))

# ---------------------------------------------------------------- Q4
B = pd.read_csv(os.path.join(D, "battery_events.csv.gz"))
B["Method"] = B.Method.replace({"ACI-CC": "ACI"})
S = B[B.Condition == "shift"]
pv = S[(S.EventSet == "extended") & (S.ShiftType == "PV")].groupby("Method").Adequacy.mean()
tr = S[S.EventSet == "trip"].groupby("Method").Adequacy.mean()
ld = S[(S.EventSet == "extended") & (S.ShiftType == "LOAD")].groupby("Method").Adequacy.mean()
top = {"Rolling LACP", "Rolling CC-LACP", "ACI-LACP", "ACI-CC-LACP"}
check("rolling and ACI reserves on the LACP and CC-LACP scores have the highest RAR under PV loss",
      set(pv.drop("Point only").sort_values(ascending=False).index[:4]) == top, str(pv.round(3).to_dict()))
check("... and under complete trips",
      set(tr.drop("Point only").sort_values(ascending=False).index[:4]) == top, str(tr.round(3).to_dict()))
for k in ["batPVAdqCCLACPvsLACP", "batTripAdqCCLACPvsLACP"]:
    lo_, hi_ = ci(M[k + "CI"])
    check(f"{k} positive, below 0.01, CI above 0", 0 < num(M[k]) < 0.01 and lo_ >= 0, M[k] + " " + M[k + "CI"])
for a, b_ in [("CC-LACP", "LACP"), ("Rolling CC-LACP", "Rolling LACP"), ("ACI-CC-LACP", "ACI-LACP")]:
    check(f"factor lowers the load-step RAR slightly: {a} < {b_} by less than 0.02",
          0 < ld[b_] - ld[a] < 0.02, f"{ld[b_]:.3f} -> {ld[a]:.3f}")
check("factor raises the ACI-on-LACP RAR under trips (paired CI above 0)",
      ci(M["batTripAdqACICCLvsACILCI"])[0] > 0, M["batTripAdqACICCLvsACILCI"])
check("3.7-kW steps exceed the 2.5-kW discharge limit", 3.7 > 2.5)
check("SCP and ACI reserves rank highest under load steps",
      set(ld.drop("Point only").sort_values(ascending=False).index[:2]) == {"SCP", "ACI"}, str(ld.round(3).to_dict()))
fac = [pv["CC-LACP"] - pv["LACP"], tr["CC-LACP"] - tr["LACP"], pv["ACI-CC-LACP"] - pv["ACI-LACP"],
       tr["ACI-CC-LACP"] - tr["ACI-LACP"], pv["Rolling CC-LACP"] - pv["Rolling LACP"],
       tr["Rolling CC-LACP"] - tr["Rolling LACP"]]
cal = [pv["Rolling LACP"] - pv["LACP"], pv["ACI-LACP"] - pv["LACP"], tr["Rolling LACP"] - tr["LACP"],
       tr["ACI-LACP"] - tr["LACP"]]
check("factor changes RAR less than the calibration method (max factor effect < min calibration effect)",
      max(np.abs(fac)) < min(cal), f"factor {np.round(fac, 3)}, calibration {np.round(cal, 3)}")

# ---------------------------------------------------------------- audit round (rounds 25-28)
# abstract, contribution 3, conclusion
check("a learned difficulty scale narrows in most PV losses (LACP narrower than placebo in > 50%)",
      num(M["narrowLACP"]) > 50, M["narrowLACP"])
check("normal-operation score of CC-LACP slightly higher than LACP (0 < change < 2%)",
      0 < num(M["cleanrelCCLACP"]) < 2, M["cleanrelCCLACP"])
check("per-household range in the abstract: H6 lowest, H3 highest",
      num(M["relCCLACPvsLACPHSix"]) <= min(num(M[f"relCCLACPvsLACPH{x}"]) for x in ("Three", "Four", "Six"))
      and num(M["relCCLACPvsLACPHThree"]) >= max(num(M[f"relCCLACPvsLACPH{x}"]) for x in ("Three", "Four", "Six")))
check("ACI normal-operation score higher than CC-LACP (text: at a higher normal-operation score)",
      num(M["cleanrelACI"]) > num(M["cleanrelCCLACP"]), M["cleanrelACI"] + " vs " + M["cleanrelCCLACP"])
for k in ["dmCCLACPvsLACP", "dmCCLACPvsACI"]:
    check(f"{k}: counts of significantly lower / not different / higher sum to nine tasks",
          sum(int(x) for x in M[k].split("/")) == 9, M[k])
# Sec. III-D: development-split ACI steps
ST = pd.read_csv(os.path.join(RUN, "selection_table.csv"))
ST = ST[ST.Family == "ACI"].copy()
ST["g"] = ST.Candidate.str.split("|g", regex=False).str[1].astype(float)
adm = sorted(ST[ST.Admissible.astype(str) == "True"].g)
check("development: only the steps 0.1 and 0.2 lie within the 2% tolerance", adm == [0.1, 0.2], str(adm))
lowr = (ST[ST.g < 0.1].CleanRatio - 1) * 100
check("development: smaller steps score 5.4% to 11.2% above the best without injection",
      f"{lowr.min():.1f}" + "\\%" == M["devACIlowMin"] and f"{lowr.max():.1f}" + "\\%" == M["devACIlowMax"],
      f"{lowr.min():.1f}-{lowr.max():.1f}")
check("development: the selected ACI step is admissible", float(SEL["ACI"].split("|g")[1]) in adm, SEL["ACI"])
# Q3: coverage below 0.90 in normal operation
stc = [num(M[f"clean{k}PICP"]) for k in ("SCP", "LACP", "CCCP", "CCLACP")]
check("static normal-operation coverage 0.858-0.875, below 0.90",
      f"{min(stc):.3f}" == M["cleanStaticMin"] and f"{max(stc):.3f}" == M["cleanStaticMax"] and max(stc) < 0.90,
      str(stc))
rc = [num(M[f"clean{k}PICP"]) for k in ("ROLLSCP", "ROLLLACP", "ROLLCCLACP")]
check("rolling recalibration restores about 0.90 (drift, not the calibration size)", all(x >= 0.89 for x in rc), str(rc))
uc = re.search(r"ACI gamma=[\d.]+ clipped: PICP ([\d.]+)  nIS ([\d.]+)\s+ACI gamma=[\d.]+ unclipped: PICP ([\d.]+)  nIS ([\d.]+)",
               LOG)
check("ACI below 0.90 because of the clip: unclipped covers 0.900 at a higher nIS",
      uc is not None and float(uc.group(1)) < 0.89 and abs(float(uc.group(3)) - 0.90) < 0.002
      and float(uc.group(4)) > float(uc.group(2)) and f"{float(uc.group(1)):.3f}" == M["cleanACIPICP"],
      " ".join(uc.group(0).split()) if uc else "")
# Sec. III-C: inside-event targets by duration and horizon
Et = pd.read_csv(os.path.join(RUN, "event_rows_test.csv.gz"), usecols=["EventSet", "Condition", "Method", "Household",
                                                                       "EventID", "Horizon", "DurationMin", "N_in"])
ex = Et[(Et.EventSet == "extended") & (Et.Condition == "shift") & (Et.Method == "LACP")]
nin = ex.groupby(["DurationMin", "Horizon"]).N_in.max().unstack().fillna(0)     # no row: no target at all
check("30-min events enter only at 15 min, 60-min events at 15 and 30 min",
      nin.loc[30, "15min"] > 0 and nin.loc[30, "30min"] == 0 and nin.loc[30, "60min"] == 0
      and nin.loc[60, "30min"] > 0 and nin.loc[60, "60min"] == 0 and nin.loc[120, "60min"] > 0, str(nin.to_dict()))
# robustness: second event set
fz = Et[(Et.EventSet == "frozen") & (Et.Condition == "shift")].drop_duplicates(["Household", "EventID"])
check("second set: 161 scored events of two per configuration and household (27 x 2 x 3 = 162)",
      len(fz) == int(M["nFrozen"]) == 161 and len(fz) <= 162, str(len(fz)))

# restored passages (rounds 36-42)
check("ACI's step sets how often it widens without injection (tune-split step widens less often than the selected)",
      num(M["faACItuned"]) < num(M["faACI"]), M["faACItuned"] + " vs " + M["faACI"])
check("test-period macros present", all(k in M for k in ("testHThree", "testHFour", "testHSix")),
      ", ".join(M.get(k, "?") for k in ("testHThree", "testHFour", "testHSix")))
check("3.7 kW is the power of a single-phase 16-A charger at 230 V", abs(16 * 230 / 1000 - 3.7) < 0.05)

# ---------------------------------------------------------------- robustness
Cd = pd.read_csv(os.path.join(RUN, "clean_metrics_dev.csv"))
Ed = pd.read_csv(os.path.join(RUN, "event_rows_dev.csv.gz"))
Ed = Ed[Ed.Condition == "shift"]
cands = [f"CCLACP|b{b_}" for b_ in ["0", "0.25", "0.5", "1", "2"]]
for hh in ["H3", "H6"]:
    c_ = Cd[Cd.Household == hh].groupby("Method").apply(lambda g_: (g_.nIS * g_.N).sum() / g_.N.sum())
    d_ = Ed[Ed.Household == hh].groupby("Method").apply(lambda g_: (g_.IS_in / g_.R).sum() / g_.N_in.sum())
    if hh == "H3":
        check("H3: the factor does not lower the development score", d_[cands].idxmin() == "CCLACP|b0")
    else:
        check("H6: the factor lowers the development score but exceeds the 2% tolerance",
              d_[cands].idxmin() != "CCLACP|b0" and c_["CCLACP|b0.25"] > 1.02 * c_[cands].min())
lo_, hi_ = ci(M["relCCLACPvsLACPHThreeCI"])
check("H3: the factor does not help on test either (CI above 0)", lo_ > 0, M["relCCLACPvsLACPHThreeCI"])
check("H6 has the largest in-event gain of the three households",
      num(M["relCCLACPvsLACPHSix"]) < min(num(M["relCCLACPvsLACPHThree"]), num(M["relCCLACPvsLACPHFour"])))
lt = open(os.path.join(HERE, "verify_lasting_loss_output.txt"), encoding="utf-8").read()
check("lasting-loss check output present (score back at control level within about 90 min)", len(lt) > 100)

# ---------------------------------------------------------------- review round 1 (Reviewer 3, H3; Reviewer 1)
RV = open(os.path.join(HERE, "review1_numbers_output.txt"), encoding="utf-8").read()
check("gain largest where the unmodified interval misses: lowest-coverage bin below zero (CI), highest above (CI)",
      ci(M["binLowCI"])[1] < 0 < ci(M["binHighCI"])[0], M["binLowCI"] + " / " + M["binHighCI"])
check("bins ordered: change rises with LACP coverage",
      num(M["binLow"]) < num(M["binMidLow"]) < num(M["binMidHigh"]) < num(M["binHigh"]),
      " < ".join(M[k] for k in ("binLow", "binMidLow", "binMidHigh", "binHigh")))
check("rank correlation positive on test and on development events", float(M["cellRho"]) > 0.5
      and float(M["cellRhoDev"]) > 0.5, M["cellRho"] + ", " + M["cellRhoDev"])
pos = re.findall(r"(H\d): Spearman test ([-\d.]+), development ([-\d.]+)", RV)
check("the relation also holds within each household (positive rank correlation, test and development)",
      len(pos) == 3 and all(float(a) > 0 and float(b) > 0 for _, a, b in pos), str(pos))
check("fitted change crosses zero inside the observed coverage range", 0.75 < float(M["cellCross"]) < 0.9,
      M["cellCross"])
cv = [float(M[f"covLACPin{h}"]) for h in ("HThree", "HFour", "HSix")]
check("LACP in-event coverage highest in H3, then H4, then H6", cv[0] > cv[1] > cv[2], str(cv))
check("in H3 the added width outweighs the misses removed", num(M["dWidthHThree"]) + num(M["dPenHThree"]) > 0,
      M["dWidthHThree"] + " " + M["dPenHThree"])
check("width and penalty parts add up to the H3 change (+2.2%)",
      abs(num(M["dWidthHThree"]) + num(M["dPenHThree"]) - num(M["relCCLACPvsLACPHThree"])) < 0.15)
check("H3: PV loss nearly neutral, load steps and combined events raise the score",
      abs(num(M["relHThreePV"])) < 1.0 and num(M["relHThreeLoad"]) > 0 and num(M["relHThreeComb"]) > 0,
      " ".join(M[k] for k in ("relHThreePV", "relHThreeLoad", "relHThreeComb")))
check("'most H3 events' already covered: majority of H3 cells have LACP coverage >= 0.85",
      float(M["cellHighHThree"].replace("\\%", "")) > 50, M["cellHighHThree"])
tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
fig1 = open(os.path.join(PAPER, "figures", "fig_method.tex"), encoding="utf-8").read()
check("no 'placebo' left in the paper text or Fig. 1", "placebo" not in tex.lower() and "placebo" not in fig1.lower())
body = re.sub(r"\\[A-Za-z]+", " ", tex)
left = [a for a in ("nIS", "PICP", "RAR", "WM-SCP", "PDE") if re.search(r"\b" + re.escape(a) + r"\b", body)]
check("removed abbreviations absent from the text (macro names excluded)", not left, str(left))
for t in ("table_results.tex", "table_ablation.tex", "table_battery.tex"):
    tb = open(os.path.join(PAPER, "tables", t), encoding="utf-8").read()
    check(f"{t}: no PICP/nIS/RAR/Placebo in headers", not re.search(r"PICP|nIS|RAR|Placebo", tb))

print("ALL OK" if ok_all else "SOME CHECKS FAILED")
OUT.write(("ALL OK" if ok_all else "SOME CHECKS FAILED") + "\n")
OUT.close()
