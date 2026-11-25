"""
Battery reserve results: event means per method, paired differences with a
cluster bootstrap over household-days, Table IV body and macros.
  python battery_stats.py -> ../tables/table_battery.tex, ../numbers_battery.tex,
                             ../verification_code/battery_stats_output.txt
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
R0 = pd.read_csv(os.path.join(PAPER, "figures", "data", "battery_events.csv.gz"))
R0["cl"] = R0.Household + "|" + R0.Day.astype(str)
R0["Ev"] = R0.EventSet + "|" + R0.EventID.astype(str)          # event ids repeat across sets
LOG = open(os.path.join(PAPER, "verification_code", "battery_stats_output.txt"), "w", encoding="utf-8")
NB = 4000
rng = np.random.default_rng(31)
MAC = {}
ORDER = ["Point only", "SCP", "LACP", "CC-CP", "CC-LACP", "Rolling LACP", "Rolling CC-LACP", "ACI", "ACI-CC",
         "ACI-LACP", "ACI-CC-LACP"]
KEY = {"Point only": "Point", "SCP": "SCP", "LACP": "LACP", "ACI": "ACI", "CC-CP": "CCCP", "CC-LACP": "CCLACP",
       "ACI-CC": "ACICC", "ACI-LACP": "ACIL", "ACI-CC-LACP": "ACICCL", "Rolling LACP": "ROLLLACP",
       "Rolling CC-LACP": "ROLLCCLACP"}
if "ACI" not in set(R0.Method) and "ACI-CC" in set(R0.Method):   # ACI-CC selected beta = 0: identical to ACI
    R0["Method"] = R0.Method.replace({"ACI-CC": "ACI"})
    ORDER.remove("ACI-CC")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def mac(k, v):
    assert k.isalpha() and k not in MAC, k
    MAC[k] = v


def sgn(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "$-$")


# keep only events present for every method in both conditions
cnt = R0.groupby(["Household", "Ev", "Condition"]).Method.nunique()
full = cnt[cnt == R0.Method.nunique()].reset_index()
ok = full.groupby(["Household", "Ev"]).Condition.nunique()
ok = ok[ok == 2].reset_index()[["Household", "Ev"]]
R0 = R0.merge(ok, on=["Household", "Ev"])
R = R0[R0.EventSet == "extended"].copy()
RT = R0[R0.EventSet == "trip"].copy()
n_ev = R.drop_duplicates(["Household", "Ev"]).shape[0]
mac("batEvents", f"{n_ev:,}".replace(",", "{,}"))
mac("batTripEvents", str(RT.drop_duplicates(["Household", "Ev"]).shape[0]))
say("events with every method in both windows: extended", n_ev, "trip", RT.drop_duplicates(["Household", "Ev"]).shape[0])


def means(X):
    return X.groupby("Method")[["Adequacy", "Shortfall_kWh", "GridImport_kWh", "PeakImport_kW"]].mean()


S, P = R[R.Condition == "shift"], R[R.Condition == "placebo"]
SPV = S[S.ShiftType == "PV"]
STR = RT[RT.Condition == "shift"]
mS, mP, mPV, mTR = means(S), means(P), means(SPV), means(STR)
say("\ninjected, complete PV trip\n", mTR.loc[ORDER].round(4).to_string())
say("\ninjected, all types\n", mS.loc[ORDER].round(4).to_string())
say("\ninjected, PV loss\n", mPV.loc[ORDER].round(4).to_string())
say("\nplacebo\n", mP.loc[ORDER].round(4).to_string())
nob_S = S.drop_duplicates(["Household", "EventID"]).NoBatteryImport_kWh.mean()
nob_P = P.drop_duplicates(["Household", "EventID"]).NoBatteryImport_kWh.mean()
say(f"no battery grid import: injected {nob_S:.4f}, placebo {nob_P:.4f} kWh per event")


def paired(X, a, b, col):
    A = X[X.Method == a].set_index(["Household", "Ev"])[[col, "cl"]]
    Bv = X[X.Method == b].set_index(["Household", "Ev"])[col]
    d = (A[col] - Bv.loc[A.index]).values
    codes, u = pd.factorize(A.cl.values)
    s = np.bincount(codes, weights=d, minlength=len(u))
    c = np.bincount(codes, minlength=len(u)).astype(float)
    ix = rng.integers(0, len(u), (NB, len(u)))
    bs = s[ix].sum(1) / c[ix].sum(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


say("\npaired differences (method - baseline), 95% cluster bootstrap")
for X, tag in [(S, "Inj"), (SPV, "PV"), (P, "Plc"), (STR, "Trip")]:
    for a, b in [("CC-LACP", "LACP"), ("CC-LACP", "ACI"), ("ACI", "LACP"), ("CC-CP", "SCP"), ("CC-CP", "ACI"),
                 ("ACI-CC", "ACI"), ("CC-LACP", "ACI-LACP"), ("CC-LACP", "Rolling LACP"),
                 ("ACI-CC-LACP", "ACI-LACP")]:
        if a not in ORDER or b not in ORDER:
            continue
        for col, ck in [("Adequacy", "Adq"), ("Shortfall_kWh", "Sho"), ("GridImport_kWh", "Imp")]:
            p, lo, hi = paired(X, a, b, col)
            star = "" if lo <= 0 <= hi else " *"
            say(f"  {tag:3s} {a:8s} - {b:5s} {col:15s} {p:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")
            k = f"bat{tag}{ck}{KEY[a]}vs{KEY[b]}"
            mac(k, sgn(p, 3 if ck == "Adq" else 3))
            mac(k + "CI", f"[{sgn(lo)}, {sgn(hi)}]")
            mac(k + "Abs", f"{abs(p):.3f}")

for m in ORDER:
    k = KEY[m]
    mac(f"batAdq{k}", f"{mS.loc[m, 'Adequacy']:.3f}")
    mac(f"batAdqPV{k}", f"{mPV.loc[m, 'Adequacy']:.3f}")
    mac(f"batAdqPlc{k}", f"{mP.loc[m, 'Adequacy']:.3f}")
    mac(f"batAdqTrip{k}", f"{mTR.loc[m, 'Adequacy']:.3f}")
    mac(f"batSho{k}", f"{mS.loc[m, 'Shortfall_kWh']:.3f}")
    mac(f"batImp{k}", f"{mS.loc[m, 'GridImport_kWh']:.3f}")
    mac(f"batImpPlc{k}", f"{mP.loc[m, 'GridImport_kWh']:.3f}")
mac("batImpNone", f"{nob_S:.3f}")
mac("batImpNonePlc", f"{nob_P:.3f}")

SL = S[S.ShiftType == "LOAD"]
mL = means(SL)
say("\ninjected, load steps\n", mL.loc[ORDER].round(4).to_string())
for m in ORDER:
    mac(f"batAdqLoad{KEY[m]}", f"{mL.loc[m, 'Adequacy']:.3f}")
ints = [m for m in ORDER if m != "Point only"]
mac("batAdqIntMin", f"{mS.loc[ints, 'Adequacy'].min():.2f}")
mac("batAdqIntMax", f"{mS.loc[ints, 'Adequacy'].max():.2f}")
imp_rise = mS.loc[ints, "GridImport_kWh"].max() / mS.loc["Point only", "GridImport_kWh"] - 1
mac("batImpRise", f"{100 * imp_rise:.1f}\\%")
say(f"grid import of interval reserves at most {100 * imp_rise:.1f}% above the point-forecast reserve")

# Table IV body: highest RAR and lowest shortfall per column in bold (interval reserves only)
def cell(v, col_vals, best="max", nd=3):
    b = max(col_vals) if best == "max" else min(col_vals)
    s = f"{v:.{nd}f}"
    return f"\\textbf{{{s}}}" if abs(v - b) < 1e-12 else s


cols = [(mPV, "Adequacy", "max"), (mL, "Adequacy", "max"), (mTR, "Adequacy", "max"), (mS, "Adequacy", "max"),
        (mS, "Shortfall_kWh", "min"), (mS, "GridImport_kWh", "min"), (mP, "Adequacy", "max")]
lines = [r"\begin{tabular}{|l|c|c|c|c|c|c|c|}", r"\hline",
         r"\multirow{2}{*}{Reserve from} & \multicolumn{4}{c|}{Adequacy, injected window} & \multirow{2}{*}{\hdb{Short-\\fall}} & \multirow{2}{*}{\hdb{Grid\\import}} & \multirow{2}{*}{\hdb{Adequacy,\\control}}\\",
         r"\cline{2-5}",
         r" & \hdb{PV\\loss} & \hdb{Load\\step} & \hdb{PV\\trip} & All"
         r" & & & \\",
         r"\hline",
         r"No battery & -- & -- & -- & -- & -- & " + f"{nob_S:.3f}" + r" & --\\"]
for m in ORDER:
    cells = []
    for tab, c, best in cols:
        v = tab.loc[m, c]
        cells.append(f"{v:.3f}" if best is None or m == "Point only" else cell(v, tab.loc[ints, c], best))
    if m == "CC-LACP":
        lines.append(r"\rowcolor{black!6}")
    show = {"ACI-LACP": "ACI on LACP", "ACI-CC-LACP": "ACI on CC-LACP"}.get(m, m)
    lines.append(f"{show} & " + " & ".join(cells) + r"\\")
lines += [r"\hline", r"\end{tabular}"]
open(os.path.join(PAPER, "tables", "table_battery.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
with open(os.path.join(PAPER, "numbers_battery.tex"), "w", encoding="utf-8") as f:
    f.write("% generated by analysis/battery_stats.py -- do not edit\n")
    for k, v in MAC.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
say(f"{len(MAC)} macros")
LOG.close()
