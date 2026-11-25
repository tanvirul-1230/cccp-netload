"""Checks behind the points taken from the correction guide of 2026-11-24 (C06-C21): each statement added to
the paper is checked against the code that produced the results, not against the text.

  python verify_correction_guide.py   -> verify_correction_guide_output.txt
"""
import ast
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
AA = os.path.abspath(os.path.join(PAPER, "..", "Additional_Analyses"))
sys.path.insert(0, AA)
import her_shift as hs  # noqa: E402

OUT = open(os.path.join(HERE, "verify_correction_guide_output.txt"), "w", encoding="utf-8")
ok_all = True


def check(label, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    line = f"[{'OK ' if cond else 'FAIL'}] {label}  {detail}"
    print(line)
    OUT.write(line + "\n")


def load_function(path, names):
    """Top-level assignments and functions of a script that runs its pipeline on import."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    ns = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


# C06: normal-behaviour model inputs = lags 1, 2, 4, 96 and means over t-4..t-1 and t-8..t-1
idx = pd.date_range("2017-01-01", periods=300, freq="15min", tz="UTC")
x = pd.DataFrame({"pv_kw": np.arange(300.0), "load_kw": np.arange(300.0) ** 1.5}, index=idx)
f = hs.make_twin_features(x)
t = 200
ok = all(f[f"pv_kw_past_{k}"].iloc[t] == x.pv_kw.iloc[t - k] for k in (1, 2, 4, 96))
ok &= np.isclose(f["load_kw_past_roll_4"].iloc[t], x.load_kw.iloc[t - 4:t].mean())
ok &= np.isclose(f["load_kw_past_roll_8"].iloc[t], x.load_kw.iloc[t - 8:t].mean())
check("C06 inputs: steps t-1, t-2, t-4, t-96 and means over the previous four and eight steps", ok)

# C08: load step = s x median load over the 24 h before the onset, constant over the event
rng = np.random.default_rng(0)
x = pd.DataFrame({"pv_kw": rng.random(300), "load_kw": rng.random(300) + 0.2}, index=idx)
start, end = idx[150], idx[157]
ev = pd.DataFrame([dict(event_id=0, shift_type="LOAD", severity=0.5, start=start, end=end)])
y = hs.apply_event_catalog(x, ev)
added = (y.load_kw - x.load_kw).loc[start:end]
base = x.load_kw[(x.index >= start - pd.Timedelta(hours=24)) & (x.index < start)].median()
check("C08 load step: 0.5 x median over [onset - 24 h, onset), constant over the event",
      np.allclose(added.values, 0.5 * base), f"{added.iloc[0]:.4f} vs {0.5 * base:.4f}")

# C09: held-out sets, 20 onsets per household and duration
Et = pd.read_csv(os.path.join(AA, "final3", "v3_tree", "event_rows_test.csv.gz"),
                 usecols=["Household", "EventID", "DurationMin", "EventSet", "Condition"])
for es in ("trip", "ev"):
    g = (Et[(Et.EventSet == es) & (Et.Condition == "shift")].drop_duplicates(["Household", "EventID"])
         .groupby(["Household", "DurationMin"]).size())
    check(f"C09 held-out set '{es}': 20 per household and duration, 180 in total",
          (g == 20).all() and g.sum() == 180 and len(g) == 9, str(g.to_dict()))

# C13: admit within 2 % of the family's best normal-operation score, then lowest in-event score
ns = load_function(os.path.join(AA, "select_params3.py"), {"pick"})
cands = ["b0", "b0.25", "b0.5"]
clean = pd.Series({"b0": 1.00, "b0.25": 1.015, "b0.5": 1.05})
dev = pd.Series({"b0": 0.50, "b0.25": 0.45, "b0.5": 0.30})
check("C13 rule order: b0.5 (lowest in-event) excluded by the 2 % rule, b0.25 selected",
      ns["pick"](cands, clean, dev, 0.02) == "b0.25")

# C17: adequacy = 1{ eta (e_t - e_min) >= min([y_{t+1}]^+, P) dt }, e_t after dispatch
ns = load_function(os.path.join(PAPER, "analysis", "battery_reserve.py"), {"simulate", "BATTERY", "DT_H"})
sim, B, dt = ns["simulate"], ns["BATTERY"], ns["DT_H"]
n = 400
net = rng.normal(0.3, 1.2, n)
nxt = np.r_[net[1:], 0.0] + rng.normal(0, 0.4, n)
res = np.abs(nxt) + rng.normal(0, 0.3, n)
imp, dis, chg, short, adeq = sim(net, nxt, res)
# energy after dispatch, rebuilt with the same rule
cap = B["capacity_kwh"]
e_min, e_max = cap * B["soc_min"], cap * B["soc_max"]
e, eta, P = cap * B["soc_initial"], B["eta_discharge"], B["p_discharge_max_kw"]
formula = np.zeros(n)
for k in range(n):
    e_res = np.clip(e_min + max(res[k], 0) * dt / eta * B["reserve_factor"], e_min, e_max)
    if net[k] < 0:
        e += min(-net[k], B["p_charge_max_kw"], max((e_max - e) / (B["eta_charge"] * dt), 0)) * dt * B["eta_charge"]
    else:
        e -= min(net[k], P, max(max(e - e_res, 0) * eta / dt, 0)) * dt / eta
    e = float(np.clip(e, e_min, e_max))
    formula[k] = float(eta * (e - e_min) >= min(max(nxt[k], 0), P) * dt - 1e-9)
check("C17 adequacy inequality reproduces simulate() on 400 random steps", np.array_equal(formula, adeq),
      f"mean {adeq.mean():.3f}")

# C19: the unscored onset of the second event set lies after a data gap that covers its day-lag inputs
ZIP = next(p for p in (os.path.join(PAPER, "..", "data", "hems_data.zip"),
                       os.path.join(PAPER, "..", "Conference", "Conference",
                                    "HEMS_CP_Paper_COMPLETE_Day7_Backup.zip")) if os.path.exists(p))
z = zipfile.ZipFile(ZIP)
fr = pd.read_csv(io.BytesIO(z.read("results/day5_events/H3_events.csv")))
miss = fr[fr.event_id == 1].iloc[0]
s0 = pd.Timestamp(miss["start"])
s0 = s0.tz_localize("UTC") if s0.tzinfo is None else s0.tz_convert("UTC")
df = hs.regularize_15min(pd.read_csv(io.BytesIO(z.read("data/processed/residential3_clean.csv"))))
w = df.loc[s0 - pd.Timedelta(hours=26):s0, "net_load_kw"]
scored = Et[(Et.EventSet == "frozen") & (Et.Condition == "shift")].drop_duplicates(["Household", "EventID"])
check("C19 second set: H3 event 1 is the unscored onset and its 26-h history has a data gap",
      len(scored) == 161 and not ((scored.Household == "H3") & (scored.EventID == 1)).any()
      and w.isna().sum() > 0, f"{int(w.isna().sum())} missing 15-min rows before {s0}")

# C21: the OPSD URL carries the underscore (text extraction of the PDF drops it)
bib = open(os.path.join(PAPER, "refs.bib"), encoding="utf-8").read()
check("C21 OPSD URL is household_data/2020-04-15/", "open-power-system-data.org/household_data/2020-04-15/" in bib)

print("ALL OK" if ok_all else "SOME CHECKS FAILED")
OUT.write(("ALL OK" if ok_all else "SOME CHECKS FAILED") + "\n")
OUT.close()
