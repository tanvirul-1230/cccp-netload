"""
Selection on development data only (development split, protocol v3). Same rule as
select_params.py, fixed before any test event is evaluated:
  selected = argmin development inside-event nIS subject to development-period
  clean nIS <= 1.02 x family minimum; ties go to the smaller parameter.
Also writes the leave-one-household-out selection (for each household, the rule
applied to the development events of the other two households).

  python select_params3.py v3_tree
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine3 as e3        # noqa: E402

e2 = e3.e2
TAG = sys.argv[1]
D = os.path.join(os.environ.get("FINAL3_ROOT", os.path.join(HERE, "final3")), TAG)
TOL = 0.02
G = [f"{g:g}" for g in e2.GAMMAS]
FAMILIES = {
    "ACI": [f"ACI|g{g}" for g in G],
    "CC": [f"CC|b{b:g}" for b in e2.BETAS],
    "CCLACP": [f"CCLACP|b{b:g}" for b in e2.BETAS_LACP],
    "ACICC": [f"ACI|g{g}" for g in G] + [f"ACICC|b{b:g}|g{g}" for b in e2.ACICC_BETAS for g in G],
    "ACIL": [f"ACIL|g{g}" for g in G],
    "ACICCL": [f"ACIL|g{g}" for g in G] + [f"ACICCL|b{b:g}|g{g}" for b in e3.BETAS_CCL for g in G],
    "ROLLCCL": ["ROLL|LACP"] + [f"ROLL|CCLACP|b{b:g}" for b in e3.BETAS_CCL],
    "PVonlyL": [f"PVonlyL|b{b:g}" for b in e2.BETAS_LACP],
    "PersistL": [f"PersistL|b{b:g}" for b in e2.BETAS_LACP],
    "THL": ["CCLACP|b0"] + [f"THL|{l}|b{b:g}" for l in e3.THR_LEVELS for b in e3.BETAS_TH],
}
for s in ["Persist", "PVonly", "LoadOnly", "LF-load", "LF-both", "LoadOnly-LF"]:
    FAMILIES[s] = [f"{s}|b{b:g}" for b in e2.BETAS]

C = pd.read_csv(os.path.join(D, "clean_metrics_dev.csv"))
E = pd.read_csv(os.path.join(D, "event_rows_dev.csv.gz"))
E = E[E.Condition == "shift"]


def scores(Cx, Ex):
    clean = Cx.groupby("Method").apply(lambda g: (g.nIS * g.N).sum() / g.N.sum())
    dev = Ex.groupby("Method").apply(lambda g: (g.IS_in / g.R).sum() / g.N_in.sum())
    return clean, dev


def pick(cands, clean, dev, tol):
    cl = clean.loc[cands]
    ok = [c for c in cands if cl[c] <= (1 + tol) * cl.min()]
    return min(ok, key=lambda c: (round(dev[c], 10), cands.index(c)))


clean, dev = scores(C, E)
devpicp = E.groupby("Method").apply(lambda g: g.Cov_in.sum() / g.N_in.sum())
rows, selected, sens, loho = [], {}, {}, {}
for fam, cands in FAMILIES.items():
    selected[fam] = pick(cands, clean, dev, TOL)
    for tol in (0.01, 0.05):
        sens.setdefault(f"tol={tol}", {})[fam] = pick(cands, clean, dev, tol)
    mn = clean.loc[cands].min()
    for c in cands:
        rows.append(dict(Family=fam, Candidate=c, CleanNIS=clean[c], CleanRatio=clean[c] / mn,
                         Admissible=clean[c] <= (1 + TOL) * mn, DevNIS=dev[c], DevPICP=devpicp[c],
                         Selected=c == selected[fam]))
for h in sorted(E.Household.unique()):
    ch, dh = scores(C[C.Household != h], E[E.Household != h])
    loho[h] = {fam: pick(cands, ch, dh, TOL) for fam, cands in FAMILIES.items()}
T = pd.DataFrame(rows)
T.to_csv(os.path.join(D, "selection_table.csv"), index=False)
n_ev = E.drop_duplicates(["Household", "EventID"]).shape[0]
with open(os.path.join(D, "selection.json"), "w") as f:
    json.dump(dict(rule="argmin development inside-event nIS s.t. development-period clean nIS <= "
                        "(1+tol) x family minimum; ties to the smaller parameter",
                   tol=TOL, n_dev_events=int(n_ev), selected=selected, sensitivity=sens,
                   leave_one_household_out=loho), f, indent=2)
pd.set_option("display.width", 220)
print(f"development events: {n_ev}")
print("selected:", json.dumps(selected, indent=1))
print("leave-one-household-out:", json.dumps(loho, indent=1))
