"""
Review round 1, Reviewer 3 (H3): numbers for the paragraph on where the consistency factor pays.

1. Inside-event score of CC-LACP against LACP per household, split into the width part and the
   miss-penalty part of the interval score (IS = width + penalty), and per event type for H3.
2. Event cells (household x type x severity x duration x horizon): change of the LACP score
   against the in-event coverage of LACP; rank correlation on test and development events,
   weighted linear crossing, and pooled bins with a household-day cluster bootstrap.

Extended test events and development events of protocol v3, tree backbone, selected gains.
Writes ../numbers_review.tex and ../verification_code/review1_numbers_output.txt
  python -I review1_numbers.py   (SmartGirlU/.venv python)
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
AA = os.path.abspath(os.path.join(PAPER, "..", "Additional_Analyses"))
D = os.path.join(AA, "final3", "v3_tree")
LOG = open(os.path.join(PAPER, "verification_code", "review1_numbers_output.txt"), "w", encoding="utf-8")
rng = np.random.default_rng(2026)
NB = 4000
MAC = {}
HOUSES = ["H3", "H4", "H6"]
WORD = {"H3": "HThree", "H4": "HFour", "H6": "HSix"}
CELL = ["Household", "ShiftType", "Severity", "DurationMin", "Horizon"]
EDGES = [0, 0.75, 0.85, 0.9, 1.01]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def mac(name, value):
    assert name.isalpha() and name not in MAC, name
    MAC[name] = value


def pct(x, nd=1):
    return f"{100 * x:+.{nd}f}".replace("-", "$-$") + "\\%"


def upct(x, nd=0):
    return f"{100 * x:.{nd}f}\\%"


CCL = json.load(open(os.path.join(D, "selection.json")))["selected"]["CCLACP"]
say("selected CC-LACP:", CCL)


def rows(split):
    E = pd.read_csv(os.path.join(D, f"event_rows_{split}.csv.gz"))
    E = E[(E.Condition == "shift") & E.Method.isin(["LACP", CCL])].copy()
    if split == "test":
        E = E[E.EventSet == "extended"]
    E["isr"] = E.IS_in / E.R
    E["wr"] = E.W_in / E.R
    E["pr"] = (E.IS_in - E.W_in) / E.R
    E["cl"] = E.Household + "|" + E.Day.astype(str)
    return E


# ---------------------------------------------------------------- 1. width / penalty split
T = rows("test")
g = T.groupby(["Household", "Method"])[["isr", "wr", "pr", "Cov_in", "N_in"]].sum()
say("\nper household, fractions of the LACP inside-event score")
for h in HOUSES:
    a, b = g.loc[(h, CCL)], g.loc[(h, "LACP")]
    rel, dw, dp = a.isr / b.isr - 1, (a.wr - b.wr) / b.isr, (a.pr - b.pr) / b.isr
    cov, share = b.Cov_in / b.N_in, b.pr / b.isr
    say(f"  {h}: rel {rel:+.4f}  width {dw:+.4f}  penalty {dp:+.4f}  LACP coverage {cov:.3f}  "
        f"penalty share {share:.3f}")
    mac(f"covLACPin{WORD[h]}", f"{cov:.2f}")
    mac(f"penShare{WORD[h]}", upct(share))
    mac(f"dWidth{WORD[h]}", pct(dw))
    mac(f"dPen{WORD[h]}", pct(dp))
gt = T[T.Household == "H3"].groupby(["ShiftType", "Method"]).isr.sum()
say("H3 by event type (rel):")
for st, w in [("PV", "PV"), ("LOAD", "Load"), ("COMBINED", "Comb")]:
    r = gt.loc[(st, CCL)] / gt.loc[(st, "LACP")] - 1
    say(f"  {st}: {r:+.4f}")
    mac(f"relHThree{w}", pct(r))


# ---------------------------------------------------------------- 2. event cells
def cells(E):
    k = E.groupby(CELL + ["Method"])[["isr", "Cov_in", "N_in"]].sum().unstack("Method")
    C = pd.DataFrame({"cov": k[("Cov_in", "LACP")] / k[("N_in", "LACP")],
                      "rel": k[("isr", CCL)] / k[("isr", "LACP")] - 1,
                      "w": k[("isr", "LACP")]}).dropna()
    return C


C = cells(T)
Cd = cells(rows("dev"))
rho_t = C[["cov", "rel"]].corr("spearman").iloc[0, 1]
rho_d = Cd[["cov", "rel"]].corr("spearman").iloc[0, 1]
say(f"\ncells: test {len(C)}, development {len(Cd)}; Spearman test {rho_t:.3f}, development {rho_d:.3f}")
for h in HOUSES:
    say(f"  {h}: Spearman test {C.loc[h][['cov', 'rel']].corr('spearman').iloc[0, 1]:.3f}, "
        f"development {Cd.loc[h][['cov', 'rel']].corr('spearman').iloc[0, 1]:.3f}")
mac("nCells", str(len(C)))
mac("cellRho", f"{rho_t:.2f}")
mac("cellRhoDev", f"{rho_d:.2f}")
X = np.c_[np.ones(len(C)), C["cov"].values]
w = (C.w / C.w.sum()).values
coef = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * C.rel.values))
cross = -coef[0] / coef[1]
say(f"weighted fit rel = {coef[0]:+.3f} + {coef[1]:+.3f} cov, zero at {cross:.3f}")
mac("cellCross", f"{cross:.2f}")
shareHigh = (C["cov"] >= 0.85).groupby(level=0).mean()
say("share of cells with LACP coverage >= 0.85:", shareHigh.round(3).to_dict())
mac("cellHighHThree", upct(shareHigh["H3"]))

# pooled bins with household-day cluster bootstrap
T = T.merge(C[["cov"]].rename(columns={"cov": "cell_cov"}), left_on=CELL, right_index=True)
P = T.pivot_table(index=CELL + ["EventID", "cl", "cell_cov"], columns="Method", values="isr", aggfunc="sum").dropna()
P = P.reset_index()
P["bin"] = pd.cut(P.cell_cov, EDGES, right=False)
names = ["Low", "MidLow", "MidHigh", "High"]
say("\nbin of LACP cell coverage: cells, events, rel [95% CI]")
for nm, (b, G) in zip(names, P.groupby("bin", observed=True)):
    codes, u = pd.factorize(G.cl)
    sl = np.bincount(codes, weights=G["LACP"].values)
    sc = np.bincount(codes, weights=G[CCL].values)
    ix = rng.integers(0, len(u), (NB, len(u)))
    bs = sc[ix].sum(1) / sl[ix].sum(1) - 1
    r, lo, hi = sc.sum() / sl.sum() - 1, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    nc = ((C["cov"] >= b.left) & (C["cov"] < b.right)).sum()
    say(f"  {str(b):14s} {nc:4d} {len(G):6d} {r:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    mac(f"bin{nm}", pct(r))
    mac(f"bin{nm}CI", f"[{pct(lo)}, {pct(hi)}]")

with open(os.path.join(PAPER, "numbers_review.tex"), "w", encoding="utf-8") as f:
    f.write("% generated by analysis/review1_numbers.py\n")
    for k, v in MAC.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
say(f"\n{len(MAC)} macros written")
LOG.close()
