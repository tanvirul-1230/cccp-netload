"""
Event-time trace of the main intervals around each extended test event
(tree/linear backbone, protocol v3, selection of v3_tree), injected and placebo window.

For every event, horizon, method and issue time t with t - t_s in
[-120, +240] min it stores the width, whether the target was covered, the
interval score divided by R, and the target offset. The construction of the
intervals is copied from engine2._eval_task, so the inside-event sums must
reproduce event_rows_test.csv.gz (checked at the end).

  python event_trace.py            -> ../figures/data/event_trace.csv.gz
"""
import io
import json
import os
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
AA = os.path.abspath(os.path.join(PAPER, "..", "Additional_Analyses"))
sys.path.insert(0, AA)
import her_shift as hs      # noqa: E402
import engine as en         # noqa: E402
import engine2 as e2        # noqa: E402
import engine3 as e3        # noqa: E402

warnings.filterwarnings("ignore")
ZIP = os.environ.get("HEMS_ZIP") or next(p for p in (os.path.join(AA, "..", "data", "hems_data.zip"),
                            os.path.join(AA, "..", "Conference", "Conference",
                                         "HEMS_CP_Paper_COMPLETE_Day7_Backup.zip")) if os.path.exists(p))
F2 = os.path.join(AA, "final3", "v3_tree")
SEL = json.load(open(os.path.join(F2, "selection.json")))["selected"]
KEEP = {"SCP": "SCP", "LACP": "LACP", "ACI": SEL["ACI"], "ACI-tuned": "ACI|tuned", "CC-CP": SEL["CC"],
        "CC-LACP": SEL["CCLACP"], "ACI-CC": SEL["ACICC"], "ACI-LACP": SEL["ACIL"], "ACI-CC-LACP": SEL["ACICCL"],
        "Rolling LACP": "ROLL|LACP", "Rolling CC-LACP": SEL["ROLLCCL"]}
HORIZONS = {"15min": 1, "30min": 2, "60min": 4}
HOMES = {"H3": "residential3", "H4": "residential4", "H6": "residential6"}
NS_MIN = 60 * 10 ** 9
OUT = os.path.join(PAPER, "figures", "data", "event_trace.csv.gz")
z = zipfile.ZipFile(ZIP)


def trace_task(H, K, event, shifted, win, rho_w, w0, w1, start, stop):
    common = K["common"]
    lo_pos = int(common.searchsorted(w0))
    hi_pos = int(common.searchsorted(w1, side="right"))
    widx = common[lo_pos:hi_pos]
    posarr = np.arange(lo_pos, hi_pos)
    steps = K["steps"]
    if shifted:
        f = hs.make_features(win, steps).dropna(subset=e2.FE + ["target", "target_time"])
        m = widx.isin(f.index)
        idx, p = widx[m], posarr[m]
        f = f.loc[idx]
        pred = K["model"].predict(f[e2.FE])
        scale = np.maximum(K["scale_model"].predict(f[e2.FE]), K["floor"])
        y = f["target"].values
        tt = pd.DatetimeIndex(f["target_time"])
    else:
        idx, p = widx, posarr
        pred, scale, y = K["pred"][p], K["scale"][p], K["y"][p]
        tt = K["target_time"][p]
    if len(idx) == 0:
        return []
    p0 = int(common.searchsorted(start))
    first_snaps = next(iter(K["aci"].values()))[4]
    if shifted and p0 not in first_snaps:
        return []
    ti = pd.DatetimeIndex(idx).as_unit("ns").asi8
    tti = pd.DatetimeIndex(tt).as_unit("ns").asi8
    s_ns, e_ns = pd.Timestamp(start).as_unit("ns").value, pd.Timestamp(stop).as_unit("ns").value
    M = {"SCP": (pred - K["q_scp"], pred + K["q_scp"]),
         "LACP": (pred - K["q_lacp"] * scale, pred + K["q_lacp"] * scale)}
    rw, sig_w = {}, {}
    for v, (comps, gains, base, q) in K["sig_cfg"].items():
        if v not in KEEP.values() and not any(sv == v for sv, *_ in K["aci"].values()):
            continue
        for c in comps:
            if c not in rw:
                rw[c] = rho_w[c].reindex(idx).values
        s = 1 + sum(g * rw[c] for g, c in zip(gains, comps))
        if base == "lacp":
            s = s * scale
        sig_w[v] = s
        M[v] = (pred - q * s, pred + q * s)
    pos_i = {int(q_): i for i, q_ in enumerate(p)}
    if shifted:
        p_end = int(p.max())
        if any(q_ not in pos_i for q_ in range(max(p0 - steps, 0), p_end + 1)):
            return []
    for a_name, (sv, g, lo_c, hi_c, snaps) in K["aci"].items():
        if a_name not in KEEP.values():
            continue
        if shifted:
            sg = sig_w[sv] if sv is not None else np.ones(len(idx))
            lr, hr = e2.aci_resume(snaps[p0], p0, p_end, pos_i, y, pred, sg, lo_c, hi_c, steps, g)
            lo = np.array([lr.get(int(q_), lo_c[q_]) for q_ in p])
            hi = np.array([hr.get(int(q_), hi_c[q_]) for q_ in p])
        else:
            lo, hi = lo_c[p], hi_c[p]
        M[a_name] = (lo, hi)
    off = (ti - s_ns) / NS_MIN
    toff = (tti - s_ns) / NS_MIN
    keep = (off >= -120) & (off <= 240)
    dur = (e_ns - s_ns) / NS_MIN
    rows = []
    for lab, name in KEEP.items():
        if name not in M:
            continue
        lo, hi = M[name]
        w = hi - lo
        cov = (y >= lo) & (y <= hi)
        sc = w + (2 / e2.ALPHA) * (lo - y) * (y < lo) + (2 / e2.ALPHA) * (y - hi) * (y > hi)
        rows.append(pd.DataFrame(dict(Method=lab, off=off[keep], toff=toff[keep], dur=dur, w=w[keep] / K["R"],
                                      cov=cov[keep].astype(np.int8), isr=sc[keep] / K["R"])))
    return rows


t0 = time.time()
frames = []
CHECK_ONLY = "--check" in sys.argv          # re-run only the reproduction check on the stored trace
B = None if CHECK_ONLY else en.Builders("cuda")
for h, name in ({} if CHECK_ONLY else HOMES).items():
    df = hs.regularize_15min(pd.read_csv(io.BytesIO(z.read(f"data/processed/{name}_clean.csv"))))
    df["net_load_kw"] = df["load_kw"] - df["pv_kw"]
    ev = pd.read_csv(os.path.join(AA, "final", "events", f"{h}_extended_events.csv"))
    for c in ["start", "end"]:
        ev[c] = pd.to_datetime(ev[c], utc=True)
    H = e3.prepare_household3(df, h, B)
    tasks = {hz: e3.prepare_task3(H, hz, st, B, "test", ev["start"]) for hz, st in HORIZONS.items()}
    print(f"{h}: tasks ready ({time.time() - t0:.0f}s)", flush=True)
    for _, e in ev.iterrows():
        start, stop = e["start"], hs.event_stop_time(e)
        w0 = start - pd.Timedelta(days=2)
        for K in tasks.values():
            p0 = int(K["common"].searchsorted(start))
            w0 = min(w0, K["common"][max(min(p0, len(K["common"]) - 1) - 12, 0)] - pd.Timedelta(hours=27))
        w1 = stop + pd.Timedelta(hours=3.5) + pd.Timedelta(minutes=60)
        win = e3.apply_event3(df.loc[w0:w1], e)
        rho_s, _ = en.rho_all(win, H["twins"], H["S_clean"])
        e3.add_thr(rho_s, H["thr"])
        for shifted, rho_w in ((True, rho_s), (False, H["rho_clean"])):
            for hz, K in tasks.items():
                for fr in trace_task(H, K, e, shifted, win if shifted else None, rho_w, w0, w1, start, stop):
                    fr["Household"], fr["Horizon"], fr["EventID"] = h, hz, int(e["event_id"])
                    fr["ShiftType"], fr["Severity"] = e["shift_type"], float(e["severity"])
                    fr["Condition"] = "shift" if shifted else "placebo"
                    fr["Day"] = str(start.tz_convert("Europe/Berlin").date())
                    frames.append(fr)
    print(f"{h}: events traced ({time.time() - t0:.0f}s)", flush=True)
    del H, tasks
if CHECK_ONLY:
    T = pd.read_csv(OUT)
else:
    T = pd.concat(frames, ignore_index=True)
    T.to_csv(OUT, index=False, compression="gzip")
    print("rows", len(T), "written", OUT)
if not os.path.exists(os.path.join(F2, "event_rows_test.csv.gz")):
    sys.exit("event rows of the test run not yet available; run with --check afterwards")

# check: inside-event sums reproduce the stored event rows
ins = T[(T.off >= 0) & (T.toff < T.dur)]
agg = ins.groupby(["Household", "Horizon", "EventID", "Condition", "Method"]).agg(IS=("isr", "sum"), N=("isr", "size"),
                                                                                   C=("cov", "sum")).reset_index()
ER = pd.read_csv(os.path.join(F2, "event_rows_test.csv.gz"))
ER = ER[ER.EventSet == "extended"]
ER["IS"] = ER.IS_in / ER.R
inv = {v: k for k, v in KEEP.items()}
ER = ER[ER.Method.isin(KEEP.values())].copy()
ER["Method"] = ER.Method.map(inv)
mg = agg.merge(ER[["Household", "Horizon", "EventID", "Condition", "Method", "IS", "N_in", "Cov_in"]],
               on=["Household", "Horizon", "EventID", "Condition", "Method"])
mg = mg[mg.N_in > 0]
print("matched pairs", len(mg), "max |IS diff|", float((mg.IS_x - mg.IS_y).abs().max()),
      "N equal", bool((mg.N == mg.N_in).all()), "coverage equal", bool((mg.C == mg.Cov_in).all()))
