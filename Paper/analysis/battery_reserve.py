"""
Battery reserve simulation on the corrected pipeline (15-min horizon, extended events,
tree/linear backbone, protocol v3, selection of v3_tree), injected and placebo window;
extended events and the complete-trip set (s = 1).

The battery model and dispatch rule are those of the original study
(Conference/Conference/06_battery_decision_support.ipynb, cell 20), unchanged:
  5 kWh, 2.5 kW charge/discharge, SOC 10-90 %, initial SOC 50 %, efficiency 0.95,
  reserve factor 1, 15-min control step.
At issue time t the battery keeps e_min + max(u_t, 0) * dt / eta_d, where u_t is the
upper bound of the interval for net load at t+1 (the point forecast for "point only"),
serves the present deficit only from energy above that reserve, and charges from PV
surplus. The reserve is adequate at t when the energy above e_min covers the actual
net load of the next step (within the power limit). The simulation runs from 6 h
before the onset to 2 h after the event, and metrics are taken over [t_s, t_s + d),
as in the original.

  python battery_reserve.py     -> ../figures/data/battery_events.csv.gz
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
KEEP = {"SCP": "SCP", "LACP": "LACP", "ACI": SEL["ACI"], "CC-CP": SEL["CC"], "CC-LACP": SEL["CCLACP"],
        "ACI-CC": SEL["ACICC"], "ACI-LACP": SEL["ACIL"], "ACI-CC-LACP": SEL["ACICCL"],
        "Rolling LACP": "ROLL|LACP", "Rolling CC-LACP": SEL["ROLLCCL"]}
HOMES = {"H3": "residential3", "H4": "residential4", "H6": "residential6"}
DT_H = 0.25
BATTERY = dict(capacity_kwh=5.0, p_charge_max_kw=2.5, p_discharge_max_kw=2.5, soc_min=0.10, soc_max=0.90,
               soc_initial=0.50, eta_charge=0.95, eta_discharge=0.95, reserve_factor=1.0)
OUT = os.path.join(PAPER, "figures", "data", "battery_events.csv.gz")
z = zipfile.ZipFile(ZIP)


def simulate(net, target_next, reserve_kw, p=BATTERY):
    """Original dispatch (notebook 06, cell 20). Returns per-step arrays."""
    cap = p["capacity_kwh"]
    e_min, e_max = cap * p["soc_min"], cap * p["soc_max"]
    energy = cap * p["soc_initial"]
    eta_c, eta_d = p["eta_charge"], p["eta_discharge"]
    p_c, p_d = p["p_charge_max_kw"], p["p_discharge_max_kw"]
    n = len(net)
    imp, dis, chg, short, adeq = (np.zeros(n) for _ in range(5))
    for k in range(n):
        risk = max(float(reserve_kw[k]), 0.0)
        reserve_energy = np.clip(e_min + risk * DT_H / eta_d * p["reserve_factor"], e_min, e_max)
        if net[k] < 0:
            room = (e_max - energy) / (eta_c * DT_H)
            c = min(-net[k], p_c, max(room, 0.0))
            energy += c * DT_H * eta_c
            chg[k] = c
        else:
            avail = max(energy - reserve_energy, 0.0)
            d = min(net[k], p_d, max(avail * eta_d / DT_H, 0.0))
            energy -= d * DT_H / eta_d
            dis[k] = d
            imp[k] = max(net[k] - d, 0.0)
        energy = float(np.clip(energy, e_min, e_max))
        need = min(max(target_next[k], 0.0), p_d) * DT_H
        have = min(max(energy - e_min, 0.0) * eta_d, p_d * DT_H)
        short[k] = max(need - have, 0.0)
        adeq[k] = float(short[k] <= 1e-9)
    return imp, dis, chg, short, adeq


def task_bounds(K, event, shifted, win, rho_w, w0, w1, start):
    """Upper bounds of the 15-min intervals over the window (construction of engine2._eval_task)."""
    common = K["common"]
    lo_pos, hi_pos = int(common.searchsorted(w0)), int(common.searchsorted(w1, side="right"))
    widx, posarr = common[lo_pos:hi_pos], np.arange(lo_pos, hi_pos)
    if shifted:
        f = hs.make_features(win, K["steps"]).dropna(subset=e2.FE + ["target", "target_time"])
        m = widx.isin(f.index)
        idx, p = widx[m], posarr[m]
        f = f.loc[idx]
        pred = K["model"].predict(f[e2.FE])
        scale = np.maximum(K["scale_model"].predict(f[e2.FE]), K["floor"])
        y = f["target"].values
        net_now = f["net_load_kw"].values
    else:
        idx, p = widx, posarr
        pred, scale, y = K["pred"][p], K["scale"][p], K["y"][p]
        net_now = None
    p0 = int(common.searchsorted(start))
    first_snaps = next(iter(K["aci"].values()))[4]
    if shifted and p0 not in first_snaps:
        return None
    U = {"Point only": pred, "SCP": pred + K["q_scp"], "LACP": pred + K["q_lacp"] * scale}
    sig_w, rw = {}, {}
    for v, (comps, gains, base, q) in K["sig_cfg"].items():
        for c in comps:
            if c not in rw:
                rw[c] = rho_w[c].reindex(idx).values
        s = 1 + sum(g * rw[c] for g, c in zip(gains, comps))
        s = s * scale if base == "lacp" else s
        sig_w[v] = s
        if v in KEEP.values():
            U[v] = pred + q * s
    pos_i = {int(q_): i for i, q_ in enumerate(p)}
    if shifted:
        p_end = int(p.max())
        if any(q_ not in pos_i for q_ in range(max(p0 - K["steps"], 0), p_end + 1)):
            return None
    for a_name, (sv, g, lo_c, hi_c, snaps) in K["aci"].items():
        if a_name not in KEEP.values():
            continue
        if shifted:
            sg = sig_w[sv] if sv is not None else np.ones(len(idx))
            lr, hr = e2.aci_resume(snaps[p0], p0, p_end, pos_i, y, pred, sg, lo_c, hi_c, K["steps"], g)
            U[a_name] = np.array([hr.get(int(q_), hi_c[q_]) for q_ in p])
        else:
            U[a_name] = hi_c[p]
    return idx, y, net_now, U


t0 = time.time()
rows = []
B = en.Builders("cuda")
for h, name in HOMES.items():
    df = hs.regularize_15min(pd.read_csv(io.BytesIO(z.read(f"data/processed/{name}_clean.csv"))))
    df["net_load_kw"] = df["load_kw"] - df["pv_kw"]
    ev = pd.read_csv(os.path.join(AA, "final", "events", f"{h}_extended_events.csv")).assign(EventSet="extended")
    tr = pd.read_csv(os.path.join(AA, "final3", "events_stress", f"{h}_trip_events.csv")).assign(EventSet="trip")
    ev = pd.concat([ev, tr], ignore_index=True)
    for c in ["start", "end"]:
        ev[c] = pd.to_datetime(ev[c], utc=True)
    H = e3.prepare_household3(df, h, B)
    K = e3.prepare_task3(H, "15min", 1, B, "test", ev["start"])
    print(f"{h}: task ready ({time.time() - t0:.0f}s)", flush=True)
    inv = {v: k for k, v in KEEP.items()}
    for _, e in ev.iterrows():
        start, stop = e["start"], hs.event_stop_time(e)
        p0 = int(K["common"].searchsorted(start))
        w0 = min(start - pd.Timedelta(days=2),
                 K["common"][max(min(p0, len(K["common"]) - 1) - 12, 0)] - pd.Timedelta(hours=27))
        w1 = stop + pd.Timedelta(hours=3.5) + pd.Timedelta(minutes=60)
        win = e3.apply_event3(df.loc[w0:w1], e)
        rho_s, _ = en.rho_all(win, H["twins"], H["S_clean"])
        e3.add_thr(rho_s, H["thr"])
        for shifted, rho_w in ((True, rho_s), (False, H["rho_clean"])):
            out = task_bounds(K, e, shifted, win if shifted else None, rho_w, w0, w1, start)
            if out is None:
                continue
            idx, y, net_now, U = out
            if net_now is None:
                net_now = df["net_load_kw"].reindex(idx).values
            sim = (idx >= start - pd.Timedelta(hours=6)) & (idx < stop + pd.Timedelta(hours=2))
            ev_mask = (idx[sim] >= start) & (idx[sim] < stop)
            if ev_mask.sum() == 0:
                continue
            n_sim, y_sim = net_now[sim], y[sim]
            base_imp = np.maximum(n_sim, 0.0)[ev_mask]
            for mname, u in U.items():
                lab = inv.get(mname, mname)
                imp, dis, chg, short, adeq = simulate(n_sim, y_sim, np.asarray(u)[sim])
                rows.append(dict(Household=h, EventSet=e["EventSet"], EventID=int(e["event_id"]),
                                 ShiftType=e["shift_type"],
                                 Severity=float(e["severity"]), DurationMin=int(e["duration_min"]),
                                 Condition="shift" if shifted else "placebo", Method=lab,
                                 Day=str(start.tz_convert("Europe/Berlin").date()), Steps=int(ev_mask.sum()),
                                 Adequacy=float(adeq[ev_mask].mean()), Shortfall_kWh=float(short[ev_mask].sum()),
                                 GridImport_kWh=float(imp[ev_mask].sum() * DT_H),
                                 PeakImport_kW=float(imp[ev_mask].max()),
                                 Throughput_kWh=float((dis[ev_mask] + chg[ev_mask]).sum() * DT_H),
                                 NoBatteryImport_kWh=float(base_imp.sum() * DT_H)))
    print(f"{h}: events done ({time.time() - t0:.0f}s)", flush=True)
    del H, K
R = pd.DataFrame(rows)
R.to_csv(OUT, index=False, compression="gzip")
print("rows", len(R), "->", OUT)
print(R[R.Condition == "shift"].groupby("Method")[["Adequacy", "Shortfall_kWh", "GridImport_kWh"]].mean().round(4))
print(R[R.Condition == "placebo"].groupby("Method")[["Adequacy", "Shortfall_kWh", "GridImport_kWh"]].mean().round(4))
