"""
Protocol v3 run (engine3): development split separate from the calibration split.

  python run3.py --stage dev  --backbone tree --tag v3_tree
  python select_params3.py v3_tree
  python run3.py --stage test --backbone tree --tag v3_tree
  python run3.py --stage test --backbone tree --tag v3_tree_clip --clip-load --selection-tag v3_tree

Test sets: frozen (original notebooks, 162 onsets), extended (1620 onsets, same as v2),
trip (complete PV loss, s = 1) and ev (3.7 kW load step), the last two drawn here
with fixed seeds in the test split.
"""
import argparse
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
sys.path.insert(0, HERE)
import her_shift as hs      # noqa: E402
import engine as en         # noqa: E402
import engine2 as e2        # noqa: E402
import engine3 as e3        # noqa: E402

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["dev", "test"], required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--backbone", default="tree", choices=["tree", "mlp"])
ap.add_argument("--tag", required=True)
ap.add_argument("--households", nargs="*", default=["H3", "H4", "H6"])
ap.add_argument("--clip-load", action="store_true")
ap.add_argument("--selection-tag", default=None)
ap.add_argument("--n-dev", type=int, default=10)
args = ap.parse_args()

ZIP = os.environ.get("HEMS_ZIP") or next(p for p in (os.path.join(HERE, "..", "data", "hems_data.zip"),
                            os.path.join(HERE, "..", "Conference", "Conference",
                                         "HEMS_CP_Paper_COMPLETE_Day7_Backup.zip")) if os.path.exists(p))
z = zipfile.ZipFile(ZIP)
ROOT = os.environ.get("FINAL3_ROOT", os.path.join(HERE, "final3"))
OUT = os.path.join(ROOT, args.tag)
EV_TEST = os.path.join(HERE, "final", "events")
EV_DEV = os.path.join(ROOT, "events_dev")
EV_STRESS = os.path.join(ROOT, "events_stress")
for d in (OUT, EV_DEV, EV_STRESS):
    os.makedirs(d, exist_ok=True)
LOG = open(os.path.join(OUT, f"run_{args.stage}.log"), "w", buffering=1)


def _mem_gb():
    """Current and peak working set of this process (Windows), in GB."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return f"[mem {c.WorkingSetSize / 2**30:.1f} GB, peak {c.PeakWorkingSetSize / 2**30:.1f} GB]"
    except Exception:
        return ""


def log(*a):
    msg = " ".join(str(x) for x in a) + " " + _mem_gb()
    print(msg, flush=True)
    LOG.write(msg + "\n")


def rcsv(name):
    return pd.read_csv(io.BytesIO(z.read(name)))


def read_events(path):
    ev = pd.read_csv(path)
    ev["start"] = pd.to_datetime(ev["start"], utc=True)
    ev["end"] = pd.to_datetime(ev["end"], utc=True)
    return ev


def cached_events(path, make):
    if os.path.exists(path):
        return read_events(path)
    ev = make()
    ev.to_csv(path, index=False)
    return read_events(path)


HORIZONS = {"15min": 1, "30min": 2, "60min": 4}
HOMES = {"H3": "residential3", "H4": "residential4", "H6": "residential6"}
DEV_SEEDS = {"H3": 6103, "H4": 6104, "H6": 6106}
TRIP_SEEDS = {"H3": 7103, "H4": 7104, "H6": 7106}
EVL_SEEDS = {"H3": 8103, "H4": 8104, "H6": 8106}
override = None
if args.backbone == "mlp":
    import nn_backbone as nb
    override = nb.fit_mlp_backbone

selected, main = None, None
if args.stage == "test":
    stag = args.selection_tag or args.tag
    with open(os.path.join(ROOT, stag, "selection.json")) as f:
        SEL = json.load(f)["selected"]
    selected = {"WM-CC": ("static", SEL["CC"]), "WM-CCLACP": ("lacp", SEL["CCLACP"])}
    main = ["SCP", "LACP", "ACI|tuned", "CC|tuned", "ROLL|SCP", "ROLL|LACP"] + sorted(set(SEL.values()))
    log("selection used:", SEL)

B = en.Builders(args.device)
meta_rows, clean_rows, ev_rows, pf_frames, info_frames = [], [], [], [], []
t0 = time.time()
for h in args.households:
    df = hs.regularize_15min(rcsv(f"data/processed/{HOMES[h]}_clean.csv"))
    if args.clip_load:
        df["load_kw"] = df["load_kw"].clip(lower=0)
    df["net_load_kw"] = df["load_kw"] - df["pv_kw"]
    tr_end, tu_end, dv_end, ca_end = e3.boundaries3(df)
    if args.stage == "dev":
        ev = cached_events(os.path.join(EV_DEV, f"{h}_dev_events.csv"),
                           lambda: e3.generate_events3(df, tu_end + pd.Timedelta(days=2),
                                                       dv_end - pd.Timedelta(hours=8), args.n_dev, DEV_SEEDS[h]))
        sets = {"dev": ev}
    else:
        fr = rcsv(f"results/day5_events/{h}_events.csv")
        fr["start"] = pd.to_datetime(fr["start"], utc=True)
        fr["end"] = pd.to_datetime(fr["end"], utc=True)
        t_from, t_to = ca_end + pd.Timedelta(days=2), df.index.max()
        sets = {"frozen": fr,
                "extended": read_events(os.path.join(EV_TEST, f"{h}_extended_events.csv")),
                "trip": cached_events(os.path.join(EV_STRESS, f"{h}_trip_events.csv"),
                                      lambda: e3.generate_events3(df, t_from, t_to, 20, TRIP_SEEDS[h],
                                                                  types=("PV",), sevs=(1.0,))),
                "ev": cached_events(os.path.join(EV_STRESS, f"{h}_ev_events.csv"),
                                    lambda: e3.generate_events3(df, t_from, t_to, 20, EVL_SEEDS[h],
                                                                types=("EV",), sevs=(3.7,)))}
    starts = pd.concat([s["start"] for s in sets.values()])
    H = e3.prepare_household3(df, h, B)
    log(f"{h}: signals ready ({time.time() - t0:.0f}s); thresholds {H['thr']}; events: "
        + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))
    tasks = {}
    for hz, steps in HORIZONS.items():
        K = e3.prepare_task3(H, hz, steps, B, args.stage, starts, backbone_override=override)
        tasks[hz] = K
        for m, st in K["clean"].items():
            clean_rows.append(dict(Household=h, Horizon=hz, Method=m, **st))
        meta_rows.append(dict(Household=h, Horizon=hz, Backbone=K["backbone"], Config=str(K["cfg"]),
                              ACI_gamma_tuned=K["gamma_tuned"], CC_gain_tuned=K["tuned_gain"],
                              q_SCP=K["q_scp"], q_LACP=K["q_lacp"], R=K["R"], N=len(K["common"]),
                              N_calib=K["n_calib"]))
        if args.stage == "test":
            train, tune, dev, cal, test = e3.make_splits3(df, steps)
            idx = K["common"]
            pf = pd.DataFrame(dict(y=K["y"], pred=K["pred"]), index=idx)
            rho = H["rho_clean"][e2.SIGNALS["CC"]]
            pf["rho"] = rho.reindex(idx).values
            rc = rho.reindex(cal.index).dropna()
            pf["band_threshold"] = float(np.median(rc[rc > 0]))
            pf["R"] = K["R"]
            pf["SCP_lo"], pf["SCP_hi"] = K["pred"] - K["q_scp"], K["pred"] + K["q_scp"]
            pf["LACP_lo"] = K["pred"] - K["q_lacp"] * K["scale"]
            pf["LACP_hi"] = K["pred"] + K["q_lacp"] * K["scale"]
            for m in (main or []):
                if m in K["sig_cfg"]:
                    q = K["sig_cfg"][m][3]
                    pf[f"{m}_lo"] = K["pred"] - q * K["sig_eval"][m]
                    pf[f"{m}_hi"] = K["pred"] + q * K["sig_eval"][m]
                elif m in K["aci"]:
                    pf[f"{m}_lo"], pf[f"{m}_hi"] = K["aci"][m][2], K["aci"][m][3]
            pf["Household"], pf["Horizon"] = h, hz
            pf_frames.append(pf.reset_index().rename(columns={"timestamp_utc": "t", "index": "t"}))
            parts = []
            for split, d in [("tune", tune), ("dev", dev), ("cal", cal), ("test", test)]:
                X = d[e2.FE]
                f = pd.DataFrame(index=d.index)
                f["split"] = split
                f["abs_err"] = np.abs(d["target"].values - K["model"].predict(X))
                f["lacp_scale"] = np.maximum(K["scale_model"].predict(X), K["floor"])
                f["hour_sin"], f["hour_cos"] = d["target_hour_sin"].values, d["target_hour_cos"].values
                f["rho"] = rho.reindex(d.index).values
                parts.append(f)
            inf = pd.concat(parts)
            inf["Household"], inf["Horizon"], inf["steps"] = h, hz, steps
            info_frames.append(inf.reset_index().rename(columns={"timestamp_utc": "t", "index": "t"}))
        log(f"  {h} {hz}: {K['backbone']}  gamma_tuned={K['gamma_tuned']}  CC gain tuned={K['tuned_gain']}  "
            f"n_calib={K['n_calib']}  ({time.time() - t0:.0f}s)")
    conds = (True,) if args.stage == "dev" else (True, False)
    for sname, ev in sets.items():
        t1 = time.time()
        for _, e in ev.iterrows():
            for shifted in conds:
                for r in e3.evaluate_event3(H, tasks, e, shifted=shifted, selected=selected):
                    r["EventSet"] = sname
                    r["Day"] = str(e["start"].tz_convert("Europe/Berlin").date())
                    ev_rows.append(r)
        log(f"  {h}: {sname} set, {len(ev)} events in {time.time() - t1:.0f}s")
    del H, tasks

sfx = args.stage
pd.DataFrame(meta_rows).to_csv(os.path.join(OUT, f"task_meta_{sfx}.csv"), index=False)
pd.DataFrame(clean_rows).to_csv(os.path.join(OUT, f"clean_metrics_{sfx}.csv"), index=False)
pd.DataFrame(ev_rows).to_csv(os.path.join(OUT, f"event_rows_{sfx}.csv.gz"), index=False, compression="gzip")
if pf_frames:
    pd.concat(pf_frames).to_csv(os.path.join(OUT, "clean_per_forecast.csv.gz"), index=False, compression="gzip")
    pd.concat(info_frames).to_csv(os.path.join(OUT, "info_inputs.csv.gz"), index=False, compression="gzip")
log(f"done in {time.time() - t0:.0f}s; {len(ev_rows)} event rows")
