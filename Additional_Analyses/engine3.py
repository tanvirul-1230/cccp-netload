"""
engine3: protocol after the fourth audit round. Reuses engine2 (and through it
engine.py and her_shift.py) unchanged except for the points below.

1. Five splits by position in the valid record: training 50 %, tune 10 %,
   development 10 %, calibration 10 %, test 20 %. Rows whose target time crosses
   a boundary are dropped (as in her_shift.make_cp_splits). The test period is
   the same as before, so the test events are unchanged.
     stage "dev"  : quantile from the development split, development events
                    injected into the development split, ACI initialised with
                    tune scores.
     stage "test" : quantile from the calibration split only, ACI initialised
                    with calibration scores; gains and steps are fixed before.
   Gains and steps therefore never depend on the calibration data.
2. Added comparators
     ACIL|g        ACI on the LACP score
     ACICCL|b|g    ACI on the CC-LACP score
     ROLL|*        rolling quantile of the latest 2000 scores without level
                   adaptation (ACI with step 0) for SCP, LACP and CC-LACP
     PVonlyL|b, PersistL|b   signal ablation on the LACP scale
     THL|lvl|b     thresholded consistency on the LACP scale:
                   s(x)(1 + beta * 1{rho > tau}), tau a tune-split quantile of rho
3. ACI on an LACP-based score divides the initial scores by s(x) as well.
4. Trip events: complete PV loss (s = 1) in the test split, reported separately.
"""
import numpy as np
import pandas as pd

import her_shift as hs
import engine as en
import engine2 as e2

ALPHA = e2.ALPHA
FE = e2.FE
CQ = e2.CQ
THR_LEVELS = {"q80": 0.80, "q90": 0.90, "q95": 0.95}
BETAS_TH = [0.25, 0.5, 1.0]
BETAS_CCL = [0.25, 0.5, 1.0]

# ---------------------------------------------------------------- step grid
# extended by 0.3 and 0.5 after the first development pass of protocol v3 selected the
# largest step (0.2) for ACI; no test event had been evaluated
GAMMAS_EXT = [0.3, 0.5]
for _g in GAMMAS_EXT:
    e2.ACI_CONFIGS[f"ACI|g{_g:g}"] = (None, _g)
    for _b in e2.ACICC_BETAS:
        e2.ACI_CONFIGS[f"ACICC|b{_b:g}|g{_g:g}"] = (f"CC|b{_b:g}", _g)
e2.GAMMAS = e2.GAMMAS + GAMMAS_EXT
e2.ACICC_GAMMAS = e2.ACICC_GAMMAS + GAMMAS_EXT

# ---------------------------------------------------------------- variants
for _s in ["PVonly", "Persist"]:
    for _b in e2.BETAS_LACP:
        e2.VARIANTS[f"{_s}L|b{_b:g}"] = ([e2.SIGNALS[_s]], (_b,), "lacp")
for _l in THR_LEVELS:
    for _b in BETAS_TH:
        e2.VARIANTS[f"THL|{_l}|b{_b:g}"] = ([("thr", _l)], (_b,), "lacp")
for _g in e2.GAMMAS:
    e2.ACI_CONFIGS[f"ACIL|g{_g:g}"] = ("CCLACP|b0", _g)
for _b in BETAS_CCL:
    for _g in e2.GAMMAS:
        e2.ACI_CONFIGS[f"ACICCL|b{_b:g}|g{_g:g}"] = (f"CCLACP|b{_b:g}", _g)
e2.ACI_CONFIGS["ROLL|SCP"] = (None, 0.0)
e2.ACI_CONFIGS["ROLL|LACP"] = ("CCLACP|b0", 0.0)
for _b in BETAS_CCL:
    e2.ACI_CONFIGS[f"ROLL|CCLACP|b{_b:g}"] = (f"CCLACP|b{_b:g}", 0.0)
VARIANTS, ACI_CONFIGS = e2.VARIANTS, e2.ACI_CONFIGS


# ---------------------------------------------------------------- splits
def boundaries3(df):
    valid = df[df["net_load_kw"].notna()].index
    n = len(valid)
    return valid[int(0.50 * n)], valid[int(0.60 * n)], valid[int(0.70 * n)], valid[int(0.80 * n)]


def make_splits3(df, steps):
    data = hs.make_features(df, steps).dropna(subset=FE + ["target", "target_time"]).copy()
    tr, tu, dv, ca = boundaries3(df)
    train = data[data["target_time"] <= tr].copy()
    tune = data[(data.index > tr) & (data["target_time"] <= tu)].copy()
    dev = data[(data.index > tu) & (data["target_time"] <= dv)].copy()
    cal = data[(data.index > dv) & (data["target_time"] <= ca)].copy()
    test = data[data.index > ca].copy()
    return train, tune, dev, cal, test


# ---------------------------------------------------------------- household
def add_thr(rho, thr):
    base = rho[e2.SIGNALS["CC"]]
    for lvl, v in thr.items():
        rho[("thr", lvl)] = (base > v).astype(float)


def prepare_household3(df, household, B):
    H = e2.prepare_household(df, household, B)
    tr, tu, _, _ = boundaries3(df)
    r = H["rho_clean"][e2.SIGNALS["CC"]]
    r_tune = r[(r.index > tr) & (r.index <= tu)].dropna()
    H["thr"] = {lvl: float(np.quantile(r_tune, q)) for lvl, q in THR_LEVELS.items()}
    add_thr(H["rho_clean"], H["thr"])
    return H


# ---------------------------------------------------------------- task
def prepare_task3(H, hz, steps, B, stage, event_starts, backbone_override=None):
    df = H["df"]
    train, tune, dev, cal, test = make_splits3(df, steps)
    sel = None
    if backbone_override is not None:
        name, model = backbone_override(train, tune)
        cfg = {}
    else:
        name, cfg, model, sel = en.select_backbone(train, tune, B)
    calib = dev if stage == "dev" else cal          # split that sets the quantile
    ev = dev if stage == "dev" else test            # split that is evaluated
    p_tune = model.predict(tune[FE])
    y_tune = tune["target"].values
    e_tune = np.abs(y_tune - p_tune)
    p_cb = model.predict(calib[FE])
    e_cb = np.abs(calib["target"].values - p_cb)
    q_scp = CQ(e_cb, ALPHA)
    sm = en.fit(B.lacp_scale(), tune[FE], e_tune)
    floor = max(1e-3, float(np.quantile(e_tune, 0.10)))
    s_tune = np.maximum(sm.predict(tune[FE]), floor)
    s_cb = np.maximum(sm.predict(calib[FE]), floor)
    q_lacp = CQ(e_cb / s_cb, ALPHA)
    gamma_tuned, _ = hs.tune_delayed_aci_gamma(y_tune, p_tune, hs.ACI_GAMMAS, horizon_steps=steps, alpha=ALPHA)
    ref = H["rho_clean"][e2.SIGNALS["CC"]]
    ti = tune.index.intersection(ref.index)
    pt = pd.Series(p_tune, index=tune.index).loc[ti].values
    tuned_gain = en.tune_gains(tune.loc[ti, "target"].values, pt, [ref.loc[ti].values], 1)
    sig_cfg = {}
    for v, (comps, gains, base) in VARIANTS.items():
        gains = tuned_gain if gains is None else gains
        sc = e2._sigma(H, v, gains, calib.index, s_cb)
        ok = np.isfinite(sc)
        sig_cfg[v] = (comps, tuple(float(g) for g in gains), base, float(CQ(e_cb[ok] / sc[ok], ALPHA)))
    common = pd.DatetimeIndex(ev.index.intersection(ref.index))
    tf = ev.loc[common]
    pred = model.predict(tf[FE])
    scale = np.maximum(sm.predict(tf[FE]), floor)
    y = tf["target"].values
    R = float(ev["target"].max() - ev["target"].min())
    clean = {"SCP": e2.interval_stats(y, pred - q_scp, pred + q_scp, R),
             "LACP": e2.interval_stats(y, pred - q_lacp * scale, pred + q_lacp * scale, R)}
    sig_eval = {}
    for v, (comps, gains, base, q) in sig_cfg.items():
        s = 1 + sum(g * H["rho_clean"][c].reindex(common).values for g, c in zip(gains, comps))
        s = s * scale if base == "lacp" else s
        sig_eval[v] = s
        clean[v] = e2.interval_stats(y, pred - q * s, pred + q * s, R)
    init_idx, init_e, init_s = (tune.index, e_tune, s_tune) if stage == "dev" else (calib.index, e_cb, s_cb)
    snap_pos = sorted({int(common.searchsorted(s)) for s in event_starts})
    snap_pos = [p for p in snap_pos if p < len(common)]
    aci = {}
    for a_name, (sv, g) in ACI_CONFIGS.items():
        g = gamma_tuned if g is None else g
        if sv is None:
            sig_e, init = np.ones(len(common)), init_e
        else:
            comps, gains, base, _ = sig_cfg[sv]
            sig_e = sig_eval[sv]
            si = 1 + sum(gg * H["rho_clean"][c].reindex(init_idx).values for gg, c in zip(gains, comps))
            if base == "lacp":
                si = si * init_s
            ok = np.isfinite(si)
            init = init_e[ok] / si[ok]
        lo, hi, snaps = e2.aci_run(y, pred, sig_e, init, g, steps, snap_pos)
        aci[a_name] = (sv, float(g), lo, hi, snaps)
        clean[a_name] = e2.interval_stats(y, lo, hi, R)
    return dict(hz=hz, steps=steps, stage=stage, backbone=name, cfg=cfg, selection=sel, model=model,
                scale_model=sm, floor=floor, q_scp=float(q_scp), q_lacp=float(q_lacp),
                gamma_tuned=float(gamma_tuned), tuned_gain=float(tuned_gain[0]), sig_cfg=sig_cfg,
                sig_eval=sig_eval, common=common, y=y, pred=pred, scale=scale,
                target_time=pd.DatetimeIndex(tf["target_time"]), aci=aci, R=R, clean=clean,
                n_calib=int(len(calib)))


def generate_events3(df, t_from, t_to, n_per_config, seed, types=("PV", "COMBINED", "LOAD"),
                     sevs=(0.25, 0.5, 0.75), min_pv=0.2):
    """engine2.generate_events_window with configurable types and severities."""
    rng = np.random.default_rng(seed)
    idx = df.index
    bad = np.concatenate([[0], np.cumsum(df["net_load_kw"].isna().values)])
    pv_cs = np.concatenate([[0.0], np.cumsum(df["pv_kw"].fillna(0).values)])
    first = max(int(idx.searchsorted(t_from)), 104)
    rows, eid = [], 0
    for st in types:
        for sev in sevs:
            for dsteps, dmin in [(2, 30), (4, 60), (8, 120)]:
                last = min(int(idx.searchsorted(t_to)), len(idx) - (dsteps + 16))
                cand = np.arange(first, last)
                if len(cand) == 0:
                    continue
                ok = (bad[cand + dsteps + 16] - bad[cand - 104]) == 0
                if st in ("PV", "COMBINED"):
                    ok &= (pv_cs[cand + dsteps] - pv_cs[cand]) / dsteps >= min_pv
                cand = cand[ok]
                if len(cand) == 0:
                    continue
                pick = rng.choice(cand, size=min(n_per_config, len(cand)), replace=False)
                for i in sorted(pick):
                    rows.append(dict(event_id=eid, shift_type=st, severity=sev, duration_steps=dsteps,
                                     duration_min=dmin, start=idx[i], end=idx[i] + (dsteps - 1) * e2.Q15))
                    eid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- EV-sized load steps
def apply_event3(df_win, event):
    """her_shift.apply_event_catalog, plus type "EV": load + severity (kW) over the event rows."""
    if event["shift_type"] != "EV":
        return hs.apply_event_catalog(df_win, pd.DataFrame([event]))
    x = df_win.copy()
    mask = (x.index >= event["start"]) & (x.index <= event["end"])
    x.loc[mask, "load_kw"] = x.loc[mask, "load_kw"] + float(event["severity"])
    x["net_load_kw"] = x["load_kw"] - x["pv_kw"]
    return x


def evaluate_event3(H, tasks, event, shifted=True, selected=None, keep=None):
    df = H["df"]
    start = event["start"]
    stop = hs.event_stop_time(event)
    w0 = start - pd.Timedelta(days=2)
    for K in tasks.values():
        p0 = int(K["common"].searchsorted(start))
        w0 = min(w0, K["common"][max(min(p0, len(K["common"]) - 1) - 12, 0)] - pd.Timedelta(hours=27))
    w1 = stop + pd.Timedelta(hours=3.5) + pd.Timedelta(minutes=60)
    win = None
    if shifted:
        win = apply_event3(df.loc[w0:w1], event)
        rho_w, _ = en.rho_all(win, H["twins"], H["S_clean"])
        add_thr(rho_w, H["thr"])
    else:
        rho_w = H["rho_clean"]
    rows = []
    for K in tasks.values():
        rows += _eval_task3(H, K, event, shifted, win, rho_w, w0, w1, start, stop, selected, keep)
    return rows


# ---------------------------------------------------------------- scoring (engine2._eval_task + additions)
NS_MIN = e2.NS_MIN


def _eval_task3(H, K, event, shifted, win, rho_w, w0, w1, start, stop, selected, keep):
    steps = K["steps"]
    common = K["common"]
    lo_pos = int(common.searchsorted(w0))
    hi_pos = int(common.searchsorted(w1, side="right"))
    widx = common[lo_pos:hi_pos]
    posarr = np.arange(lo_pos, hi_pos)
    if shifted:
        f = hs.make_features(win, steps).dropna(subset=FE + ["target", "target_time"])
        m = widx.isin(f.index)
        idx, p = widx[m], posarr[m]
        f = f.loc[idx]
        pred = K["model"].predict(f[FE])
        scale = np.maximum(K["scale_model"].predict(f[FE]), K["floor"])
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
    orig_in = (ti >= s_ns) & (ti < e_ns)
    inside = orig_in & (tti < e_ns)
    rec = orig_in & (tti == e_ns)
    tgt = (tti >= s_ns) & (tti < e_ns)          # added: every target inside the disturbance
    leg = orig_in & (tti <= e_ns)
    if not leg.any():
        return []
    pre = np.where(ti < s_ns)[0][-8:]
    onset = (ti >= s_ns) & (ti < min(e_ns, s_ns + 30 * NS_MIN))
    legsearch = (ti >= s_ns) & (ti <= e_ns + 120 * NS_MIN)

    M = {"SCP": (pred - K["q_scp"], pred + K["q_scp"]),
         "LACP": (pred - K["q_lacp"] * scale, pred + K["q_lacp"] * scale)}
    rw = {}
    sig_w = {}
    for v, (comps, gains, base, q) in K["sig_cfg"].items():
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
        if shifted:
            sg = sig_w[sv] if sv is not None else np.ones(len(idx))
            lr, hr = e2.aci_resume(snaps[p0], p0, p_end, pos_i, y, pred, sg, lo_c, hi_c, steps, g)
            lo = np.array([lr.get(int(q_), lo_c[q_]) for q_ in p])
            hi = np.array([hr.get(int(q_), hi_c[q_]) for q_ in p])
        else:
            lo, hi = lo_c[p], hi_c[p]
        M[a_name] = (lo, hi)

    # width-matched controls: selected = {tag: ("static" | "lacp", variant name)}
    match = inside if inside.any() else orig_in
    for tag, (kind, v) in (selected or {}).items():
        if v not in M:
            continue
        lo_v, hi_v = M[v]
        target_w = float(np.mean((hi_v - lo_v)[match]))
        if kind == "static":
            M[tag] = (pred - target_w / 2, pred + target_w / 2)
        else:
            k = target_w / float(np.mean(2 * K["q_lacp"] * scale[match]))
            M[tag] = (pred - k * K["q_lacp"] * scale, pred + k * K["q_lacp"] * scale)

    rows = []
    dur = int(event["duration_min"])
    for name, (lo, hi) in M.items():
        if keep is not None and name not in keep:
            continue
        w = hi - lo
        cov = (y >= lo) & (y <= hi)
        sc = w + (2 / ALPHA) * (lo - y) * (y < lo) + (2 / ALPHA) * (y - hi) * (y > hi)
        base = float(np.median(w[pre])) if len(pre) >= 3 else np.nan
        r = dict(Household=H["household"], Horizon=K["hz"], EventID=int(event["event_id"]),
                 ShiftType=event["shift_type"], Severity=float(event["severity"]),
                 DurationMin=dur, Method=name, Condition="shift" if shifted else "placebo",
                 N_in=int(inside.sum()), Cov_in=int(cov[inside].sum()),
                 W_in=float(w[inside].sum()), IS_in=float(sc[inside].sum()),
                 N_rec=int(rec.sum()), Cov_rec=int(cov[rec].sum()),
                 W_rec=float(w[rec].sum()), IS_rec=float(sc[rec].sum()),
                 N_leg=int(leg.sum()), Cov_leg=int(cov[leg].sum()),
                 W_leg=float(w[leg].sum()), IS_leg=float(sc[leg].sum()),
                 N_orig=int(orig_in.sum()), Worig=float(w[orig_in].mean()), PreW=base, R=K["R"])
        exc = np.maximum(y - hi, 0.0)                # added: upper-bound exceedance (kW per target)
        r.update(N_tgt=int(tgt.sum()), Cov_tgt=int(cov[tgt].sum()), W_tgt=float(w[tgt].sum()),
                 IS_tgt=float(sc[tgt].sum()), Exc_tgt=float(exc[tgt].sum()), Exc_in=float(exc[inside].sum()))
        if np.isfinite(base):
            wi = w[orig_in]
            for ratio in (1.10, 1.25, 1.50):
                r[f"Ad{int(round(ratio * 100))}"] = float((wi >= ratio * base).any())
            r["Onset15"] = float((w[onset] >= 1.25 * base).any())
            hit = np.where(orig_in & (w >= 1.25 * base))[0]
            r["DelayStrict"] = float((ti[hit[0]] - s_ns) / NS_MIN) if len(hit) else np.nan
            hit = np.where(legsearch & (w >= 1.25 * base))[0]
            ld = float((ti[hit[0]] - s_ns) / NS_MIN) if len(hit) else np.nan
            r["LegacyAdapted"] = float(np.isfinite(ld) and ld <= dur)
        else:
            for k_ in ("Ad110", "Ad125", "Ad150", "Onset15", "DelayStrict", "LegacyAdapted"):
                r[k_] = np.nan
        rows.append(r)
    return rows
