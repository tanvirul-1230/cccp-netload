"""
engine2: corrected evaluation, development-split selection and extra comparators.

Changes relative to engine.py (model training, features, splits, consistency
signal, event injection and the windowed replay are reused unchanged):

1. Conformal quantile: the k-th smallest calibration score,
   k = ceil((n+1)(1-alpha)) (Lei et al. 2018, Alg. 2). The environment variable
   QUANTILE=legacy keeps the original function, which returns the (k+1)-th.
2. Event scoring. Injection modifies rows t_s ... t_s+d-15 min, so the row at
   t_s+d is already clean.
     inside   : issue time in [t_s, t_s+d) and target time < t_s+d
     recovery : issue time in [t_s, t_s+d) and target time = t_s+d
     legacy   : issue time in [t_s, t_s+d) and target time <= t_s+d (original rule)
3. Adaptation counts a width crossing only at issue times inside the event,
   [t_s, t_s+d), at 1.10 / 1.25 / 1.50 x the median of the 8 pre-event widths;
   "within 15 min" = issue times t_s and t_s+15 min. The original rule (search
   to t_s+d+2 h, delay <= d) is kept as a legacy flag.
4. Continuous response: mean interval width over issue times inside the event.
5. Variants: every consistency signal at gains {0, .25, .5, 1, 2, 4} on a
   constant base; the main signal on the LACP scale at {0, .25, .5, 1, 2};
   ACI at steps {.005, .01, .02, .05, .1}; ACI on the consistency-scaled score
   (ACI-CC) over beta x gamma; a persistence-residual signal as a simpler
   comparator; the original clean-data tuned gain and ACI step as references.
6. Width-matched controls for the selected constant-base and LACP-base methods.
"""
import os

import numpy as np
import pandas as pd

import her_shift as hs
import engine as en

ALPHA = hs.ALPHA
Q15 = pd.Timedelta(minutes=15)
FE = hs.FEATURES
LEGACY_Q = os.environ.get("QUANTILE", "exact") == "legacy"


def conformal_quantile_exact(scores, alpha=0.10):
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n == 0:
        raise ValueError("Empty conformal scores.")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(s, k - 1)[k - 1])


if not LEGACY_Q:
    hs.conformal_quantile = conformal_quantile_exact


def CQ(scores, alpha):
    return hs.conformal_quantile(scores, alpha=alpha)


# ----------------------------------------------------------------------
# signals and variants
# ----------------------------------------------------------------------
class Persistence:
    """Expected value = last observed value (no learned model)."""

    def __init__(self, col):
        self.col = col

    def predict(self, X):
        return np.asarray(X[self.col], dtype=float)


TWIN_SETS = dict(en.TWIN_SETS)
TWIN_SETS["persist"] = (["pv_kw_past_1"], ["load_kw_past_1"])
SIGNALS = {"CC": ("std", "max"), "PVonly": ("std", "pv"), "LoadOnly": ("std", "load"),
           "LF-load": ("lfload", "max"), "LF-both": ("lf", "max"),
           "LoadOnly-LF": ("lf", "load"), "Persist": ("persist", "max")}
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
BETAS_LACP = [0.0, 0.25, 0.5, 1.0, 2.0]
# step grids extended to 0.2 after the first development pass selected the
# largest step (0.1 for ACI, 0.05 for ACI-CC); no test event had been evaluated
GAMMAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
ACICC_BETAS = [0.25, 0.5, 1.0, 2.0]
ACICC_GAMMAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

VARIANTS = {}
for _s, _c in SIGNALS.items():
    for _b in BETAS:
        VARIANTS[f"{_s}|b{_b:g}"] = ([_c], (_b,), "const")
for _b in BETAS_LACP:
    VARIANTS[f"CCLACP|b{_b:g}"] = ([SIGNALS["CC"]], (_b,), "lacp")
VARIANTS["CC|tuned"] = ([SIGNALS["CC"]], None, "const")   # original clean-data protocol

ACI_CONFIGS = {f"ACI|g{_g:g}": (None, _g) for _g in GAMMAS}
ACI_CONFIGS["ACI|tuned"] = (None, None)                     # original clean-data protocol
for _b in ACICC_BETAS:
    for _g in ACICC_GAMMAS:
        ACI_CONFIGS[f"ACICC|b{_b:g}|g{_g:g}"] = (f"CC|b{_b:g}", _g)


def train_twin(df, key, B):
    pv_f, load_f = TWIN_SETS[key]
    if key != "persist":
        return en.train_twin(df, pv_f, load_f, B)
    train_end, tune_end, _ = hs.get_cp_boundaries(df)
    x = hs.make_twin_features(df).dropna(subset=list(set(pv_f + load_f + ["pv_kw", "load_kw"])))
    tu = x[(x.index > train_end) & (x.index <= tune_end)]
    pv_m, ld_m = Persistence(pv_f[0]), Persistence(load_f[0])
    pp, lp = pv_m.predict(tu), ld_m.predict(tu)
    ps0, pc = hs.fit_residual_scale(pp, tu["pv_kw"].values - pp)
    ls0, lc = hs.fit_residual_scale(lp, tu["load_kw"].values - lp)
    T = dict(pv_model=pv_m, load_model=ld_m, pv_f=pv_f, load_f=load_f, pv_s0=float(ps0),
             pv_c=float(pc), load_s0=float(ls0), load_c=float(lc))
    S = en.smooth(en.z_scores(df, T))
    tm = (S.index > train_end) & (S.index <= tune_end)
    T["normal"] = {c: float(np.median(S.loc[tm, c])) for c in S.columns}
    return T


def prepare_household(df, household, B):
    twins = {k: train_twin(df, k, B) for k in TWIN_SETS}
    rho, S = en.rho_all(df, twins)
    return dict(household=household, df=df, twins=twins, rho_clean=rho, S_clean=S)


# ----------------------------------------------------------------------
# ACI on a (possibly scaled) score
# ----------------------------------------------------------------------
def aci_run(y, pred, sig, init, gamma, h, snap_pos=(), max_buffer=2000):
    """Delayed ACI (original recursion and clipping) on scores |y - pred| / sig;
    intervals pred +/- q * sig. Stores the state before each snap position."""
    n = len(y)
    scores = list(np.asarray(init, dtype=float))
    a = ALPHA
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    snaps, want = {}, set(int(p) for p in snap_pos)
    for t in range(n):
        if t in want:
            snaps[t] = (a, np.asarray(scores, dtype=float))
        j = t - h
        if j >= 0:
            miss = int((y[j] < lo[j]) or (y[j] > hi[j]))
            a = float(np.clip(a + gamma * (ALPHA - miss), 0.001, 0.50))
            scores.append(abs(y[j] - pred[j]) / sig[j])
            if len(scores) > max_buffer:
                scores = scores[-max_buffer:]
        q = CQ(scores, a)
        lo[t] = pred[t] - q * sig[t]
        hi[t] = pred[t] + q * sig[t]
    return lo, hi, snaps


def aci_resume(snap, p0, p_end, pos_i, y, pred, sig, lo_c, hi_c, h, gamma, max_buffer=2000):
    a, scores = snap[0], list(snap[1])
    lo, hi = {}, {}
    for t in range(p0, p_end + 1):
        j = t - h
        if j >= 0:
            lj = lo[j] if j >= p0 else lo_c[j]
            hj = hi[j] if j >= p0 else hi_c[j]
            ij = pos_i[j]
            miss = int((y[ij] < lj) or (y[ij] > hj))
            a = float(np.clip(a + gamma * (ALPHA - miss), 0.001, 0.50))
            scores.append(abs(y[ij] - pred[ij]) / sig[ij])
            if len(scores) > max_buffer:
                scores = scores[-max_buffer:]
        q = CQ(scores, a)
        it = pos_i[t]
        lo[t] = pred[it] - q * sig[it]
        hi[t] = pred[it] + q * sig[it]
    return lo, hi


# ----------------------------------------------------------------------
# task preparation
# ----------------------------------------------------------------------
def _sigma(H, vname, gains, idx, scale):
    comps, _, base = VARIANTS[vname]
    s = 1 + sum(g * H["rho_clean"][c].reindex(idx).values for g, c in zip(gains, comps))
    return s * scale if base == "lacp" else s


def interval_stats(y, lo, hi, R):
    w = hi - lo
    sc = w + (2 / ALPHA) * (lo - y) * (y < lo) + (2 / ALPHA) * (y - hi) * (y > hi)
    return dict(PICP=float(((y >= lo) & (y <= hi)).mean()), MPIW=float(w.mean()),
                nIS=float(sc.mean() / R), N=int(len(y)))


def prepare_task(H, hz, steps, B, stage, event_starts, backbone_override=None):
    """stage 'test': evaluation period = test split, ACI initialised with
    calibration scores. stage 'dev': evaluation period = calibration split
    (development events), ACI initialised with tune-split scores."""
    df = H["df"]
    train, tune, cal, test = hs.make_cp_splits(df, steps)
    sel = None
    if backbone_override is not None:
        name, model = backbone_override(train, tune)
        cfg = {}
    else:
        name, cfg, model, sel = en.select_backbone(train, tune, B)
    p_tune = model.predict(tune[FE])
    y_tune = tune["target"].values
    e_tune = np.abs(y_tune - p_tune)
    p_cal = model.predict(cal[FE])
    y_cal = cal["target"].values
    e_cal = np.abs(y_cal - p_cal)
    q_scp = CQ(e_cal, ALPHA)
    sm = en.fit(B.lacp_scale(), tune[FE], e_tune)
    floor = max(1e-3, float(np.quantile(e_tune, 0.10)))
    s_tune = np.maximum(sm.predict(tune[FE]), floor)
    s_cal = np.maximum(sm.predict(cal[FE]), floor)
    q_lacp = CQ(e_cal / s_cal, ALPHA)
    gamma_tuned, _ = hs.tune_delayed_aci_gamma(y_tune, p_tune, hs.ACI_GAMMAS,
                                               horizon_steps=steps, alpha=ALPHA)

    ref = H["rho_clean"][SIGNALS["CC"]]
    ti = tune.index.intersection(ref.index)
    pt = pd.Series(p_tune, index=tune.index).loc[ti].values
    tuned_gain = en.tune_gains(tune.loc[ti, "target"].values, pt, [ref.loc[ti].values], 1)
    sig_cfg = {}
    for v, (comps, gains, base) in VARIANTS.items():
        gains = tuned_gain if gains is None else gains
        sc = _sigma(H, v, gains, cal.index, s_cal)
        ok = np.isfinite(sc)
        sig_cfg[v] = (comps, tuple(float(g) for g in gains), base,
                      float(CQ(e_cal[ok] / sc[ok], ALPHA)))

    ev = test if stage == "test" else cal
    common = pd.DatetimeIndex(ev.index.intersection(ref.index))
    tf = ev.loc[common]
    pred = model.predict(tf[FE])
    scale = np.maximum(sm.predict(tf[FE]), floor)
    y = tf["target"].values
    R = float(ev["target"].max() - ev["target"].min())
    clean = {"SCP": interval_stats(y, pred - q_scp, pred + q_scp, R),
             "LACP": interval_stats(y, pred - q_lacp * scale, pred + q_lacp * scale, R)}
    sig_eval = {}
    for v, (comps, gains, base, q) in sig_cfg.items():
        s = 1 + sum(g * H["rho_clean"][c].reindex(common).values for g, c in zip(gains, comps))
        s = s * scale if base == "lacp" else s
        sig_eval[v] = s
        clean[v] = interval_stats(y, pred - q * s, pred + q * s, R)

    init_idx, init_e, init_s = (tune.index, e_tune, s_tune) if stage == "dev" else \
                               (cal.index, e_cal, s_cal)
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
            si = 1 + sum(gg * H["rho_clean"][c].reindex(init_idx).values
                         for gg, c in zip(gains, comps))
            ok = np.isfinite(si)
            init = init_e[ok] / si[ok]
        lo, hi, snaps = aci_run(y, pred, sig_e, init, g, steps, snap_pos)
        aci[a_name] = (sv, float(g), lo, hi, snaps)
        clean[a_name] = interval_stats(y, lo, hi, R)
    return dict(hz=hz, steps=steps, stage=stage, backbone=name, cfg=cfg, selection=sel,
                model=model, scale_model=sm, floor=floor, q_scp=float(q_scp),
                q_lacp=float(q_lacp), gamma_tuned=float(gamma_tuned),
                tuned_gain=float(tuned_gain[0]), sig_cfg=sig_cfg, sig_eval=sig_eval,
                common=common, y=y, pred=pred, scale=scale,
                target_time=pd.DatetimeIndex(tf["target_time"]), aci=aci, R=R, clean=clean)


# ----------------------------------------------------------------------
# event evaluation
# ----------------------------------------------------------------------
NS_MIN = 60 * 10 ** 9


def evaluate_event(H, tasks, event, shifted=True, selected=None, keep=None):
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
        win = hs.apply_event_catalog(df.loc[w0:w1], pd.DataFrame([event]))
        rho_w, _ = en.rho_all(win, H["twins"], H["S_clean"])
    else:
        rho_w = H["rho_clean"]
    rows = []
    for K in tasks.values():
        rows += _eval_task(H, K, event, shifted, win, rho_w, w0, w1, start, stop, selected, keep)
    return rows


def _eval_task(H, K, event, shifted, win, rho_w, w0, w1, start, stop, selected, keep):
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
            lr, hr = aci_resume(snaps[p0], p0, p_end, pos_i, y, pred, sg, lo_c, hi_c, steps, g)
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


# ----------------------------------------------------------------------
# event generation within a time window
# ----------------------------------------------------------------------
def generate_events_window(df, t_from, t_to, n_per_config, seed, min_pv=0.2):
    """Same rules as engine.generate_events, with onsets restricted to
    [t_from, t_to]."""
    rng = np.random.default_rng(seed)
    idx = df.index
    bad = np.concatenate([[0], np.cumsum(df["net_load_kw"].isna().values)])
    pv_cs = np.concatenate([[0.0], np.cumsum(df["pv_kw"].fillna(0).values)])
    first = max(int(idx.searchsorted(t_from)), 104)
    rows, eid = [], 0
    for st in ["PV", "COMBINED", "LOAD"]:
        for sev in [0.25, 0.5, 0.75]:
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
                    rows.append(dict(event_id=eid, shift_type=st, severity=sev,
                                     duration_steps=dsteps, duration_min=dmin,
                                     start=idx[i], end=idx[i] + (dsteps - 1) * Q15))
                    eid += 1
    return pd.DataFrame(rows)
