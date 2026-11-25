"""
Windowed evaluation engine for the consistency-conditioned conformal study.

- Feature construction, splits, event injection, causal eligibility, interval
  metrics and adaptation delay are the original functions (her_shift.py).
- Model builders are pluggable. device="cpu" reproduces the original models
  (saved backbones, saved beta, scikit-learn random forests). device="cuda"
  trains every tree model with XGBoost on the GPU (random forests in XGBoost's
  random-forest mode) and re-selects backbones on the tune split with the
  original grids; trained models then predict on the CPU.
- Each event is evaluated on a window from 2 days before onset to 3.5 h after
  the event. Features look back at most one day, the event baseline looks
  back 24 h and the smoothing half-life is 30 min (initial-condition weight
  after one day: 0.5^48), so window results equal a full-series replay.
- Delayed ACI depends on the whole test history. It is run once on the clean
  series; its state is stored at each event onset and resumed there with the
  shifted data, which reproduces a full replay exactly.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBRFRegressor

import her_shift as hs

ALPHA = hs.ALPHA
RS = hs.RANDOM_STATE
GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
RIDGE_ALPHAS = [0.1, 1.0, 10.0]
RF_CONFIGS = [dict(n_estimators=200, max_depth=12, min_samples_leaf=2),
              dict(n_estimators=300, max_depth=20, min_samples_leaf=2)]
XGB_CONFIGS = [dict(n_estimators=300, max_depth=4, learning_rate=0.05),
               dict(n_estimators=500, max_depth=4, learning_rate=0.03),
               dict(n_estimators=300, max_depth=6, learning_rate=0.05)]
CAL = ["current_hour_sin", "current_hour_cos", "current_dow_sin",
       "current_dow_cos", "current_doy_sin", "current_doy_cos"]
PV_LF = ["pv_kw_past_96"] + CAL
LOAD_LF = ["load_kw_past_96"] + CAL
Q15 = pd.Timedelta(minutes=15)

# expected-behaviour model sets: key -> (PV inputs, load inputs)
TWIN_SETS = {"std": (hs.PV_TWIN_FEATURES, hs.LOAD_TWIN_FEATURES),
             "lfload": (hs.PV_TWIN_FEATURES, LOAD_LF),
             "lf": (PV_LF, LOAD_LF)}
# interval variants: name -> list of signal components (twin set, channel)
VARIANTS = {
    "Proposed": [("std", "max")],
    "PVonly": [("std", "pv")],
    "LoadOnly": [("std", "load")],
    "PerChannel": [("std", "pv"), ("std", "load")],
    "LF-load": [("lfload", "max")],
    "LF-both": [("lf", "max")],
    "LoadOnly-LF": [("lf", "load")],
    "LACP-Cons": [("std", "max")],
}
# fixed-gain sweep (gain not tuned): 1 + b*rho and LACP scale x (1 + b*rho)
SWEEP_GAINS = [0.25, 0.5, 1.0, 2.0]
FIXED = {}
for _b in SWEEP_GAINS:
    VARIANTS[f"Prop-b{_b:g}"] = [("std", "max")]
    VARIANTS[f"LxC-b{_b:g}"] = [("std", "max")]
    FIXED[f"Prop-b{_b:g}"] = (_b,)
    FIXED[f"LxC-b{_b:g}"] = (_b,)
# variants whose scale is the LACP scale times (1 + gain * signal)
ON_LACP = {"LACP-Cons"} | {f"LxC-b{_b:g}" for _b in SWEEP_GAINS}
N_FOLDS = 5


# ----------------------------------------------------------------------
# model builders
# ----------------------------------------------------------------------
class Builders:
    def __init__(self, device="cpu"):
        self.gpu = device == "cuda"

    def _xgb(self, **kw):
        return XGBRegressor(objective="reg:squarederror", subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist",
                            device="cuda", random_state=RS, verbosity=0, **kw)

    def _xgbrf(self, n, depth, leaf):
        return XGBRFRegressor(n_estimators=n, max_depth=depth,
                              min_child_weight=leaf, subsample=0.632,
                              colsample_bynode=1.0, tree_method="hist",
                              device="cuda", random_state=RS, verbosity=0)

    def backbone(self, name, cfg):
        if name == "Ridge":
            return hs.build_ridge(cfg["alpha"])
        if not self.gpu:
            return hs.build_rf(cfg) if name == "RandomForest" else hs.build_xgb(cfg)
        if name == "RandomForest":
            return self._xgbrf(cfg["n_estimators"], cfg["max_depth"],
                               cfg["min_samples_leaf"])
        return self._xgb(**cfg)

    def twin(self):
        if not self.gpu:
            return hs.build_twin_model()
        return self._xgb(n_estimators=300, max_depth=4, learning_rate=0.05)

    def lacp_scale(self):
        if not self.gpu:
            return RandomForestRegressor(n_estimators=200, max_depth=10,
                                         min_samples_leaf=20, n_jobs=-1,
                                         random_state=RS)
        return self._xgbrf(200, 10, 20)


def fit(model, X, y):
    """Fit; GPU-trained XGBoost models are moved to the CPU for prediction."""
    model.fit(X, y)
    if isinstance(model, XGBRegressor) and model.get_params().get("device") == "cuda":
        model.set_params(device="cpu")
    return model


# ----------------------------------------------------------------------
# consistency signals
# ----------------------------------------------------------------------
def twin_inputs(T):
    return list(set(T["pv_f"] + T["load_f"] + ["pv_kw", "load_kw"]))


def z_scores(df, T):
    x = hs.make_twin_features(df).dropna(subset=twin_inputs(T))
    if len(x) == 0:
        return pd.DataFrame(columns=["pv", "load", "max"], index=x.index, dtype=float)
    pe = T["pv_model"].predict(x[T["pv_f"]])
    le = T["load_model"].predict(x[T["load_f"]])
    zp = np.abs(x["pv_kw"].values - pe) / (T["pv_s0"] + T["pv_c"] * np.abs(pe))
    zl = np.abs(x["load_kw"].values - le) / (T["load_s0"] + T["load_c"] * np.abs(le))
    return pd.DataFrame(dict(pv=zp, load=zl, max=np.maximum(zp, zl)), index=x.index)


def smooth(Z, S_clean=None):
    """EWMA (half-life 2 steps) over valid rows. With S_clean, the recursion
    is seeded with the clean smoothed value of the valid row just before the
    first row of Z, which makes a window equal to the full-series result."""
    if S_clean is not None and len(Z):
        k = int(S_clean.index.searchsorted(Z.index[0])) - 1
        if k >= 0:
            Zs = pd.concat([S_clean.iloc[[k]][Z.columns], Z])
            return Zs.ewm(halflife=2, adjust=False).mean().iloc[1:]
    return Z.ewm(halflife=2, adjust=False).mean()


def train_twin(df, pv_f, load_f, B):
    """Original training protocol with chosen inputs. The centring constant of
    each channel is the tune-split median of the smoothed full-series score."""
    train_end, tune_end, _ = hs.get_cp_boundaries(df)
    x = hs.make_twin_features(df)
    x = x.dropna(subset=list(set(pv_f + load_f + ["pv_kw", "load_kw"])))
    tr = x[x.index <= train_end]
    tu = x[(x.index > train_end) & (x.index <= tune_end)]
    pv_m = fit(B.twin(), tr[pv_f], tr["pv_kw"])
    ld_m = fit(B.twin(), tr[load_f], tr["load_kw"])
    pp, lp = pv_m.predict(tu[pv_f]), ld_m.predict(tu[load_f])
    ps0, pc = hs.fit_residual_scale(pp, tu["pv_kw"].values - pp)
    ls0, lc = hs.fit_residual_scale(lp, tu["load_kw"].values - lp)
    T = dict(pv_model=pv_m, load_model=ld_m, pv_f=pv_f, load_f=load_f,
             pv_s0=float(ps0), pv_c=float(pc), load_s0=float(ls0),
             load_c=float(lc))
    S = smooth(z_scores(df, T))
    tm = (S.index > train_end) & (S.index <= tune_end)
    T["normal"] = {c: float(np.median(S.loc[tm, c])) for c in S.columns}
    return T


def orig_as_twin(A):
    """Original consistency assets in the same form (only the max channel is
    centred, with the original rho_normal)."""
    return dict(pv_model=A["pv_model"], load_model=A["load_model"],
                pv_f=hs.PV_TWIN_FEATURES, load_f=hs.LOAD_TWIN_FEATURES,
                pv_s0=A["pv_s0"], pv_c=A["pv_c"], load_s0=A["load_s0"],
                load_c=A["load_c"],
                normal={"max": A["rho_normal"], "pv": np.nan, "load": np.nan})


def rho_all(df, twins, S_clean=None):
    """Centred smoothed signal for every (twin set, channel) component, and
    the uncentred smoothed scores."""
    out, S_all = {}, {}
    for tk, T in twins.items():
        S = smooth(z_scores(df, T), None if S_clean is None else S_clean[tk])
        S_all[tk] = S
        for c in S.columns:
            out[(tk, c)] = np.maximum(0, S[c] - T["normal"][c])
    return out, S_all


# ----------------------------------------------------------------------
# delayed ACI with stored states
# ----------------------------------------------------------------------
def aci_run(y, pred, init_scores, gamma, h, snap_pos=(), max_buffer=2000):
    """Same recursion as the original run_aci_delayed; also stores the state
    (alpha, score buffer) just before each position in snap_pos."""
    n = len(y)
    scores = list(np.asarray(init_scores))
    a = ALPHA
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    snaps = {}
    want = set(int(p) for p in snap_pos)
    for t in range(n):
        if t in want:
            snaps[t] = (a, np.asarray(scores, dtype=float))
        j = t - h
        if j >= 0:
            miss = int((y[j] < lo[j]) or (y[j] > hi[j]))
            a = np.clip(a + gamma * (ALPHA - miss), 0.001, 0.50)
            scores.append(abs(y[j] - pred[j]))
            if len(scores) > max_buffer:
                scores = scores[-max_buffer:]
        q = hs.conformal_quantile(scores, alpha=a)
        lo[t] = pred[t] - q
        hi[t] = pred[t] + q
    return lo, hi, snaps


def aci_resume(snap, p0, p_end, y, pred, lo_clean, hi_clean, h, gamma,
               max_buffer=2000):
    """Continue the recursion from position p0 with y/pred (dicts keyed by
    position, covering p0-h .. p_end) taken from the shifted series."""
    a, scores = snap[0], list(snap[1])
    lo, hi = {}, {}
    for t in range(p0, p_end + 1):
        j = t - h
        if j >= 0:
            lj = lo[j] if j >= p0 else lo_clean[j]
            hj = hi[j] if j >= p0 else hi_clean[j]
            miss = int((y[j] < lj) or (y[j] > hj))
            a = np.clip(a + gamma * (ALPHA - miss), 0.001, 0.50)
            scores.append(abs(y[j] - pred[j]))
            if len(scores) > max_buffer:
                scores = scores[-max_buffer:]
        q = hs.conformal_quantile(scores, alpha=a)
        lo[t] = pred[t] - q
        hi[t] = pred[t] + q
    return lo, hi


# ----------------------------------------------------------------------
# household / task preparation
# ----------------------------------------------------------------------
def select_backbone(train, tune, B):
    """Original selection rule: lowest tune RMSE over the original grids."""
    best, table = None, []
    cands = ([("Ridge", {"alpha": a}) for a in RIDGE_ALPHAS]
             + [("RandomForest", c) for c in RF_CONFIGS]
             + [("XGBoost", c) for c in XGB_CONFIGS])
    for name, cfg in cands:
        m = fit(B.backbone(name, cfg), train[hs.FEATURES], train["target"])
        r = float(np.sqrt(np.mean((tune["target"].values
                                   - m.predict(tune[hs.FEATURES])) ** 2)))
        table.append(dict(Backbone=name, Config=str(cfg), TuneRMSE=r))
        if best is None or r < best[0]:
            best = (r, name, cfg, m)
    return best[1], best[2], best[3], pd.DataFrame(table)


def tune_gains(y, pred, comps, k, base=None):
    """Original beta protocol: first half of the tune split calibrates, the
    second half is scored by interval score, then MPIW. base multiplies the
    scale (out-of-fold LACP scale for the LACP-Cons variant)."""
    s = len(y) // 2
    best = None
    base = np.ones(len(y)) if base is None else base
    for gains in itertools.product(GRID, repeat=k):
        sig = base * (1 + sum(g * c for g, c in zip(gains, comps)))
        q = hs.conformal_quantile(np.abs(y[:s] - pred[:s]) / sig[:s], ALPHA)
        m = hs.interval_metrics(y[s:], pred[s:] - q * sig[s:],
                                pred[s:] + q * sig[s:], ALPHA)
        key = (m["IntervalScore"], m["MPIW"])
        if best is None or key < best[0]:
            best = (key, gains)
    return best[1]


def prepare_household(df, household, B, reproduce=False):
    twins = {k: train_twin(df, pv_f, ld_f, B) for k, (pv_f, ld_f) in TWIN_SETS.items()}
    orig = None
    if reproduce:
        orig = hs.train_consistency_assets(df)
        twins["orig"] = orig_as_twin(orig)
    rho, S = rho_all(df, twins)
    return dict(household=household, df=df, twins=twins, rho_clean=rho,
                S_clean=S, reproduce=reproduce, orig_assets=orig)


def variant_components(H, vname):
    if H["reproduce"] and vname == "Proposed":
        return [("orig", "max")]
    return VARIANTS[vname]


def prepare_task(H, hz, steps, B, event_starts, backbone_override=None):
    df, h = H["df"], H["household"]
    train, tune, cal, test = hs.make_cp_splits(df, steps)
    sel = None
    if backbone_override is not None:
        name, model = backbone_override(train, tune)
        cfg = {}
    elif H["reproduce"]:
        name, cfg = hs.get_saved_backbone(h, hz)
        model = hs.fit_backbone(name, cfg, train)
    else:
        name, cfg, model, sel = select_backbone(train, tune, B)
    p_tune = model.predict(tune[hs.FEATURES])
    y_tune = tune["target"].values
    tune_err = np.abs(y_tune - p_tune)
    p_cal = pd.Series(model.predict(cal[hs.FEATURES]), index=cal.index)
    y_cal = cal["target"]
    cal_scores = np.abs(y_cal.values - p_cal.values)
    q_scp = hs.conformal_quantile(cal_scores, ALPHA)

    sm = fit(B.lacp_scale(), tune[hs.FEATURES], tune_err)
    floor = max(1e-3, float(np.quantile(tune_err, 0.10)))
    s_cal = np.maximum(sm.predict(cal[hs.FEATURES]), floor)
    q_lacp = hs.conformal_quantile(cal_scores / s_cal, ALPHA)
    # out-of-fold LACP scale on the tune split (contiguous folds), used only
    # to tune the gain of variants built on the LACP scale
    s_oof = np.empty(len(tune))
    for f in np.array_split(np.arange(len(tune)), N_FOLDS):
        rest = np.setdiff1d(np.arange(len(tune)), f)
        mf = fit(B.lacp_scale(), tune[hs.FEATURES].iloc[rest], tune_err[rest])
        s_oof[f] = mf.predict(tune[hs.FEATURES].iloc[f])
    s_oof = pd.Series(np.maximum(s_oof, floor), index=tune.index)
    s_cal = pd.Series(s_cal, index=cal.index)

    gamma, _ = hs.tune_delayed_aci_gamma(y_tune, p_tune, hs.ACI_GAMMAS,
                                         horizon_steps=steps, alpha=ALPHA)

    rho_ref = H["rho_clean"][("orig", "max") if H["reproduce"] else ("std", "max")]
    p_tune = pd.Series(p_tune, index=tune.index)
    sig_cfg = {}
    for vname in VARIANTS:
        comps = variant_components(H, vname)
        vidx = H["rho_clean"][comps[0]].index
        ti = tune.index.intersection(vidx)
        ci = cal.index.intersection(vidx)
        pt = p_tune.loc[ti].values
        yt = tune.loc[ti, "target"].values
        lacp = vname in ON_LACP
        if H["reproduce"] and vname == "Proposed":
            gains = (hs.get_saved_beta(h, hz),)
        elif vname in FIXED:
            gains = FIXED[vname]
        else:
            gains = tune_gains(yt, pt, [H["rho_clean"][c].loc[ti].values
                                        for c in comps], len(comps),
                               base=s_oof.loc[ti].values if lacp else None)
        sig = 1 + sum(g * H["rho_clean"][c].loc[ci].values
                      for g, c in zip(gains, comps))
        if lacp:
            sig = sig * s_cal.loc[ci].values
        q = hs.conformal_quantile(
            np.abs(y_cal.loc[ci].values - p_cal.loc[ci].values) / sig, ALPHA)
        sig_cfg[vname] = (comps, tuple(float(g) for g in gains), float(q))

    # clean test on the index used by the original evaluation
    common = test.index.intersection(rho_ref.index)
    tf = test.loc[common]
    pred = model.predict(tf[hs.FEATURES])
    scale = np.maximum(sm.predict(tf[hs.FEATURES]), floor)
    y = tf["target"].values
    common = pd.DatetimeIndex(common)
    snap_pos = sorted({int(common.searchsorted(s)) for s in event_starts})
    snap_pos = [p for p in snap_pos if p < len(common)]
    lo, hi, snaps = aci_run(y, pred, cal_scores, gamma, steps, snap_pos)
    R = float(test["target"].max() - test["target"].min())
    return dict(hz=hz, steps=steps, backbone=name, cfg=cfg, selection=sel,
                model=model, scale_model=sm, floor=floor, cal_scores=cal_scores,
                q_scp=float(q_scp),
                q_lacp=float(q_lacp), gamma=float(gamma), sig_cfg=sig_cfg,
                common=common, y=y, pred=pred, scale=scale,
                target_time=pd.DatetimeIndex(tf["target_time"]),
                aci_lo=lo, aci_hi=hi, snaps=snaps, R=R)


def clean_test_frame(H, K):
    """Per-forecast clean-test intervals of every method (for clean metrics)."""
    idx = K["common"]
    out = pd.DataFrame(dict(y=K["y"], pred=K["pred"]), index=idx)
    out["SCP_lo"], out["SCP_hi"] = K["pred"] - K["q_scp"], K["pred"] + K["q_scp"]
    out["LACP_lo"] = K["pred"] - K["q_lacp"] * K["scale"]
    out["LACP_hi"] = K["pred"] + K["q_lacp"] * K["scale"]
    out["ACI_lo"], out["ACI_hi"] = K["aci_lo"], K["aci_hi"]
    for vname, (comps, gains, q) in K["sig_cfg"].items():
        sig = 1 + sum(g * H["rho_clean"][c].reindex(idx).values
                      for g, c in zip(gains, comps))
        if vname in ON_LACP:
            sig = sig * K["scale"]
        out[f"{vname}_lo"], out[f"{vname}_hi"] = K["pred"] - q * sig, K["pred"] + q * sig
    ref = ("orig", "max") if H["reproduce"] else ("std", "max")
    out["rho"] = H["rho_clean"][ref].reindex(idx).values
    return out


# ----------------------------------------------------------------------
# event evaluation
# ----------------------------------------------------------------------
def evaluate_event(H, tasks, event, shifted=True, trace=None):
    """Rows for one event over all tasks (horizons), shifted or placebo.
    With a list as trace, the interval series of every method are appended."""
    df = H["df"]
    start = event["start"]
    stop = hs.event_stop_time(event)
    # window: covers 12 valid forecasts before onset plus one day of inputs
    w0 = start - pd.Timedelta(days=2)
    for K in tasks.values():
        p0 = int(K["common"].searchsorted(start))
        w0 = min(w0, K["common"][max(p0 - 12, 0)] - pd.Timedelta(hours=27))
    w1 = stop + pd.Timedelta(hours=3.5) + pd.Timedelta(minutes=15 * 4)
    win = None
    if shifted:
        win = hs.apply_event_catalog(df.loc[w0:w1], pd.DataFrame([event]))
        rho_w, _ = rho_all(win, H["twins"], H["S_clean"])
    else:
        rho_w = H["rho_clean"]
    rows = []
    for K in tasks.values():
        rows += _evaluate_task(H, K, event, shifted, win, rho_w, w0, w1, start, trace)
    return rows


def _evaluate_task(H, K, event, shifted, win, rho_w, w0, w1, start, trace=None):
    steps = K["steps"]
    common = K["common"]
    lo_pos = int(common.searchsorted(w0))
    hi_pos = int(common.searchsorted(w1, side="right"))
    widx = common[lo_pos:hi_pos]
    pos = pd.Series(np.arange(lo_pos, hi_pos), index=widx)

    if shifted:
        f = hs.make_features(win, steps)
        f = f.dropna(subset=hs.FEATURES + ["target", "target_time"])
        idx = widx.intersection(f.index)
        f = f.loc[idx]
        pred = pd.Series(K["model"].predict(f[hs.FEATURES]), index=idx)
        scale = pd.Series(np.maximum(K["scale_model"].predict(f[hs.FEATURES]),
                                     K["floor"]), index=idx)
        y = f["target"]
        tt = pd.DatetimeIndex(f["target_time"])
    else:
        idx = widx
        sl = slice(lo_pos, hi_pos)
        pred = pd.Series(K["pred"][sl], index=idx)
        scale = pd.Series(K["scale"][sl], index=idx)
        y = pd.Series(K["y"][sl], index=idx)
        tt = K["target_time"][sl]

    methods = {
        "SCP": (pred - K["q_scp"], pred + K["q_scp"]),
        "LACP": (pred - K["q_lacp"] * scale, pred + K["q_lacp"] * scale),
    }
    for vname, (comps, gains, q) in K["sig_cfg"].items():
        sig = 1 + sum(g * rho_w[c].reindex(idx).values for g, c in zip(gains, comps))
        if vname in ON_LACP:
            sig = sig * scale.values
        half = pd.Series(q * sig, index=idx)
        methods[vname] = (pred - half, pred + half)

    p = pos.loc[idx].values
    p0 = int(common.searchsorted(start))
    if shifted:
        if p0 not in K["snaps"]:
            return []
        yd = dict(zip(p, y.values))
        pdict = dict(zip(p, pred.values))
        p_end = int(p.max())
        if any(q_ not in yd for q_ in range(max(p0 - steps, 0), p_end + 1)):
            return []
        lo_r, hi_r = aci_resume(K["snaps"][p0], p0, p_end, yd, pdict,
                                K["aci_lo"], K["aci_hi"], steps, K["gamma"])
        lo_a = np.array([lo_r.get(q_, K["aci_lo"][q_]) for q_ in p])
        hi_a = np.array([hi_r.get(q_, K["aci_hi"][q_]) for q_ in p])
    else:
        lo_a, hi_a = K["aci_lo"][p], K["aci_hi"][p]
    methods["ACI"] = (pd.Series(lo_a, index=idx), pd.Series(hi_a, index=idx))

    frame = pd.DataFrame({"target_time": tt}, index=idx)
    eidx = idx[hs.causal_event_mask(frame, event)]
    if len(eidx) == 0:
        return []
    plo, phi = methods["Proposed"]
    wm_half = float((phi.loc[eidx] - plo.loc[eidx]).mean()) / 2
    methods["WidthMatched"] = (pred - wm_half, pred + wm_half)
    if trace is not None:
        rho_p = rho_w[VARIANTS["Proposed"][0] if not H["reproduce"] else ("orig", "max")]
        trace.append(dict(Horizon=K["hz"], shifted=shifted, y=y, pred=pred,
                          target_time=pd.Series(tt, index=idx), methods=methods,
                          eidx=eidx, rho=rho_p.reindex(idx)))

    rows = []
    for mname, (lo, hi) in methods.items():
        m = hs.interval_metrics_detailed(y.loc[eidx], lo.loc[eidx], hi.loc[eidx], ALPHA)
        d = hs.adaptation_delay_corrected(idx, lo, hi, event)
        w = (hi - lo)
        pre = w[w.index < start].tail(8)
        rows.append(dict(
            Household=H["household"], Horizon=K["hz"],
            EventID=int(event["event_id"]), ShiftType=event["shift_type"],
            Severity=float(event["severity"]),
            DurationMin=int(event["duration_min"]), Method=mname,
            Condition="shift" if shifted else "placebo",
            EventPICP=m["PICP"], EventMPIW=m["MPIW"],
            EventIntervalScore=m["IntervalScore"], CoveredN=m["CoveredN"],
            EligibleN=m["EligibleN"], WidthSum=m["WidthSum"],
            IntervalScoreSum=m["IntervalScoreSum"], AdaptDelayMin=d,
            AdaptedWithinEvent=(False if np.isnan(d)
                                else d <= int(event["duration_min"])),
            PreWidth=float(pre.median()) if len(pre) else np.nan,
            R=K["R"]))
    return rows


# ----------------------------------------------------------------------
# extended event set (rules inferred from the frozen events)
# ----------------------------------------------------------------------
def generate_events(df, n_per_config, seed, min_pv=0.2):
    """Events in the test split. Per configuration (type x severity x
    duration) n_per_config onsets are drawn at random from all admissible
    15-min onsets: complete data from 26 h before onset to 4 h after the
    event, onset at least 2 days into the test split, and for PV and
    combined events a mean clean PV of at least min_pv kW over the event."""
    rng = np.random.default_rng(seed)
    _, _, cal_end = hs.get_cp_boundaries(df)
    idx = df.index
    bad = np.concatenate([[0], np.cumsum(df["net_load_kw"].isna().values)])
    pv = df["pv_kw"].fillna(0).values
    pv_cs = np.concatenate([[0.0], np.cumsum(pv)])
    first = int(idx.searchsorted(cal_end + pd.Timedelta(days=2)))
    rows, eid = [], 0
    for st in ["PV", "COMBINED", "LOAD"]:
        for sev in [0.25, 0.5, 0.75]:
            for dsteps, dmin in [(2, 30), (4, 60), (8, 120)]:
                last = len(idx) - (dsteps + 16)
                cand = np.arange(first, last)
                lo = cand - 104
                hi = cand + dsteps - 1 + 16
                ok = (bad[hi + 1] - bad[lo]) == 0
                if st in ("PV", "COMBINED"):
                    mean_pv = (pv_cs[cand + dsteps] - pv_cs[cand]) / dsteps
                    ok &= mean_pv >= min_pv
                cand = cand[ok]
                pick = rng.choice(cand, size=min(n_per_config, len(cand)),
                                  replace=False)
                for i in sorted(pick):
                    t = idx[i]
                    rows.append(dict(event_id=eid, shift_type=st, severity=sev,
                                     duration_steps=dsteps, duration_min=dmin,
                                     start=t, end=t + (dsteps - 1) * Q15))
                    eid += 1
    return pd.DataFrame(rows)
