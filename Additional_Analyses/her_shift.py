# Shift-study functions, copied verbatim from the causal shift notebook
# (master base functions, corrected functions, task assets, event evaluation).
# Only the Colab upload/restore cells are omitted; the globals below replace them.
import ast
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
ALPHA = 0.10
ACI_GAMMAS = [0.005, 0.01, 0.02, 0.05]
WIDTH_THRESHOLD_RATIO = 1.25

# set by the caller before use
cp_metadata = None
day4_results = None

# ============================================================
# DAY 5R MASTER BASE FUNCTIONS
# ============================================================


# ============================================================
# 1. REGULARIZE EXACT 15-MINUTE TIMELINE
# ============================================================

def regularize_15min(df):

    x = df.copy()

    x["timestamp_utc"] = pd.to_datetime(
        x["timestamp"],
        utc=True
    )

    x = (
        x.sort_values("timestamp_utc")
         .drop_duplicates("timestamp_utc")
         .set_index("timestamp_utc")
    )

    full_index = pd.date_range(
        start=x.index.min(),
        end=x.index.max(),
        freq="15min",
        tz="UTC"
    )

    x = x.reindex(
        full_index
    )

    x.index.name = "timestamp_utc"

    x["timestamp_local"] = (
        x.index.tz_convert(
            "Europe/Berlin"
        )
    )

    return x


# ============================================================
# 2. FORECAST FEATURES
# ============================================================

def make_features(
    df,
    horizon_steps
):

    x = df.copy()

    base_cols = [
        "load_kw",
        "pv_kw",
        "net_load_kw"
    ]

    for col in base_cols:

        for lag in [
            1,
            2,
            4,
            96
        ]:

            x[
                f"{col}_lag_{lag}"
            ] = (
                x[col]
                .shift(lag)
            )

        x[
            f"{col}_roll_4"
        ] = (
            x[col]
            .rolling(
                4,
                min_periods=4
            )
            .mean()
        )

        x[
            f"{col}_roll_8"
        ] = (
            x[col]
            .rolling(
                8,
                min_periods=8
            )
            .mean()
        )

    x["target"] = (
        x["net_load_kw"]
        .shift(
            -horizon_steps
        )
    )

    x["target_time"] = (
        x.index
        +
        pd.Timedelta(
            minutes=
            15*horizon_steps
        )
    )

    local = (
        x["target_time"]
        .dt.tz_convert(
            "Europe/Berlin"
        )
    )

    hour = (
        local.dt.hour
        +
        local.dt.minute/60
    )

    x["target_hour_sin"] = np.sin(
        2*np.pi*hour/24
    )

    x["target_hour_cos"] = np.cos(
        2*np.pi*hour/24
    )

    dow = (
        local.dt.dayofweek
    )

    x["target_dow_sin"] = np.sin(
        2*np.pi*dow/7
    )

    x["target_dow_cos"] = np.cos(
        2*np.pi*dow/7
    )

    doy = (
        local.dt.dayofyear
    )

    x["target_doy_sin"] = np.sin(
        2*np.pi*doy/365.25
    )

    x["target_doy_cos"] = np.cos(
        2*np.pi*doy/365.25
    )

    return x


FEATURES = [

    "load_kw",
    "pv_kw",
    "net_load_kw",

    "load_kw_lag_1",
    "load_kw_lag_2",
    "load_kw_lag_4",
    "load_kw_lag_96",
    "load_kw_roll_4",
    "load_kw_roll_8",

    "pv_kw_lag_1",
    "pv_kw_lag_2",
    "pv_kw_lag_4",
    "pv_kw_lag_96",
    "pv_kw_roll_4",
    "pv_kw_roll_8",

    "net_load_kw_lag_1",
    "net_load_kw_lag_2",
    "net_load_kw_lag_4",
    "net_load_kw_lag_96",
    "net_load_kw_roll_4",
    "net_load_kw_roll_8",

    "target_hour_sin",
    "target_hour_cos",

    "target_dow_sin",
    "target_dow_cos",

    "target_doy_sin",
    "target_doy_cos"
]


# ============================================================
# 3. 50 / 10 / 20 / 20 SPLITS
# ============================================================

def get_cp_boundaries(df):

    valid_times = df[
        df["net_load_kw"].notna()
    ].index

    n = len(
        valid_times
    )

    train_end = valid_times[
        int(0.50*n)
    ]

    tune_end = valid_times[
        int(0.60*n)
    ]

    calibration_end = valid_times[
        int(0.80*n)
    ]

    return (
        train_end,
        tune_end,
        calibration_end
    )


def make_cp_splits(
    df,
    horizon_steps
):

    data = make_features(
        df,
        horizon_steps
    )

    (
        train_end,
        tune_end,
        calibration_end
    ) = get_cp_boundaries(
        df
    )

    required = (
        FEATURES
        +
        [
            "target",
            "target_time"
        ]
    )

    data = (
        data.dropna(
            subset=required
        )
        .copy()
    )

    train = data[
        data["target_time"]
        <=
        train_end
    ].copy()

    tune = data[
        (
            data.index
            >
            train_end
        )
        &
        (
            data["target_time"]
            <=
            tune_end
        )
    ].copy()

    calibration = data[
        (
            data.index
            >
            tune_end
        )
        &
        (
            data["target_time"]
            <=
            calibration_end
        )
    ].copy()

    test = data[
        data.index
        >
        calibration_end
    ].copy()

    return (
        train,
        tune,
        calibration,
        test
    )


# ============================================================
# 4. POINT MODEL BUILDERS
# ============================================================

def build_ridge(alpha):

    return Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=alpha
            )
        )
    ])


def build_rf(config):

    return RandomForestRegressor(
        **config,
        n_jobs=-1,
        random_state=
            RANDOM_STATE
    )


def build_xgb(config):

    return XGBRegressor(
        **config,
        objective=
            "reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=
            RANDOM_STATE
    )


def get_saved_backbone(
    household,
    horizon
):

    row = cp_metadata[
        (
            cp_metadata[
                "Household"
            ]
            ==
            household
        )
        &
        (
            cp_metadata[
                "Horizon"
            ]
            ==
            horizon
        )
    ]

    if len(row) != 1:

        raise ValueError(
            f"Backbone lookup failed: "
            f"{household}, {horizon}"
        )

    row = row.iloc[0]

    model_name = (
        row["Backbone"]
    )

    config = ast.literal_eval(
        row[
            "Backbone_Config"
        ]
    )

    return (
        model_name,
        config
    )


def fit_backbone(
    model_name,
    config,
    train
):

    if model_name == "Ridge":

        model = build_ridge(
            config["alpha"]
        )

    elif model_name == "RandomForest":

        model = build_rf(
            config
        )

    elif model_name == "XGBoost":

        model = build_xgb(
            config
        )

    else:

        raise ValueError(
            model_name
        )

    model.fit(
        train[FEATURES],
        train["target"]
    )

    return model


# ============================================================
# 5. CONFORMAL QUANTILE
# ============================================================

def conformal_quantile(
    scores,
    alpha=0.10
):

    scores = np.asarray(
        scores
    )

    if len(scores) == 0:

        raise ValueError(
            "Empty conformal scores."
        )

    n = len(scores)

    level = (
        np.ceil(
            (n+1)
            *
            (1-alpha)
        )
        /
        n
    )

    level = min(
        level,
        1.0
    )

    return np.quantile(
        scores,
        level,
        method="higher"
    )


# ============================================================
# 6. INTERVAL METRICS
# ============================================================

def interval_metrics(
    y,
    lower,
    upper,
    alpha=0.10
):

    y = np.asarray(y)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    covered = (
        (y >= lower)
        &
        (y <= upper)
    )

    width = (
        upper-lower
    )

    score_each = (
        width
        +
        (2/alpha)
        *
        (lower-y)
        *
        (y < lower)
        +
        (2/alpha)
        *
        (y-upper)
        *
        (y > upper)
    )

    return {

        "PICP":
            float(
                covered.mean()
            ),

        "MPIW":
            float(
                width.mean()
            ),

        "IntervalScore":
            float(
                score_each.mean()
            )
    }


def interval_metrics_detailed(
    y,
    lower,
    upper,
    alpha=0.10
):

    y = np.asarray(y)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    covered = (
        (y >= lower)
        &
        (y <= upper)
    )

    width = (
        upper-lower
    )

    score_each = (
        width
        +
        (2/alpha)
        *
        (lower-y)
        *
        (y < lower)
        +
        (2/alpha)
        *
        (y-upper)
        *
        (y > upper)
    )

    return {

        "PICP":
            float(
                covered.mean()
            ),

        "MPIW":
            float(
                width.mean()
            ),

        "IntervalScore":
            float(
                score_each.mean()
            ),

        "CoveredN":
            int(
                covered.sum()
            ),

        "EligibleN":
            int(
                len(y)
            ),

        "WidthSum":
            float(
                width.sum()
            ),

        "IntervalScoreSum":
            float(
                score_each.sum()
            )
    }


# ============================================================
# 7. DAY 4 BETA
# ============================================================

def get_saved_beta(
    household,
    horizon
):

    row = day4_results[
        (
            day4_results[
                "Household"
            ]
            ==
            household
        )
        &
        (
            day4_results[
                "Horizon"
            ]
            ==
            horizon
        )
    ]

    if len(row) != 1:

        raise ValueError(
            f"Beta lookup failed: "
            f"{household}, {horizon}"
        )

    return float(
        row.iloc[0][
            "Beta"
        ]
    )


# ============================================================
# 8. TWIN / CONSISTENCY FEATURES
# ============================================================

def make_twin_features(df):

    x = df.copy()

    for col in [
        "pv_kw",
        "load_kw"
    ]:

        for lag in [
            1,
            2,
            4,
            96
        ]:

            x[
                f"{col}_past_{lag}"
            ] = (
                x[col]
                .shift(lag)
            )

        x[
            f"{col}_past_roll_4"
        ] = (
            x[col]
            .shift(1)
            .rolling(
                4,
                min_periods=4
            )
            .mean()
        )

        x[
            f"{col}_past_roll_8"
        ] = (
            x[col]
            .shift(1)
            .rolling(
                8,
                min_periods=8
            )
            .mean()
        )

    local = (
        x.index
        .tz_convert(
            "Europe/Berlin"
        )
    )

    hour = (
        local.hour
        +
        local.minute/60
    )

    x["current_hour_sin"] = np.sin(
        2*np.pi*hour/24
    )

    x["current_hour_cos"] = np.cos(
        2*np.pi*hour/24
    )

    dow = (
        local.dayofweek
    )

    x["current_dow_sin"] = np.sin(
        2*np.pi*dow/7
    )

    x["current_dow_cos"] = np.cos(
        2*np.pi*dow/7
    )

    doy = (
        local.dayofyear
    )

    x["current_doy_sin"] = np.sin(
        2*np.pi*doy/365.25
    )

    x["current_doy_cos"] = np.cos(
        2*np.pi*doy/365.25
    )

    return x


PV_TWIN_FEATURES = [

    "pv_kw_past_1",
    "pv_kw_past_2",
    "pv_kw_past_4",
    "pv_kw_past_96",

    "pv_kw_past_roll_4",
    "pv_kw_past_roll_8",

    "current_hour_sin",
    "current_hour_cos",

    "current_dow_sin",
    "current_dow_cos",

    "current_doy_sin",
    "current_doy_cos"
]


LOAD_TWIN_FEATURES = [

    "load_kw_past_1",
    "load_kw_past_2",
    "load_kw_past_4",
    "load_kw_past_96",

    "load_kw_past_roll_4",
    "load_kw_past_roll_8",

    "current_hour_sin",
    "current_hour_cos",

    "current_dow_sin",
    "current_dow_cos",

    "current_doy_sin",
    "current_doy_cos"
]


def build_twin_model():

    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=
            "reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=
            RANDOM_STATE
    )


# ============================================================
# 9. RESIDUAL NORMALIZATION
# ============================================================

def fit_residual_scale(
    pred,
    residual
):

    pred = np.asarray(
        pred
    )

    residual = np.asarray(
        residual
    )

    abs_res = np.abs(
        residual
    )

    A = np.column_stack([
        np.ones(
            len(pred)
        ),
        np.abs(pred)
    ])

    coef, _ = nnls(
        A,
        abs_res
    )

    robust_floor = max(
        float(
            np.quantile(
                abs_res,
                0.10
            )
        ),
        1e-3
    )

    s0 = max(
        float(
            coef[0]
        ),
        robust_floor
    )

    c = max(
        float(
            coef[1]
        ),
        0.0
    )

    return (
        s0,
        c
    )


# ============================================================
# 10. TRAIN CONSISTENCY ASSETS
# ============================================================

def train_consistency_assets(
    df
):

    (
        train_end,
        tune_end,
        calibration_end
    ) = get_cp_boundaries(
        df
    )

    x = make_twin_features(
        df
    )

    required = list(
        set(
            PV_TWIN_FEATURES
            +
            LOAD_TWIN_FEATURES
            +
            [
                "pv_kw",
                "load_kw"
            ]
        )
    )

    x = (
        x.dropna(
            subset=required
        )
        .copy()
    )

    train = x[
        x.index
        <=
        train_end
    ].copy()

    tune = x[
        (
            x.index
            >
            train_end
        )
        &
        (
            x.index
            <=
            tune_end
        )
    ].copy()

    pv_model = (
        build_twin_model()
    )

    pv_model.fit(
        train[
            PV_TWIN_FEATURES
        ],
        train[
            "pv_kw"
        ]
    )

    load_model = (
        build_twin_model()
    )

    load_model.fit(
        train[
            LOAD_TWIN_FEATURES
        ],
        train[
            "load_kw"
        ]
    )

    tune_pv_pred = (
        pv_model.predict(
            tune[
                PV_TWIN_FEATURES
            ]
        )
    )

    tune_load_pred = (
        load_model.predict(
            tune[
                LOAD_TWIN_FEATURES
            ]
        )
    )

    r_pv = (
        tune["pv_kw"].values
        -
        tune_pv_pred
    )

    r_load = (
        tune["load_kw"].values
        -
        tune_load_pred
    )

    pv_s0, pv_c = (
        fit_residual_scale(
            tune_pv_pred,
            r_pv
        )
    )

    load_s0, load_c = (
        fit_residual_scale(
            tune_load_pred,
            r_load
        )
    )

    pv_scale = (
        pv_s0
        +
        pv_c
        *
        np.abs(
            tune_pv_pred
        )
    )

    load_scale = (
        load_s0
        +
        load_c
        *
        np.abs(
            tune_load_pred
        )
    )

    z_pv = (
        np.abs(
            r_pv
        )
        /
        pv_scale
    )

    z_load = (
        np.abs(
            r_load
        )
        /
        load_scale
    )

    rho_raw = np.maximum(
        z_pv,
        z_load
    )

    rho_smooth = (
        pd.Series(
            rho_raw,
            index=tune.index
        )
        .ewm(
            halflife=2,
            adjust=False
        )
        .mean()
    )

    rho_normal = float(
        np.median(
            rho_smooth
        )
    )

    return {

        "pv_model":
            pv_model,

        "load_model":
            load_model,

        "pv_s0":
            float(pv_s0),

        "pv_c":
            float(pv_c),

        "load_s0":
            float(load_s0),

        "load_c":
            float(load_c),

        "rho_normal":
            rho_normal
    }


# ============================================================
# 11. COMPUTE FROZEN CONSISTENCY SIGNAL
# ============================================================

def compute_consistency_fixed(
    df,
    assets
):

    x = make_twin_features(
        df
    )

    required = list(
        set(
            PV_TWIN_FEATURES
            +
            LOAD_TWIN_FEATURES
            +
            [
                "pv_kw",
                "load_kw"
            ]
        )
    )

    valid = (
        x.dropna(
            subset=required
        )
        .copy()
    )

    valid[
        "pv_expected"
    ] = (
        assets[
            "pv_model"
        ]
        .predict(
            valid[
                PV_TWIN_FEATURES
            ]
        )
    )

    valid[
        "load_expected"
    ] = (
        assets[
            "load_model"
        ]
        .predict(
            valid[
                LOAD_TWIN_FEATURES
            ]
        )
    )

    valid["r_pv"] = (
        valid["pv_kw"]
        -
        valid[
            "pv_expected"
        ]
    )

    valid["r_load"] = (
        valid["load_kw"]
        -
        valid[
            "load_expected"
        ]
    )

    pv_scale = (
        assets["pv_s0"]
        +
        assets["pv_c"]
        *
        np.abs(
            valid[
                "pv_expected"
            ]
        )
    )

    load_scale = (
        assets["load_s0"]
        +
        assets["load_c"]
        *
        np.abs(
            valid[
                "load_expected"
            ]
        )
    )

    valid["z_pv"] = (
        np.abs(
            valid["r_pv"]
        )
        /
        pv_scale
    )

    valid["z_load"] = (
        np.abs(
            valid["r_load"]
        )
        /
        load_scale
    )

    valid["rho_raw"] = np.maximum(
        valid["z_pv"],
        valid["z_load"]
    )

    valid["rho_smooth"] = (
        valid["rho_raw"]
        .ewm(
            halflife=2,
            adjust=False
        )
        .mean()
    )

    valid["rho_tilde"] = np.maximum(
        0,
        valid["rho_smooth"]
        -
        assets[
            "rho_normal"
        ]
    )

    return valid


# ============================================================
# 12. APPLY FROZEN EVENT(S)
# ============================================================

def apply_event_catalog(
    clean_df,
    event_catalog
):

    x = clean_df.copy()

    x["is_disturbed"] = False
    x["event_id"] = -1

    for _, event in (
        event_catalog.iterrows()
    ):

        start = event["start"]
        end = event["end"]

        mask = (
            (x.index >= start)
            &
            (x.index <= end)
        )

        pre_start = (
            start
            -
            pd.Timedelta(
                hours=24
            )
        )

        pre = (
            x.loc[
                (
                    x.index
                    >=
                    pre_start
                )
                &
                (
                    x.index
                    <
                    start
                ),
                "load_kw"
            ]
            .dropna()
        )

        if len(pre) > 0:

            baseline_load = float(
                pre.median()
            )

        else:

            baseline_load = float(
                x[
                    "load_kw"
                ]
                .median()
            )

        severity = float(
            event["severity"]
        )

        shift_type = (
            event["shift_type"]
        )

        if shift_type in [
            "PV",
            "COMBINED"
        ]:

            x.loc[
                mask,
                "pv_kw"
            ] = (
                x.loc[
                    mask,
                    "pv_kw"
                ]
                *
                (1-severity)
            )

        if shift_type in [
            "LOAD",
            "COMBINED"
        ]:

            added_load = (
                severity
                *
                baseline_load
            )

            x.loc[
                mask,
                "load_kw"
            ] = (
                x.loc[
                    mask,
                    "load_kw"
                ]
                +
                added_load
            )

        x.loc[
            mask,
            "is_disturbed"
        ] = True

        x.loc[
            mask,
            "event_id"
        ] = int(
            event[
                "event_id"
            ]
        )

    x["net_load_kw"] = (
        x["load_kw"]
        -
        x["pv_kw"]
    )

    return x


print(
    "BASE FUNCTIONS DEFINED"
)

# ===== CELL 4 =====
# ============================================================
# DAY 5R CORRECTED FUNCTIONS
# ============================================================


# ============================================================
# 1. HORIZON-DELAYED ACI
# ============================================================

def run_aci_delayed(
    y_true,
    y_pred,
    initial_scores,
    gamma,
    horizon_steps,
    alpha=0.10,
    max_buffer=2000
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    n = len(
        y_true
    )

    scores = list(
        np.asarray(
            initial_scores
        )
    )

    alpha_t = alpha

    lowers = np.full(
        n,
        np.nan
    )

    uppers = np.full(
        n,
        np.nan
    )

    alpha_history = np.full(
        n,
        np.nan
    )

    for t in range(n):

        # ------------------------------------
        # Only now does outcome t-h become known
        # ------------------------------------

        j = (
            t
            -
            horizon_steps
        )

        if j >= 0:

            miss = int(
                (
                    y_true[j]
                    <
                    lowers[j]
                )
                or
                (
                    y_true[j]
                    >
                    uppers[j]
                )
            )

            alpha_t = (
                alpha_t
                +
                gamma
                *
                (
                    alpha
                    -
                    miss
                )
            )

            alpha_t = np.clip(
                alpha_t,
                0.001,
                0.50
            )

            scores.append(
                abs(
                    y_true[j]
                    -
                    y_pred[j]
                )
            )

            if len(scores) > max_buffer:

                scores = (
                    scores[
                        -max_buffer:
                    ]
                )

        q = conformal_quantile(
            scores,
            alpha=
                alpha_t
        )

        lowers[t] = (
            y_pred[t]
            -
            q
        )

        uppers[t] = (
            y_pred[t]
            +
            q
        )

        alpha_history[t] = (
            alpha_t
        )

    return (
        lowers,
        uppers,
        alpha_history
    )


# ============================================================
# 2. DELAYED ACI GAMMA TUNING
# ============================================================

def tune_delayed_aci_gamma(
    tune_y,
    tune_pred,
    gamma_grid,
    horizon_steps,
    alpha=0.10
):

    tune_y = np.asarray(
        tune_y
    )

    tune_pred = np.asarray(
        tune_pred
    )

    errors = np.abs(
        tune_y
        -
        tune_pred
    )

    n = len(
        tune_y
    )

    split = max(
        200,
        int(
            0.30*n
        )
    )

    initial_scores = (
        errors[
            :split
        ]
    )

    stream_y = (
        tune_y[
            split:
        ]
    )

    stream_pred = (
        tune_pred[
            split:
        ]
    )

    rows = []

    for gamma in gamma_grid:

        lo, hi, _ = (
            run_aci_delayed(
                stream_y,
                stream_pred,
                initial_scores,
                gamma=gamma,
                horizon_steps=
                    horizon_steps,
                alpha=alpha
            )
        )

        m = interval_metrics(
            stream_y,
            lo,
            hi,
            alpha
        )

        rows.append({

            "gamma":
                float(gamma),

            "PICP":
                m["PICP"],

            "MPIW":
                m["MPIW"],

            "IntervalScore":
                m[
                    "IntervalScore"
                ],

            "CoverageError":
                abs(
                    m["PICP"]
                    -
                    (1-alpha)
                )
        })

    table = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "CoverageError",
                "IntervalScore"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    best_gamma = float(
        table.iloc[0][
            "gamma"
        ]
    )

    return (
        best_gamma,
        table
    )


# ============================================================
# 3. EVENT STOP TIME
#
# Frozen CSV "end" is last affected 15-min row.
# True experimental duration is start + duration_min.
# ============================================================

def event_stop_time(
    event
):

    return (
        event["start"]
        +
        pd.Timedelta(
            minutes=
                int(
                    event[
                        "duration_min"
                    ]
                )
        )
    )


# ============================================================
# 4. CAUSALLY ELIGIBLE FORECASTS
# ============================================================

def causal_event_mask(
    test_df,
    event
):

    start = (
        event["start"]
    )

    stop = (
        event_stop_time(
            event
        )
    )

    return (
        (
            test_df.index
            >=
            start
        )
        &
        (
            test_df.index
            <
            stop
        )
        &
        (
            test_df[
                "target_time"
            ]
            <=
            stop
        )
    )


# ============================================================
# 5. ADAPTATION DELAY
# ============================================================

def adaptation_delay_corrected(
    index,
    lower,
    upper,
    event,
    pre_steps=8,
    threshold_ratio=
        WIDTH_THRESHOLD_RATIO
):

    width = pd.Series(
        np.asarray(upper)
        -
        np.asarray(lower),
        index=index
    )

    start = (
        event["start"]
    )

    stop = (
        event_stop_time(
            event
        )
    )

    pre = (
        width[
            width.index
            <
            start
        ]
        .tail(
            pre_steps
        )
    )

    if len(pre) < 3:

        return np.nan

    baseline = float(
        pre.median()
    )

    threshold = (
        threshold_ratio
        *
        baseline
    )

    search_end = (
        stop
        +
        pd.Timedelta(
            hours=2
        )
    )

    search = width[
        (
            width.index
            >=
            start
        )
        &
        (
            width.index
            <=
            search_end
        )
    ]

    hit = search[
        search
        >=
        threshold
    ]

    if len(hit) == 0:

        return np.nan

    first_hit = (
        hit.index[0]
    )

    return float(
        (
            first_hit
            -
            start
        ).total_seconds()
        /60
    )


# ============================================================
# 6. CAUSAL WARNING LEAD
# ============================================================

def warning_lead_corrected(
    test_df,
    y,
    scp_lower,
    scp_upper,
    prop_lower,
    prop_upper,
    event
):

    start = (
        event["start"]
    )

    stop = (
        event_stop_time(
            event
        )
    )

    idx = (
        test_df.index
    )

    y = pd.Series(
        np.asarray(y),
        index=idx
    )

    scp_lower = pd.Series(
        np.asarray(
            scp_lower
        ),
        index=idx
    )

    scp_upper = pd.Series(
        np.asarray(
            scp_upper
        ),
        index=idx
    )

    prop_width = pd.Series(
        np.asarray(
            prop_upper
        )
        -
        np.asarray(
            prop_lower
        ),
        index=idx
    )

    pre = (
        prop_width[
            prop_width.index
            <
            start
        ]
        .tail(8)
    )

    if len(pre) < 3:

        return np.nan

    threshold = (
        WIDTH_THRESHOLD_RATIO
        *
        float(
            pre.median()
        )
    )

    # Proposed must widen while event is active
    after = prop_width[
        (
            prop_width.index
            >=
            start
        )
        &
        (
            prop_width.index
            <
            stop
        )
    ]

    wide = after[
        after
        >=
        threshold
    ]

    if len(wide) == 0:

        return np.nan

    t_wide = (
        wide.index[0]
    )

    eligible = (
        causal_event_mask(
            test_df,
            event
        )
    )

    misses = (
        (
            y
            <
            scp_lower
        )
        |
        (
            y
            >
            scp_upper
        )
    )

    eligible_misses = (
        eligible
        &
        misses
    )

    miss_rows = (
        test_df.loc[
            eligible_misses
        ]
    )

    if len(miss_rows) == 0:

        return np.nan

    # Miss becomes observable when its target occurs
    t_miss_observed = (
        miss_rows[
            "target_time"
        ]
        .min()
    )

    return float(
        (
            t_miss_observed
            -
            t_wide
        ).total_seconds()
        /60
    )


print(
    "CORRECTED ACI + CAUSAL FUNCTIONS DEFINED"
)

# ===== CELL 5 =====
def prepare_task_assets_corrected(
    clean_df,
    household,
    horizon,
    horizon_steps,
    consistency_assets
):

    (
        train,
        tune,
        cal,
        test
    ) = make_cp_splits(
        clean_df,
        horizon_steps
    )

    (
        model_name,
        config
    ) = get_saved_backbone(
        household,
        horizon
    )

    point_model = fit_backbone(
        model_name,
        config,
        train
    )

    # ========================================================
    # TUNE
    # ========================================================

    pred_tune = (
        point_model.predict(
            tune[
                FEATURES
            ]
        )
    )

    y_tune = (
        tune[
            "target"
        ]
        .values
    )

    tune_error = np.abs(
        y_tune
        -
        pred_tune
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    pred_cal = (
        point_model.predict(
            cal[
                FEATURES
            ]
        )
    )

    y_cal = (
        cal[
            "target"
        ]
        .values
    )

    cal_scores = np.abs(
        y_cal
        -
        pred_cal
    )

    # ========================================================
    # SCP
    # ========================================================

    q_scp = conformal_quantile(
        cal_scores,
        ALPHA
    )

    # ========================================================
    # LACP
    # ========================================================

    scale_model = (
        RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=
                RANDOM_STATE
        )
    )

    scale_model.fit(
        tune[
            FEATURES
        ],
        tune_error
    )

    scale_floor = max(
        1e-3,
        float(
            np.quantile(
                tune_error,
                0.10
            )
        )
    )

    scale_cal = np.maximum(
        scale_model.predict(
            cal[
                FEATURES
            ]
        ),
        scale_floor
    )

    lacp_scores = (
        cal_scores
        /
        scale_cal
    )

    q_lacp = conformal_quantile(
        lacp_scores,
        ALPHA
    )

    # ========================================================
    # PROPOSED
    # ========================================================

    beta = get_saved_beta(
        household,
        horizon
    )

    clean_consistency = (
        compute_consistency_fixed(
            clean_df,
            consistency_assets
        )
    )

    common_cal = (
        cal.index
        .intersection(
            clean_consistency.index
        )
    )

    rho_cal = (
        clean_consistency.loc[
            common_cal,
            "rho_tilde"
        ]
    )

    sigma_cal = (
        1
        +
        beta
        *
        rho_cal
    )

    pred_cal_series = pd.Series(
        pred_cal,
        index=cal.index
    )

    prop_scores = (
        np.abs(
            cal.loc[
                common_cal,
                "target"
            ]
            -
            pred_cal_series.loc[
                common_cal
            ]
        )
        /
        sigma_cal
    )

    q_prop = conformal_quantile(
        prop_scores,
        ALPHA
    )

    # ========================================================
    # DELAYED ACI GAMMA
    # ========================================================

    (
        best_gamma,
        gamma_table
    ) = tune_delayed_aci_gamma(
        y_tune,
        pred_tune,
        ACI_GAMMAS,
        horizon_steps=
            horizon_steps,
        alpha=ALPHA
    )

    return {

        "Backbone":
            model_name,

        "point_model":
            point_model,

        "scale_model":
            scale_model,

        "scale_floor":
            scale_floor,

        "q_scp":
            float(
                q_scp
            ),

        "q_lacp":
            float(
                q_lacp
            ),

        "cal_scores":
            cal_scores,

        "beta":
            float(
                beta
            ),

        "q_prop":
            float(
                q_prop
            ),

        "aci_gamma":
            float(
                best_gamma
            ),

        "aci_gamma_table":
            gamma_table
    }


print(
    "CORRECTED TASK ASSET FUNCTION DEFINED"
)

# ===== CELL 6 =====
def evaluate_single_event_corrected(
    shifted_df,
    event,
    household,
    horizon,
    horizon_steps,
    task_assets,
    consistency_assets
):

    shifted_features = (
        make_features(
            shifted_df,
            horizon_steps
        )
    )

    (
        train_end,
        tune_end,
        cal_end
    ) = get_cp_boundaries(
        shifted_df
    )

    required = (
        FEATURES
        +
        [
            "target",
            "target_time"
        ]
    )

    test = (
        shifted_features[
            shifted_features.index
            >
            cal_end
        ]
        .dropna(
            subset=required
        )
        .copy()
    )

    # ========================================================
    # POINT FORECAST
    # ========================================================

    pred = pd.Series(
        task_assets[
            "point_model"
        ]
        .predict(
            test[
                FEATURES
            ]
        ),
        index=test.index
    )

    y = (
        test[
            "target"
        ]
        .copy()
    )

    # ========================================================
    # CONSISTENCY SIGNAL
    # ========================================================

    consistency = (
        compute_consistency_fixed(
            shifted_df,
            consistency_assets
        )
    )

    common = (
        test.index
        .intersection(
            consistency.index
        )
    )

    test = (
        test.loc[
            common
        ]
    )

    pred = (
        pred.loc[
            common
        ]
    )

    y = (
        y.loc[
            common
        ]
    )

    rho = (
        consistency.loc[
            common,
            "rho_tilde"
        ]
    )

    # ========================================================
    # SCP
    # ========================================================

    q_scp = (
        task_assets[
            "q_scp"
        ]
    )

    scp_lower = (
        pred
        -
        q_scp
    )

    scp_upper = (
        pred
        +
        q_scp
    )

    # ========================================================
    # LACP
    # ========================================================

    local_scale = np.maximum(
        task_assets[
            "scale_model"
        ]
        .predict(
            test[
                FEATURES
            ]
        ),
        task_assets[
            "scale_floor"
        ]
    )

    local_scale = pd.Series(
        local_scale,
        index=common
    )

    lacp_half = (
        task_assets[
            "q_lacp"
        ]
        *
        local_scale
    )

    lacp_lower = (
        pred
        -
        lacp_half
    )

    lacp_upper = (
        pred
        +
        lacp_half
    )

    # ========================================================
    # PROPOSED
    # ========================================================

    sigma = (
        1
        +
        task_assets[
            "beta"
        ]
        *
        rho
    )

    prop_half = (
        task_assets[
            "q_prop"
        ]
        *
        sigma
    )

    prop_lower = (
        pred
        -
        prop_half
    )

    prop_upper = (
        pred
        +
        prop_half
    )

    # ========================================================
    # CORRECTED HORIZON-DELAYED ACI
    # ========================================================

    (
        aci_lo,
        aci_hi,
        aci_alpha
    ) = run_aci_delayed(

        y.values,

        pred.values,

        task_assets[
            "cal_scores"
        ],

        gamma=
            task_assets[
                "aci_gamma"
            ],

        horizon_steps=
            horizon_steps,

        alpha=
            ALPHA
    )

    aci_lower = pd.Series(
        aci_lo,
        index=common
    )

    aci_upper = pd.Series(
        aci_hi,
        index=common
    )

    # ========================================================
    # CAUSAL EVENT PREDICTIONS
    # ========================================================

    event_mask = (
        causal_event_mask(
            test,
            event
        )
    )

    event_idx = (
        test.index[
            event_mask
        ]
    )

    if len(event_idx) == 0:

        return (
            pd.DataFrame(),
            {
                "eligible":
                    False,

                "EligibleN":
                    0
            }
        )

    # ========================================================
    # EXACT EVENT-WIDTH-MATCHED STATIC ABLATION
    #
    # Same event MPIW as Proposed.
    # Analysis-only oracle baseline.
    # ========================================================

    mean_prop_event_width = float(
        (
            prop_upper.loc[
                event_idx
            ]
            -
            prop_lower.loc[
                event_idx
            ]
        )
        .mean()
    )

    wm_half = (
        mean_prop_event_width
        / 2
    )

    wm_lower = (
        pred
        -
        wm_half
    )

    wm_upper = (
        pred
        +
        wm_half
    )

    methods = {

        "SCP":
            (
                scp_lower,
                scp_upper
            ),

        "LACP":
            (
                lacp_lower,
                lacp_upper
            ),

        "ACI":
            (
                aci_lower,
                aci_upper
            ),

        "Proposed":
            (
                prop_lower,
                prop_upper
            ),

        "WidthMatched":
            (
                wm_lower,
                wm_upper
            )
    }

    rows = []

    for method, (
        lower,
        upper
    ) in methods.items():

        m = interval_metrics_detailed(

            y.loc[
                event_idx
            ],

            lower.loc[
                event_idx
            ],

            upper.loc[
                event_idx
            ],

            ALPHA
        )

        delay = (
            adaptation_delay_corrected(
                common,
                lower,
                upper,
                event
            )
        )

        rows.append({

            "Household":
                household,

            "Horizon":
                horizon,

            "EventID":
                int(
                    event[
                        "event_id"
                    ]
                ),

            "ShiftType":
                event[
                    "shift_type"
                ],

            "Severity":
                float(
                    event[
                        "severity"
                    ]
                ),

            "DurationMin":
                int(
                    event[
                        "duration_min"
                    ]
                ),

            "Method":
                method,

            "EventPICP":
                m[
                    "PICP"
                ],

            "EventMPIW":
                m[
                    "MPIW"
                ],

            "EventIntervalScore":
                m[
                    "IntervalScore"
                ],

            "CoveredN":
                m[
                    "CoveredN"
                ],

            "EligibleN":
                m[
                    "EligibleN"
                ],

            "WidthSum":
                m[
                    "WidthSum"
                ],

            "IntervalScoreSum":
                m[
                    "IntervalScoreSum"
                ],

            "AdaptDelayMin":
                delay,

            "AdaptedWithinEvent":
                (
                    False
                    if np.isnan(
                        delay
                    )
                    else
                    delay
                    <=
                    int(
                        event[
                            "duration_min"
                        ]
                    )
                )
        })

    # ========================================================
    # CORRECTED WARNING LEAD
    # ========================================================

    warning_lead = (
        warning_lead_corrected(

            test,

            y,

            scp_lower,

            scp_upper,

            prop_lower,

            prop_upper,

            event
        )
    )

    details = {

        "eligible":
            True,

        "EligibleN":
            len(
                event_idx
            ),

        "test":
            test,

        "y":
            y,

        "pred":
            pred,

        "rho":
            rho,

        "sigma":
            sigma,

        "methods":
            methods,

        "event_idx":
            event_idx,

        "warning_lead":
            warning_lead,

        "aci_alpha":
            pd.Series(
                aci_alpha,
                index=common
            )
    }

    return (
        pd.DataFrame(
            rows
        ),
        details
    )


print(
    "INDEPENDENT EVENT EVALUATION FUNCTION DEFINED"
)
