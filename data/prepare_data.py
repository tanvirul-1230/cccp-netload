"""
Processed household records from the Open Power System Data household package
(version 2020-04-15, 15-min, single index):
  cumulative meter readings differenced, negative differences and OPSD-interpolated
  rows removed, kWh per 15 min -> kW, load = import + PV - export, net load = load - PV,
  two meter jumps in H3 removed, calendar columns added.

  python prepare_data.py [household_data_15min_singleindex.csv]
      -> processed/residential{3,4,6}_clean.csv
  python prepare_data.py [csv] --check
      -> compares with the files in hems_data.zip
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://data.open-power-system-data.org/household_data/2020-04-15/household_data_15min_singleindex.csv"
H3_METER_JUMPS = ["2017-07-08 18:00:00+02:00",   # grid-import cumulative-meter jump
                  "2017-04-19 02:00:00+02:00"]   # night-time PV cumulative-meter jump


def prepare_household(df, household):
    imp = f"DE_KN_residential{household}_grid_import"
    exp = f"DE_KN_residential{household}_grid_export"
    pv = f"DE_KN_residential{household}_pv"
    x = df[["timestamp", "interpolated", imp, exp, pv]].copy()
    x["grid_import_kwh"] = x[imp].diff()
    x["grid_export_kwh"] = x[exp].diff()
    x["pv_kwh"] = x[pv].diff()
    for c in ["grid_import_kwh", "grid_export_kwh", "pv_kwh"]:
        x.loc[x[c] < 0, c] = np.nan
    x["bad_interpolation"] = x["interpolated"].fillna("").apply(lambda s: any(c in s for c in [imp, exp, pv]))
    x.loc[x["bad_interpolation"], ["grid_import_kwh", "grid_export_kwh", "pv_kwh"]] = np.nan
    x["grid_import_kw"] = x["grid_import_kwh"] / 0.25
    x["grid_export_kw"] = x["grid_export_kwh"] / 0.25
    x["pv_kw"] = x["pv_kwh"] / 0.25
    x["load_kw"] = x["grid_import_kw"] + x["pv_kw"] - x["grid_export_kw"]
    x["net_load_kw"] = x["load_kw"] - x["pv_kw"]
    return x.dropna(subset=["load_kw", "pv_kw", "net_load_kw"]).copy()


def add_time_features(x):
    x = x.copy()
    x["hour"] = x["timestamp"].dt.hour + x["timestamp"].dt.minute / 60
    x["dayofweek"] = x["timestamp"].dt.dayofweek
    x["month"] = x["timestamp"].dt.month
    x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24)
    return x


def build(src):
    df = pd.read_csv(src)
    df["timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True).dt.tz_convert("Europe/Berlin")
    out = {}
    for h in (3, 4, 6):
        x = add_time_features(prepare_household(df, h))
        if h == 3:
            x = x[~x["timestamp"].isin(pd.to_datetime(H3_METER_JUMPS))].copy()
        out[h] = x
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data = build(args[0] if args else URL)
    if "--check" in sys.argv:
        z = zipfile.ZipFile(os.path.join(HERE, "hems_data.zip"))
        for h, x in data.items():
            ref = pd.read_csv(io.BytesIO(z.read(f"data/processed/residential{h}_clean.csv")))
            new = pd.read_csv(io.StringIO(x.to_csv(index=False)))
            num = new.select_dtypes("number").columns
            same = (list(new.columns) == list(ref.columns) and len(new) == len(ref)
                    and (new["timestamp"] == ref["timestamp"]).all()
                    and np.allclose(new[num].values, ref[num].values, equal_nan=True, rtol=0, atol=1e-9))
            print(f"H{h}: {len(new)} rows, identical to hems_data.zip: {same}")
    else:
        os.makedirs(os.path.join(HERE, "processed"), exist_ok=True)
        for h, x in data.items():
            x.to_csv(os.path.join(HERE, "processed", f"residential{h}_clean.csv"), index=False)
            print(f"H{h}: {len(x)} rows written")
