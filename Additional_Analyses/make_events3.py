"""
Every injected event set, drawn with fixed seeds; existing files are checked against
the generator instead of overwritten.
  final/events/H*_extended_events.csv    1620 test events (20 per configuration and household)
  final3/events_dev/H*_dev_events.csv     810 development events (10 per configuration and household)
  final3/events_stress/H*_trip|ev_events.csv   held-out complete PV trips and 3.7 kW load steps
The 161-event set of the original study is read from data/hems_data.zip (results/day5_events).
  python make_events3.py
"""
import io
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import her_shift as hs      # noqa: E402
import engine as en         # noqa: E402
import engine3 as e3        # noqa: E402

ZIP = os.environ.get("HEMS_ZIP") or next(p for p in (os.path.join(HERE, "..", "data", "hems_data.zip"),
                            os.path.join(HERE, "..", "Conference", "Conference",
                                         "HEMS_CP_Paper_COMPLETE_Day7_Backup.zip")) if os.path.exists(p))
HOMES = {"H3": "residential3", "H4": "residential4", "H6": "residential6"}
SEEDS = {"extended": {"H3": 3103, "H4": 3104, "H6": 3106}, "dev": {"H3": 6103, "H4": 6104, "H6": 6106},
         "trip": {"H3": 7103, "H4": 7104, "H6": 7106}, "ev": {"H3": 8103, "H4": 8104, "H6": 8106}}
z = zipfile.ZipFile(ZIP)


def keep(path, ev):
    new = pd.read_csv(io.StringIO(ev.to_csv(index=False)))
    if os.path.exists(path):
        assert pd.read_csv(path).equals(new), f"{path} differs from the generator"
        print(f"{os.path.relpath(path, HERE)}: {len(new)} events, matches the generator")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new.to_csv(path, index=False)
        print(f"{os.path.relpath(path, HERE)}: {len(new)} events written")


for h, name in HOMES.items():
    df = hs.regularize_15min(pd.read_csv(io.BytesIO(z.read(f"data/processed/{name}_clean.csv"))))
    df["net_load_kw"] = df["load_kw"] - df["pv_kw"]
    _, tu_end, dv_end, ca_end = e3.boundaries3(df)
    keep(os.path.join(HERE, "final", "events", f"{h}_extended_events.csv"),
         en.generate_events(df, 20, SEEDS["extended"][h]))
    keep(os.path.join(HERE, "final3", "events_dev", f"{h}_dev_events.csv"),
         e3.generate_events3(df, tu_end + pd.Timedelta(days=2), dv_end - pd.Timedelta(hours=8), 10, SEEDS["dev"][h]))
    t_from, t_to = ca_end + pd.Timedelta(days=2), df.index.max()
    keep(os.path.join(HERE, "final3", "events_stress", f"{h}_trip_events.csv"),
         e3.generate_events3(df, t_from, t_to, 20, SEEDS["trip"][h], types=("PV",), sevs=(1.0,)))
    keep(os.path.join(HERE, "final3", "events_stress", f"{h}_ev_events.csv"),
         e3.generate_events3(df, t_from, t_to, 20, SEEDS["ev"][h], types=("EV",), sevs=(3.7,)))
