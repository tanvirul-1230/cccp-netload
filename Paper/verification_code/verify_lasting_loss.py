"""
Two limitation statements, checked on the corrected pipeline (engine2 signals):
 1  during a 120-min PV loss the consistency score decays as the PV model follows
    the loss: excess of the injected over the placebo score by minute after onset
 2  load steps are small relative to PV losses: mean injected change in kW
Output: verify_lasting_loss_output.txt
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AA = os.path.abspath(os.path.join(HERE, "..", "..", "Additional_Analyses"))
sys.path.insert(0, AA)
import her_shift as hs      # noqa: E402
import engine as en         # noqa: E402
import engine2 as e2        # noqa: E402

ZIP = os.environ.get("HEMS_ZIP") or next(p for p in (os.path.join(AA, "..", "data", "hems_data.zip"),
                            os.path.join(AA, "..", "Conference", "Conference",
                                         "HEMS_CP_Paper_COMPLETE_Day7_Backup.zip")) if os.path.exists(p))
HOMES = {"H3": "residential3", "H4": "residential4", "H6": "residential6"}
OUT = open(os.path.join(HERE, "verify_lasting_loss_output.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")


z = zipfile.ZipFile(ZIP)
B = en.Builders("cuda")
prof, mags = [], []
for h, name in HOMES.items():
    df = hs.regularize_15min(pd.read_csv(io.BytesIO(z.read(f"data/processed/{name}_clean.csv"))))
    df["net_load_kw"] = df["load_kw"] - df["pv_kw"]
    H = e2.prepare_household(df, h, B)
    ev = pd.read_csv(os.path.join(AA, "final", "events", f"{h}_extended_events.csv"))
    for c in ["start", "end"]:
        ev[c] = pd.to_datetime(ev[c], utc=True)
    for _, e in ev.iterrows():
        stop = hs.event_stop_time(e)
        win = df.loc[e["start"] - pd.Timedelta(hours=30): stop + pd.Timedelta(hours=2)]
        inj = hs.apply_event_catalog(win, pd.DataFrame([e]))
        m = (win.index >= e["start"]) & (win.index < stop)
        mags.append(dict(Household=h, ShiftType=e["shift_type"],
                         dPV=float((win.pv_kw[m] - inj.pv_kw[m]).mean()),
                         dLoad=float((inj.load_kw[m] - win.load_kw[m]).mean())))
        if e["shift_type"] != "PV" or int(e["duration_min"]) != 120:
            continue
        rho_s, _ = en.rho_all(inj, H["twins"], H["S_clean"])
        rs = rho_s[e2.SIGNALS["CC"]]
        rc = H["rho_clean"][e2.SIGNALS["CC"]].reindex(rs.index)
        for k in range(8):
            t = e["start"] + pd.Timedelta(minutes=15 * k)
            if t in rs.index and t in rc.index:
                prof.append(dict(Household=h, Severity=e["severity"], Minute=15 * k,
                                 excess=float(rs.loc[t] - rc.loc[t])))
    del H
P = pd.DataFrame(prof)
tab = P.groupby(["Severity", "Minute"]).excess.mean().unstack("Severity")
say("Mean excess of the consistency score (injected - placebo) during 120-min PV losses, by minute after onset")
say(tab.round(3).to_string())
peak = tab.max()
say("\nShare of the peak remaining at each minute")
say((tab / peak).round(2).to_string())
M = pd.DataFrame(mags)
say("\nMean injected change inside the event (kW)")
say(M.groupby(["ShiftType", "Household"])[["dPV", "dLoad"]].mean().round(3).to_string())
OUT.close()
