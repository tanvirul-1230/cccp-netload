# Consistency-Conditioned Conformal Prediction for Residential Net Load Forecasting under Abrupt Shifts

Code, settings, seeds, data and results behind the paper

I. Jahan, M. S. Israt and M. T. Islam, "Consistency-Conditioned Conformal Prediction for Residential Net Load
Forecasting under Abrupt Shifts with Application to Home Battery Reserves," in *Proc. 29th Int. Conf. Computer
and Information Technology (ICCIT)*, Cox's Bazar, Bangladesh, Dec. 2026.

Three result files (`Additional_Analyses/final3/<run>/event_rows_test.csv.gz`, about 134 MB each) exceed
GitHub's file limit and are attached to release [v1.0](https://github.com/tanvirul-1230/cccp-netload/releases/tag/v1.0).
`bash fetch_results.sh` downloads them; `reproduce.sh` runs it first.

## Requirements

- Python 3.12 and `requirements.txt` (the versions used).
- NVIDIA GPU with CUDA for the model runs (XGBoost and PyTorch on the GPU). Not needed to rebuild the paper from
  the included results.
- pdfLaTeX and BibTeX. The figures use Times New Roman.

## Reproduce

```
pip install -r requirements.txt
bash reproduce.sh paper    # numbers, tables, figures, checks and PDF from the included results
bash reproduce.sh full     # everything from the data archive; overwrites the included results
```

On Windows run the scripts from Git Bash. `PYTHON` selects the interpreter.

## Contents

| Path | Content |
|---|---|
| `data/hems_data.zip` | Households residential3, residential4 and residential6 (H3, H4, H6), 15 min, and the 161-event set of the original study |
| `data/prepare_data.py` | Builds the processed households from the OPSD file; `--check` compares with the archive |
| `Additional_Analyses/her_shift.py`, `engine.py`, `engine2.py`, `engine3.py` | Forecasters, normal-behaviour models, consistency score, conformal intervals, ACI, event injection and scoring |
| `Additional_Analyses/nn_backbone.py` | Multilayer-perceptron backbone (robustness run) |
| `Additional_Analyses/make_events3.py` | All injected event sets with their seeds; checks the included files |
| `Additional_Analyses/run3.py`, `pipeline3.sh`, `merge3.py`, `select_params3.py` | Model runs per household, development-stage selection, test stage |
| `Additional_Analyses/final_stats2.py` | Tables and tests of a run (`final3/v3_tree/tables`) |
| `Additional_Analyses/final/events`, `final3/events_dev`, `final3/events_stress` | Test, development and held-out event sets |
| `Additional_Analyses/final3/v3_tree`, `v3_mlp`, `v3_tree_clip` | Results: tree backbone, neural backbone, load clipped at zero |
| `Paper/analysis` | Paper numbers, event trace, battery reserve |
| `Paper/figures` | `make_figures.py` (data figures), `fig_method.tex` (Fig. 1), figure data |
| `Paper/verification_code` | Checks of every statement in the text and their outputs |

## Settings and seeds

| Item | Setting | Defined in |
|---|---|---|
| Data | OPSD cumulative 15-min readings differenced and divided by 0.25 h: the value at time stamp t is the mean power (kW) over [t, t + 15 min); a forecast issued at step t is made at the end of that interval. Removed rows are set to missing on a full 15-min grid | `data/prepare_data.py` |
| Splits | training / tune / development / calibration / test = 50/10/10/10/20 % in time; targets crossing a boundary dropped | `engine3.py` |
| Point forecaster | lowest tune-split RMSE per household and horizon among ridge (alpha 0.1, 1, 10, standardised inputs), random forest (200 trees depth 12, or 300 trees depth 20, min leaf 2) and gradient-boosted trees (300 trees depth 4 rate 0.05, 500 trees depth 4 rate 0.03, 300 trees depth 6 rate 0.05; subsample and column sample 0.8) | `engine.py` (`RIDGE_ALPHAS`, `RF_CONFIGS`, `XGB_CONFIGS`) |
| Normal-behaviour models | gradient-boosted trees, 300 trees, depth 4, rate 0.05, subsample and column sample 0.8, fitted on the training split | `engine.py` (`Builders.twin`) |
| Residual scale, eq. (4) | nonnegative least squares of the absolute residual on the absolute prediction over the tune split; s0 >= max(10th percentile of the absolute residual, 1e-3), c >= 0 | `her_shift.py` (`fit_residual_scale`) |
| Consistency score | EWMA half-life 2 steps (30 min), not tuned; m = tune-split median of the smoothed score | `engine.py` |
| LACP scale s(x) | random forest of the absolute tune-split error, 200 trees, depth 10, min leaf 20 (XGBoost random forest on the GPU, row subsample 0.632), floored at its 10th percentile | `engine.py` (`Builders.lacp_scale`) |
| ACI and rolling | alpha = 0.1, alpha_t clipped to [0.001, 0.5], latest 2000 scores, started from the calibration scores (tune scores on the development split); rolling = step 0. Each event is injected alone and resumes the ACI and rolling state of the run without injection at its onset | `engine2.py` (`aci_resume`), `engine3.py` |
| Neural backbone | multilayer perceptron 256-256-128, dropout 0.1, AdamW (rate 1e-3, weight decay 1e-4), batches of 256, early stopping on tune RMSE (patience 20) | `nn_backbone.py` |
| Model seeds | 42 (`her_shift.RANDOM_STATE`, `nn_backbone.SEED`) | |
| Event seeds (H3 / H4 / H6) | test set 3103 / 3104 / 3106, development 6103 / 6104 / 6106, PV trips 7103 / 7104 / 7106, 3.7-kW steps 8103 / 8104 / 8106 | `make_events3.py` (`SEEDS`) |
| Second event set | original study: two events per configuration and household (162), 161 scored | `data/hems_data.zip` (`results/day5_events`) |
| Bootstrap seeds | 2026 (`paper_numbers.py`), 29 (`paper_extra.py`), 31 (`battery_stats.py`), 42 (`final_stats2.py`); 4000 resamples of household-days within each household | `Paper/analysis` |

## Results

Files in `Additional_Analyses/final3/<run>/`:

- `event_rows_test.csv.gz`: one row per event, horizon, method and window (`Condition`: `shift` injected,
  `placebo` same window without injection; `EventSet`: `extended`, `frozen`, `trip`, `ev`). `N_in`, `Cov_in`,
  `W_in`, `IS_in`: number of inside-event targets, covered targets, summed width and summed interval score (kW);
  `_rec` the recovery target, `_tgt` every target inside the disturbance; `R`: test-period range of net load;
  `Worig`: mean width over issue times in the event; `Ad125`: width above 1.25 times the pre-event median.
- `clean_metrics_test.csv`: coverage, mean width and nIS per household, horizon and method in the test period.
- `clean_per_forecast.csv.gz`: every test forecast with its interval for the main methods.
- `selection.json`, `selection_table.csv`: selected gains and steps and the development scores behind them.
- `Paper/figures/data/event_trace.csv.gz`: width and coverage at every issue time around each event;
  `battery_events.csv.gz`: battery reserve per event.

Method names: `SCP`, `LACP`, `CC|b<beta>` (CC-CP), `CCLACP|b<beta>` (CC-LACP), `ACI|g<gamma>`,
`ACICC|b|g` (ACI on CC-CP), `ACIL|g` (ACI on LACP), `ACICCL|b|g` (ACI on CC-LACP), `ROLL|...` (rolling window),
`PVonly`, `LoadOnly`, `Persist`, `LF-load`, `LF-both` (signal variants, suffix `L` on the LACP scale),
`THL|q<level>|b` (indicator), `WM-CC`, `WM-CCLACP` (width-matched controls), `ACI|tuned`, `CC|tuned`
(tune-split parameters).

## Data

Open Power System Data, Data Package Household Data, version 2020-04-15,
https://data.open-power-system-data.org/household_data/2020-04-15/, licence CC BY 4.0.

Changes made to the data: cumulative readings differenced and divided by 0.25 h, negative differences,
interpolated rows and two meter jumps in residential3 set to missing, load computed as import plus PV minus export
(`data/prepare_data.py`).

## Licence

Code: MIT (`LICENSE`). Data: derived from Open Power System Data, CC BY 4.0. Paper sources in `Paper/`:
(c) 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other
uses, in any current or future media, including reprinting/republishing this material for advertising or
promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse
of any copyrighted component of this work in other works.
