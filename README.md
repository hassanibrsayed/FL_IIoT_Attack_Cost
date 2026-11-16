# Client Heterogeneity Lowers the Attack Cost of Byzantine-Robust Federated Intrusion Detection

Code and per-run results for the manuscript by Hassan I. Ahmed (Electronics Research Institute, Cairo, Egypt).

The study measures the **empirical attack cost**: the smallest number of malicious clients (out of 10) that breaks a
federated intrusion detector. It does so for FedAvg, coordinate-wise median, trimmed mean and several defences, under
IID and Dirichlet non-IID partitions, on Edge-IIoTset and CIC-IoT2023.

## Contents

| Path | Content |
|---|---|
| `poc_edge_iiot.py` | data loading, cleaning, class capping, temporal split, static label-flip / sign-flip grid, leakage check |
| `adaptive_attack.py` | adaptive (white-box, colluding) attacker; aggregators `fedavg`, `median`, `trimmed`, `krum`, `*_clip`, `cosw`, `cosdrop`, `cclip`, `bucket_*`; `--dispersion` mode; cost summaries |
| `prepare_ciciot.py` | builds the class-capped CIC-IoT2023 sample (`ciciot_sample.csv`) from the per-capture CSV files |
| `cic_winners_ci.py` | bootstrap CIs and winning strategies (balanced-detection criterion) |
| `threshold_sensitivity.py` | attack cost recomputed for other break thresholds (no retraining) |
| `mechanism_analysis.py` | spread of honest updates versus attack cost |
| `run_*.ps1` | the experiment launchers (Windows PowerShell) |
| `paper/` | scripts that build the figures and the manuscript from the result files |
| `poc_full2/` | static attack grid, Edge-IIoTset, 5 seeds |
| `adaptive_out/`, `cic_adaptive/` | adaptive attacker, 10 seeds, Edge-IIoTset / CIC-IoT2023 |
| `defense_out/`, `cic_defense/`, `cic_defense_partial/` | norm clipping, 5 seeds |
| `cos_out/`, `cic_cos/`, `cic_cos_partial/` | cosine-similarity filters, 5 seeds |
| `cic_hetdef/` | centred clipping and bucketing, CIC-IoT2023, 5 seeds |
| `disp_edge/`, `disp_cic/` | clean runs with honest-update dispersion logs, 10 seeds |
| `analysis/` | threshold sensitivity and mechanism outputs |

Every run is appended to `adaptive_results.csv` (or `fl_results.csv`) as soon as it finishes, so each folder holds
the raw per-run metrics as well as the summaries (`adaptive_cost_per_seed.csv`, `adaptive_cost.csv`, ...).

## Data

The datasets are not redistributed here; download them from their providers:

* **Edge-IIoTset** (Ferrag et al., IEEE Access 10, 40281–40306, 2022): file `DNN-EdgeIIoT-dataset.csv`.
* **CIC-IoT2023** (Neto et al., Sensors 23, 5941, 2023): per-capture CSV files, one folder per class.
  Build the sample with `python prepare_ciciot.py` (see its header for options).

The launchers expect `archive (8)/DNN-EdgeIIoT-dataset.csv` and `ciciot_sample.csv`; edit the paths if yours differ.

## Requirements

Python 3.11 with numpy 1.26, pandas, scikit-learn, PyTorch (CUDA optional), matplotlib and python-docx (paper build only).

## Reproducing

```
# adaptive attacker, Edge-IIoTset (10 seeds)
python adaptive_attack.py --data "archive (8)/DNN-EdgeIIoT-dataset.csv" --seeds 10 --out adaptive_out
# adaptive attacker, CIC-IoT2023 (balanced-detection criterion)
python adaptive_attack.py --data ciciot_sample.csv --static none --static-breaks none --crit bd \
    --aggs fedavg median trimmed --fl-rows 100000 --rounds 30 --seeds 10 --out cic_adaptive
# revision experiments (dispersion, centred clipping, bucketing)
powershell .\run_revision.ps1
# analyses and manuscript
python threshold_sensitivity.py
python mechanism_analysis.py
python paper/make_figures.py
python paper/build_paper_srep.py
```

All scripts skip runs that are already saved, so an interrupted run can be restarted with the same command.
