<div align="center">
    <h1>
        EOG_REM: <br> 
        EOG-Only Digital Biomarkers for REM Sleep Behavior Disorder and Parkinson’s Disease
    </h1>
</div> 

<div align="center">
    
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![CPU](https://img.shields.io/badge/CPU-supported-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-optional-yellow)
</div>    

#### Goal: <br>
Develop and validate EOG-only markers of abnormal REM physiology and build machine-learning models to detect RBD (REM Sleep Behavior Disorder) and PD (Parkinson's Disease) in a mixed clinical cohort.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Credits / Acknowledgements](#credits--acknowledgements)

---

## Background

REM Sleep Behavior Disorder (RBD) is a strong prodromal marker of Parkinson's disease (PD) and related synucleinopathies. The polysomnographic hallmark of RBD is REM Sleep Without Atonia (RSWA), typically quantified using chin/limb EMG. However, many emerging wearable sleep systems do not record high-quality EMG, motivating the need for biomarkers that rely on minimal channels.

Electrooculography (EOG) is present in essentially all PSGs and can capture both eye movements and peri-ocular/facial muscle contamination, potentially serving as an indirect proxy for RSWA. This project develops and validates EOG-only markers using polysomnography recordings from the Danish Center for Sleep Medicine (DCSM).

**Expected outcomes:**
- A reproducible EOG-only feature pipeline for REM phenotyping.
- Quantitative evidence for which EOG-derived markers best capture RBD/PD signatures.
- A validated ML screening model suitable for translation to minimal-sensor wearable paradigms.

---

## Install

![Python](https://img.shields.io/badge/python-3.11-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

Clone the repository:
```bash
git clone https://github.com/EOG-Bachelor-project/EOG_REM.git
cd EOG_REM
```

### 1. Create conda environment

#### Windows
```powershell
conda env create -f environment-win.yml
conda activate BPML
```

#### macOS/Linux
```bash
conda env create -f environment-mac.yml
conda activate BPML
```

### 2. Install GSSC

> This step is required. GSSC must be installed separately from GitHub to avoid pip overriding the pinned numpy version required by numba.

#### Windows
```powershell
.\post_install.bat
```

#### macOS/Linux
```bash
pip install git+https://github.com/jshanna100/gssc.git --no-deps
pip install "numpy>=1.24,<2.0" --force-reinstall
```

#### Deactivate environment
```bash
conda deactivate
```

---

## Usage

![CPU](https://img.shields.io/badge/CPU-supported-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-optional-yellow)

The pipeline is driven by `main.py` and runs in two phases: **preprocessing** and **feature extraction**. GSSC inference runs on CPU by default — if a CUDA-capable GPU is available, set `use_cuda=True` in `GSSC_to_csv.py` and `extract_rems_n.py` for faster staging.

### Data setup

> **Note:** This pipeline was developed using data from the Danish Center for Sleep Medicine (DCSM) and some parts are written with their data format in mind. If you are using a different dataset, some parts of the code may need to be adapted.
```bash
/data/raw/
├── PatientMetadata.xlsx      # metadata and diagnostic labels (see format below)
├── PATIENT_1/
│   ├── recording.edf
│   └── lights.txt
├── PATIENT_2/
│   ├── recording.edf
│   └── lights.txt
└── ...
```
The `lights.txt` files must contain the lights-off and lights-on timestamps used to trim each recording to the intended sleep period. The Excel file must contain at minimum a session ID column and a diagnostic group column matching the expected labels: `Control`, `iRBD`, `PD(+RBD)`, `PD(-RBD)`, `PLM`.

### Phase 1 — Preprocessing (`main.py process`)

Processes raw EDF recordings through 7 stages per patient:

1. Index sessions from raw EDF directory
2. Extract EOG signals (LOC, ROC) from EDF —> `eog_csv/`
3. Run GSSC automated sleep staging —> `gssc_csv/`
4. Extract REM eye movement events —> `extracted_rems/`
5. Detect & classify eye movements (phasic/tonic) —> `detected_ems/`
6. Extract EEG proxy signals via DTCWT —> `eeg_csv/`
7. Merge all outputs into a unified CSV —> `merged_csv_eog/`

The pipeline skips any stage whose output already exists. To reprocess from scratch:

```bash
rm -rf eog_csv/ gssc_csv/ extracted_rems/ detected_ems/ eeg_csv/ merged_csv_eog/
python main.py process /data/raw --batch-size 9999
```

Otherwise:

```bash
python main.py process /data/raw                   # process up to 10 patients
python main.py process /data/raw --batch-size 50   # process 50
```

### Phase 2 — Feature extraction (`main.py extract`)

Extracts features per subject across five modules (`eog`, `gssc`, `eeg`, `bout`, `patient`) and merges them into `features_csv/features.csv`.

```bash
python main.py extract GlostrupRBDData.xlsx                     # all modules
python main.py extract GlostrupRBDData.xlsx --modules bout eog  # specific modules
python main.py extract GlostrupRBDData.xlsx --force             # re-extract from scratch
```

### Full pipeline

```bash
python main.py all /data/raw GlostrupRBDData.xlsx          # process + extract + report
python main.py all /data/raw GlostrupRBDData.xlsx --force  # full pipeline, re-extract features
```

### Report & cleanup

```bash
python main.py report               # generate HTML report from features.csv
python main.py cleanup              # compress intermediate CSVs to free disk space
python main.py cleanup --dry-run    # preview without compressing
```

### Machine learning

```bash
python ml/train.py --mode binary --binary-mode control_vs_irbd
python ml/train.py --mode multiclass
```

**Modes:**
- `binary`: pairwise — `control_vs_irbd`, `control_vs_pd_rbd`, `control_vs_pd_nrbd`
- `multiclass`: four-class — Control, iRBD, PD(+RBD), PD(−RBD)

Results and figures are saved to `reports/evaluation/`.

### Statistical analysis

After model training, three scripts support the statistical analysis and interpretation of results:

```bash
# 1. Assess normality of feature distributions per class
python ml/qq-plot.py --csv features_csv/features.csv --label-col group --positive-class iRBD --negative-class Control --out-dir reports/qq

# 2. Run univariate hypothesis testing (Welch's t-test or Mann-Whitney U)
python ml/univariate_stats.py --csv features_csv/features.csv --label-col group --out-dir reports/stats

# 3. Aggregate feature importance across all sweep runs
python ml/aggregate_importance.py --sweep-dir reports/sweep --out-dir reports/importance
```

Run in order: QQ-plots inform which test to use in the univariate step, and aggregate importance combines model results across runs.

---
## Features

115 features are extracted per subject across five modules, saved to `features_csv/features.csv`.

| Module | Features | Description |
|--------|----------|-------------|
| `eog` | 51 | EOG signal properties and eye movement characteristics during REM |
| `gssc` | 12 | Sleep staging probabilities and REM stability from the GSSC model |
| `eeg` | 31 | EEG proxy band power (delta, theta, alpha, beta) per sleep stage |
| `bout` | 17 | Phasic/tonic bout structure during REM |
| `extra` | 30 | Spectral band power per REM context (phasic/tonic), extended phasic/tonic bout structure, EM morphology, and sleep architecture |
| `patient` | 4 | Demographics and clinical metadata from patient records |

### Feature groups

**Sleep structure (17)** — recording and REM duration, stage fractions, REM episode count and duration statistics.

**EOG amplitude (6)** — mean, std, and 95th percentile of LOC/ROC amplitude during REM.

**REM events (8)** — count, rate, duration, amplitude, and rise slope of detected REM eye movement events.

**EM classification (10)** — slow (SEM) vs. rapid eye movement counts, rates, fractions, durations, and amplitudes during REM.

**EM stage counts (5)** — total eye movement counts per sleep stage (W, N1, N2, N3, REM).

**Phasic / Tonic (4)** — sub-epoch counts and fractions; `phasic_fraction` is a key RBD biomarker.

**Phasic / Tonic bouts (17)** — bout count, duration statistics, and rate for phasic and tonic REM sub-epochs; phasic↔tonic transition count.

**GSSC probabilities (8)** — mean stage probabilities during REM; `rem_certainty` replicates the Cesari et al. micro-sleep structure feature.

**REM stability (4)** — staging stability index, fragmentation index, wake-transition fraction, and overall REM amount.

**EEG band power (31)** — delta, theta, alpha, beta, total power, and theta ratio per stage (REM, N1, N2, N3, W); overall theta/beta ratio.

**Spectral band power by REM context (12)** — delta, theta, and gamma power during overall REM, phasic REM, and tonic REM; theta/delta ratio per context.

**Extended phasic/tonic structure (12)** — bout duration percentiles (p25, p75, p90), longest bout, long bout fraction, and phasic/tonic transition rate; latency to first phasic bout from REM onset.

**EM morphology (4)** — mean rise and fall slopes averaged across LOC and ROC channels, amplitude variance, and SEM fraction across all EM events.

**Sleep architecture (2)** — REM latency from sleep onset and number of distinct REM cycles.

> A full feature cheat-sheet with descriptions and distribution plots is available in the HTML report generated by `python main.py report`.

---
## Notebooks

Three notebooks are provided for post-extraction inspection, quality control, and results analysis.

### `feat_dashboard.ipynb`
Feature extraction dashboard and data quality checks.
- Run extraction and append new patients to an existing `features.csv`
- NaN counts per feature and per subject, with detailed per-patient missing feature reports
- Group label completeness check (Control, iRBD, PD(-RBD), PD(+RBD), PLM)
- Compress merged CSVs to free disk space
- Clean up duplicate ID/label columns after merging

### `feat_inspect.ipynb`
Cohort overview and per-patient inspector.
- Summary table: subject count, feature count, NaN distribution across the cohort
- Group breakdown bar chart and feature category completeness (colour-coded by completeness %)
- NaN audit: features ranked by missing rate
- Feature distributions overlaid by diagnostic group
- Per-patient inspector: set a `PATIENT_ID` to see all feature values, NaN flags, and a z-score comparison against the patient's group mean

### `sweep_summary.ipynb`
Model sweep results analysis. Useful for collecting and evaluating results across all training runs.
- Summary table of all runs with CV and test balanced accuracy
- Bar chart of test balanced accuracy by classification mode and model
- Overfitting check: CV vs. test accuracy difference per run
- McNemar p-value heatmaps per classification mode
- Exports all figures to a PDF report

---
## License

---

## Credits / Acknowledgements
- Thanks to our supervisor and co-supervisor for their guidance and support:
    - A. Brink-Kjaer
    - P. Jenum
    - U. Hanif
- Sleep staging powered by [GSSC](https://github.com/bdsp-core/GSSC)
- REM event detection based on the `detect_rem_jaec` algorithm
