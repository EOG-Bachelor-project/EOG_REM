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

![Python](https://img.shields.io/badge/python-3.10+-blue)
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

### 2. Run post-install script

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

The script installs GSSC without its declared dependencies and then re-pins numpy to `>=1.24,<2.0` to remain compatible with numba. It finishes by verifying that numpy, numba, and gssc all import correctly.

#### Deactivate environment
```bash
conda deactivate
```

---
## Usage
![CPU](https://img.shields.io/badge/CPU-supported-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-optional-yellow)

The pipeline is driven by `main.py` and runs in two phases: **preprocessing** and **feature extraction**. GSSC inference runs on CPU by default — if a CUDA-capable GPU is available, set `use_cuda=True` in `GSSC_to_csv.py` and `extract_rems_n.py` for faster staging.

### Phase 1 — Preprocessing (`main.py process`)

Processes raw EDF recordings through 7 stages per patient:
1. Extract EOG signals (LOC, ROC) from EDF
2. Run GSSC sleep staging
3. Extract REM eye movement events
4. Mask artefacts (amplitude > 300 µV)
5. Detect & classify eye movements
6. Extract EEG proxy signals
7. Merge all outputs into a unified CSV

The pipeline skips any stage whose output already exists. To reprocess a patient, delete its intermediate files and re-run.

```bash
python main.py process /data/raw                  # process up to 10 patients
python main.py process /data/raw --batch-size 5   # process 5
```

### Phase 2 — Feature extraction (`main.py extract`)

Extracts 115 features per subject across five modules (`eog`, `gssc`, `eeg`, `bout`, `patient`) and merges them into `features_csv/features.csv`.

```bash
python main.py extract patient_info.xlsx                    # all modules
python main.py extract patient_info.xlsx --modules bout eog # specific modules
python main.py extract patient_info.xlsx --force            # re-extract from scratch
```

### Full pipeline

```bash
python main.py all /data/raw patient_info.xlsx         # process + extract + report
python main.py all /data/raw patient_info.xlsx --force # full pipeline, re-extract features
```

### Report & cleanup

```bash
python main.py report    # generate HTML report from features.csv
python main.py cleanup   # compress intermediate CSVs to free disk space
python main.py cleanup --dry-run  # preview without compressing
```

### Model evaluation

After feature extraction, use `evaluate.py` to assess a trained model:

```python
from evaluate import evaluate_model

evaluate_model(
    model=fitted_model,
    X_test=X_test,
    y_test=y_test,
    class_names=["Control", "RBD", "PD"],
    model_name="RandomForest",
    save_dir="reports/",
)
# Saves confusion matrix, ROC curves, and feature importance plots to reports/randomforest.pdf
```

---
## Features

115 features are extracted per subject across five modules, saved to `features_csv/features.csv`.

| Module | Features | Description |
|--------|----------|-------------|
| `eog` | 51 | EOG signal properties and eye movement characteristics during REM |
| `gssc` | 12 | Sleep staging probabilities and REM stability from the GSSC model |
| `eeg` | 31 | EEG proxy band power (delta, theta, alpha, beta) per sleep stage |
| `bout` | 17 | Phasic/tonic bout structure during REM |
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

> A full feature cheat-sheet with descriptions and distribution plots is available in the HTML report generated by `python main.py report`.
---
## Notebooks

Two notebooks are provided for post-extraction inspection and quality control.

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
