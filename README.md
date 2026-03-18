# EOG_REM

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

Clone the repository:
```bash
git clone https://github.com/EOG-Bachelor-project/EOG_REM.git
cd EOG_REM
```

### Setup environment

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

#### Deactivate environment
```bash
conda deactivate
```

---
## Usage
The pipeline runs in four main stages:

### 1. Extract EOG signals from EDF
Extracts LOC and ROC channels from raw EDF recordings and saves them as CSV. Channels are automatically renamed from Danish conventions (EOGV → LOC, EOGH → ROC).
```python
from preprocessing.edf_to_csv import edf_to_csv
from pathlib import Path

edf_to_csv(
    edf_path=Path("path/to/DCSM__/contiguous.edf"),
    lights_path=Path("path/to/DCSM__/lights.txt")  # optional, trims to sleep period
)
```

### 2. Run GSSC sleep staging
Runs the GSSC deep learning model on EOG channels to produce per-epoch sleep stage predictions (W, N1, N2, N3, REM) with probabilities.
```python
from preprocessing.GSSC_to_csv import GSSC_to_csv

GSSC_to_csv(
    edf_path=Path("path/to/DCSM__/contiguous.edf"),
    lights_path=Path("path/to/DCSM__/lights.txt")
)
```

### 3. Extract REM eye movement events
Detects REM eye movement events using the `detect_rem_jaec` algorithm, aligned to the GSSC hypnogram.
```python
from preprocessing.extract_rems_n import extract_rems_from_edf

extract_rems_from_edf(
    edf_path=Path("path/to/DCSM__/contiguous.edf"),
    lights_path=Path("path/to/DCSM__/lights.txt")
)
```

### 4. Merge into unified CSV
Merges the EOG signal, GSSC staging, and REM event annotations into a single per-sample CSV for downstream feature extraction and analysis.
```python
from preprocessing.merge import merge_all
from pathlib import Path

merge_all(
    eog_file=Path("eog_csv/DCSM___contiguous_eog.csv"),
    gssc_file=Path("gssc_csv/DCSM___gssc.csv"),
    events_file=Path("extracted_rems/DCSM___extracted_rems.csv"),
    output_file=Path("merged_csv_eog/DCSM___merged.csv"),
)
```

### 5. Visualize
Plot EOG epochs for a given sleep stage, full-night overviews, or stage transition windows.
```python
from analysis.plot import plot_eog_epochs, plot_fullnight_overview

# Plot 30-second REM epochs with eye movement overlays
plot_eog_epochs(
    file="merged_csv_eog/DCSM___merged.csv",
    stage="REM",
    window_sec=30.0,
    max_epochs=10,
)

# Full-night overview
plot_fullnight_overview(file="merged_csv_eog/DCSM___merged.csv")
```

---
## Features


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