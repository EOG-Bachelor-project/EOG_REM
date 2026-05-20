# Filename: model2.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Question:     "Does our RBD-specific EOG feature extraction add value beyond what standard sleep metrics already tell you?"
# Description:  Defines the feature set for Model 2 — the clinical baseline model.
#               Model 2 uses only standard sleep macrostructure metrics and EEG 
#               spectral features, i.e. features a clinician would have access to 
#               without any RBD-specific EOG analysis.


# Pipeline overview:
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Define MODEL2_FEATURES: the explicit list of macrostructure + spectral columns
# 2. get_model2_features(df): filters df to MODEL2_FEATURES that are present
# 3. Used by compare_models.py to select columns before calling prepare()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ================================================================================
# Imports
# ================================================================================

from __future__ import annotations  
import pandas as pd

# ================================================================================
# Model 2 feature set
# ================================================================================

# ---- EEG spectral features ----
# Band power (delta, theta, alpha, beta, total) and theta ratio per sleep stage.
EEG_SPECTRAL_FEATURES = [
    # Wake
    "eeg__w__delta", "eeg__w__theta", "eeg__w__alpha", "eeg__w__beta",
    "eeg__w__total", "eeg__w__theta_ratio",
    # N1
    "eeg__n1__delta", "eeg__n1__theta", "eeg__n1__alpha", "eeg__n1__beta",
    "eeg__n1__total", "eeg__n1__theta_ratio",
    # N2
    "eeg__n2__delta", "eeg__n2__theta", "eeg__n2__alpha", "eeg__n2__beta",
    "eeg__n2__total", "eeg__n2__theta_ratio",
    # N3
    "eeg__n3__delta", "eeg__n3__theta", "eeg__n3__alpha", "eeg__n3__beta",
    "eeg__n3__total", "eeg__n3__theta_ratio",
    # REM
    "eeg__rem__delta", "eeg__rem__theta", "eeg__rem__alpha", "eeg__rem__beta",
    "eeg__rem__total", "eeg__rem__theta_ratio",
    # Overall
    "eeg__overall__theta_beta_ratio",
    # REM-specific EEG band power (delta, theta, gamma during REM sub-epochs)
    "eeg_delta_rem_power", "eeg_theta_rem_power", "eeg_gamma_rem_power",
    "eeg_theta_delta_ratio_rem",
    "eeg_delta_phasic_power", "eeg_theta_phasic_power", "eeg_gamma_phasic_power",
    "eeg_theta_delta_ratio_phasic",
    "eeg_delta_tonic_power", "eeg_theta_tonic_power", "eeg_gamma_tonic_power",
    "eeg_theta_delta_ratio_tonic",
]

# ---- Sleep macrostructure features ----
# Standard clinical sleep architecture metrics derived from 30-s epoch staging.
MACROSTRUCTURE_FEATURES = [
    # Recording duration
    "total_recording_min",          # total recording time (lights off to lights on)
    # Wake
    "w_duration_min",               # total wake duration (proxy for WASO)
    "w_fraction",                   # wake fraction of total recording
    # N1
    "n1_duration_min",              # time in N1
    "n1_fraction",                  # N1 fraction of total sleep time
    # N2
    "n2_duration_min",              # time in N2
    "n2_fraction",                  # N2 fraction of total sleep time
    # N3 (slow wave sleep)
    "n3_duration_min",              # time in N3
    "n3_fraction",                  # N3 fraction of total sleep time
    # REM
    "rem_duration_min",             # total REM duration
    "rem_fraction",                 # REM fraction of total sleep time
    "rem_latency_min",              # latency from sleep onset to first REM epoch
    # REM cycle structure
    "n_rem_cycles",                 # number of REM cycles
    "n_rem_epochs",                 # total number of REM epochs
    "rem_epoch_count",              # number of continuous REM bouts
    "rem_epoch_mean_duration_min",  # mean REM bout duration
    "rem_epoch_std_duration_min",   # variability of REM bout duration
    "rem_epoch_min_duration_min",   # shortest REM bout
    "rem_epoch_max_duration_min",   # longest REM bout
]

# ---- Combined Model 2 feature set ----
MODEL2_FEATURES: list[str] = EEG_SPECTRAL_FEATURES + MACROSTRUCTURE_FEATURES


# ================================================================================
# Assign group labels
# ================================================================================
def get_model2_features(df: pd.DataFrame) -> list[str]:
    """
    Return the subset of MODEL2_FEATURES that are present in df.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (output of load_features).

    Returns
    -------
    list[str]
        Feature column names available in df from the Model 2 set.
    """
    available = [f for f in MODEL2_FEATURES if f in df.columns]
    missing   = [f for f in MODEL2_FEATURES if f not in df.columns]

    if missing:
        print(f"  [model2] {len(missing)} features not found in df and will be skipped:")
        for f in missing:
            print(f"    - {f}")

    print(f"  [model2] Using {len(available)} / {len(MODEL2_FEATURES)} Model 2 features")
    return available