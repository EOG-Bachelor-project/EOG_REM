# Filename: feat_report.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Collects EOG, EEG and GSSC features from all merged CSVs and generates
#              an HTML report with distribution plots per feature.


# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from features.eog_feats import extract_features
from features.gssc_feats import extract_gssc_features
from features.eeg_feats import extract_eeg_features
from features.patient_feats import extract_patient_features

# =====================================================================
# 1)  COLLECT FEATURES 
# =====================================================================

def collect_features(
    merged_dir: str | Path,
    fs: float = 250.0,
    pattern: str = "*_merged.csv",
    patient_excel: str | Path | None = None,
    file_list: list[Path] | None = None,
) -> pd.DataFrame:
    """
    Run all feature extractors on every merged CSV and return a single DataFrame.

    Parameters
    ----------
    merged_dir : str or Path
        Directory containing merged CSV files from the preprocessing pipeline.
    fs : float, optional
        Sampling frequency of the recordings (default 250 Hz).
    pattern : str, optional
        Glob pattern to match merged CSV files (default "*_merged.csv").
    patient_excel : str or Path, optional
        Path to the patient Excel file for diagnostic group labels.
    file_list : list of Path, optional
        Explicit list of files to process; overrides merged_dir + pattern.

    Returns
    -------
    pd.DataFrame
        One row per subject, columns are all extracted features.
    """
    merged_dir = Path(merged_dir)

    if file_list is not None:
        files = sorted(file_list)
    else:
        files = sorted(merged_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {merged_dir}")

    n_total = len(files)
    print(f"\nFound {n_total} merged CSV(s) in {merged_dir}\n")

    rows = []
    for i, f in enumerate(files, start=1):
        print(f"  [{i}/{n_total}] {f.name}")
        record: dict = {}

        try:
            record.update(extract_features(f, fs=fs))
        except Exception as e:
            print(f"    [SKIP EOG] {e}")

        try:
            record.update(extract_gssc_features(f, fs=fs))
        except Exception as e:
            print(f"    [SKIP GSSC] {e}")

        try:
            record.update(extract_eeg_features(f, fs=fs))
        except Exception as e:
            print(f"    [SKIP EEG] {e}")

        if patient_excel is not None:
            try:
                record.update(extract_patient_features(f, patient_excel=patient_excel))
            except Exception as e:
                print(f"    [SKIP PATIENT] {e}")

        if record:
            rows.append(record)

    combined = pd.DataFrame(rows)
    print(f"\nDone: {combined.shape[0]} subjects | {combined.shape[1] - 1} features\n")
    return combined


# =====================================================================
# 2)  HTML REPORT  
# =====================================================================
def _svg_no_data(width: int = 520, height: int = 240) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<text x="50%" y="50%" text-anchor="middle" fill="#94A3B8" font-size="13" '
        f'font-style="italic">Not enough data</text></svg>'
    )

def _build_histogram_svg(
        values: list[float],
        x_label: str = "Value",
        width: int = 520,
        height: int = 280,
        ) -> str:
    """
    Build a warm-paper styled SVG histogram with labeled axes.

    Parameters
    ----------
    values : list of float
        Numeric values to plot.
    x_label : str
        Label for the X-axis (feature description + units).
    width, height : int
        SVG dimensions in pixels.

    Returns
    -------
    str
        SVG markup.
    """
    if not values or len(values) < 2:
        return _svg_no_data(width, height)

    n_bins = min(20, max(5, len(values) // 3))
    counts, edges = np.histogram(values, bins=n_bins)
    max_count = max(counts) if max(counts) > 0 else 1

    # Padding reserves room for axis titles
    pad_l, pad_r, pad_t, pad_b = 58, 22, 22, 58
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    bar_w = plot_w / n_bins
    gap = max(1, bar_w * 0.1)

    parts = []

    # --- Horizontal gridlines + Y-axis tick labels ---
    for i in range(5):
        frac = i / 4
        y = pad_t + plot_h * (1 - frac)
        val = int(max_count * frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="#F5EDD8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="#94A3B8" '
            f'text-anchor="end">{val}</text>'
        )

    # --- Bars ---
    for i, c in enumerate(counts):
        bh = (c / max_count) * plot_h if max_count > 0 else 0
        x = pad_l + i * bar_w + gap / 2
        y = pad_t + plot_h - bh
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - gap:.1f}" '
            f'height="{bh:.1f}" fill="#0E7490" rx="2"/>'
        )

    # --- Axis lines ---
    y_axis = pad_t + plot_h
    parts.append(
        f'<line x1="{pad_l}" y1="{y_axis:.1f}" x2="{width - pad_r}" y2="{y_axis:.1f}" '
        f'stroke="#1C2A3A" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{y_axis:.1f}" '
        f'stroke="#1C2A3A" stroke-width="1.2"/>'
    )

    # --- X-axis tick labels (min, mid, max) ---
    vmin, vmax = edges[0], edges[-1]
    vmid = (vmin + vmax) / 2
    tick_y = y_axis + 14
    parts.append(f'<text x="{pad_l}" y="{tick_y}" font-size="11" fill="#475569" text-anchor="start">{vmin:.3g}</text>')
    parts.append(f'<text x="{pad_l + plot_w / 2}" y="{tick_y}" font-size="11" fill="#475569" text-anchor="middle">{vmid:.3g}</text>')
    parts.append(f'<text x="{width - pad_r}" y="{tick_y}" font-size="11" fill="#475569" text-anchor="end">{vmax:.3g}</text>')

    # --- X-axis title ---
    parts.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 10}" font-size="12" '
        f'fill="#1C2A3A" text-anchor="middle" font-weight="600">{x_label}</text>'
    )

    # --- Y-axis title ---
    y_title_x = 16
    y_title_y = pad_t + plot_h / 2
    parts.append(
        f'<text x="{y_title_x}" y="{y_title_y}" font-size="12" fill="#1C2A3A" '
        f'text-anchor="middle" font-weight="600" '
        f'transform="rotate(-90, {y_title_x}, {y_title_y})">Count (subjects)</text>'
    )

    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

def _build_boxplot_svg(
        values: list[float],
        x_label: str = "Value",
        width: int = 520,
        height: int = 260,
        ) -> str:
    """
    Build a SVG boxplot with jittered data points and outliers highlighted.

    Parameters
    ----------
    values : list of float
        Numeric values to plot.
    x_label : str
        Label for the X-axis (feature description + units).
    width, height : int
        SVG dimensions in pixels.
    
    Returns
    -------
    str
        SVG markup for a boxplot with jittered data points and outliers highlighted.
    """
    if not values or len(values) < 2:
        return _svg_no_data(width, height)

    arr = np.array(values)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    whisker_lo = max(arr.min(), q1 - 1.5 * iqr)
    whisker_hi = min(arr.max(), q3 + 1.5 * iqr)
    outliers = arr[(arr < whisker_lo) | (arr > whisker_hi)]

    vmin, vmax = arr.min(), arr.max()
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    pad_l, pad_r, pad_t, pad_b = 28, 28, 38, 58
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def xpos(v):
        return pad_l + (v - vmin) / (vmax - vmin) * plot_w

    box_top = pad_t + plot_h * 0.25
    box_bot = pad_t + plot_h * 0.75
    box_mid = pad_t + plot_h * 0.5
    box_h = box_bot - box_top

    parts = []

    # --- Whisker line ---
    parts.append(
        f'<line x1="{xpos(whisker_lo):.1f}" y1="{box_mid:.1f}" '
        f'x2="{xpos(whisker_hi):.1f}" y2="{box_mid:.1f}" stroke="#64748B" stroke-width="1.5"/>'
    )

    # --- Whisker caps ---
    for wv in [whisker_lo, whisker_hi]:
        x = xpos(wv)
        parts.append(
            f'<line x1="{x:.1f}" y1="{box_top + 10:.1f}" x2="{x:.1f}" y2="{box_bot - 10:.1f}" '
            f'stroke="#64748B" stroke-width="1.5"/>'
        )

    # --- IQR box (ocean palette) ---
    bx = xpos(q1)
    bw = xpos(q3) - xpos(q1)
    parts.append(
        f'<rect x="{bx:.1f}" y="{box_top:.1f}" width="{bw:.1f}" height="{box_h:.1f}" '
        f'fill="#CFFAFE" stroke="#0E7490" stroke-width="2" rx="3"/>'
    )

    # --- Median line (coral accent) ---
    mx = xpos(med)
    parts.append(
        f'<line x1="{mx:.1f}" y1="{box_top:.1f}" x2="{mx:.1f}" y2="{box_bot:.1f}" '
        f'stroke="#F43F5E" stroke-width="2.5"/>'
    )

    # --- Jittered data points ---
    rng = np.random.RandomState(42)
    jitter = rng.uniform(-plot_h * 0.15, plot_h * 0.15, size=len(arr))
    for i, v in enumerate(arr):
        cx = xpos(v)
        cy = box_mid + jitter[i]
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="#0E7490" opacity="0.55"/>'
        )

    # --- Outliers ---
    for v in outliers:
        cx = xpos(v)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{box_mid:.1f}" r="3.5" fill="none" stroke="#F43F5E" stroke-width="1.6"/>'
        )

    # --- X-axis line ---
    y_axis = pad_t + plot_h
    parts.append(
        f'<line x1="{pad_l}" y1="{y_axis:.1f}" x2="{width - pad_r}" y2="{y_axis:.1f}" '
        f'stroke="#1C2A3A" stroke-width="1.2"/>'
    )

    # --- X-axis tick labels ---
    tick_y = y_axis + 14
    vmid_label = (vmin + vmax) / 2
    parts.append(f'<text x="{pad_l}" y="{tick_y}" font-size="11" fill="#475569" text-anchor="start">{vmin:.3g}</text>')
    parts.append(f'<text x="{pad_l + plot_w / 2}" y="{tick_y}" font-size="11" fill="#475569" text-anchor="middle">{vmid_label:.3g}</text>')
    parts.append(f'<text x="{width - pad_r}" y="{tick_y}" font-size="11" fill="#475569" text-anchor="end">{vmax:.3g}</text>')

    # --- Stat labels at top ---
    parts.append(f'<text x="{xpos(q1):.1f}" y="{pad_t - 6}" font-size="10" fill="#64748B" text-anchor="middle">Q1={q1:.2g}</text>')
    parts.append(f'<text x="{xpos(med):.1f}" y="{pad_t - 6}" font-size="10" fill="#F43F5E" text-anchor="middle" font-weight="700">med={med:.2g}</text>')
    parts.append(f'<text x="{xpos(q3):.1f}" y="{pad_t - 6}" font-size="10" fill="#64748B" text-anchor="middle">Q3={q3:.2g}</text>')

    # --- X-axis title ---  
    parts.append(
        f'<text x="{pad_l + plot_w / 2}" y="{height - 10}" font-size="12" '
        f'fill="#1C2A3A" text-anchor="middle" font-weight="600">{x_label}</text>'
    )

    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

# ---- Feature descriptions ----
FEATURE_DESCRIPTIONS = {
    # Sleep structure
    "total_recording_min":         "Total recording duration [minutes]",
    "rem_duration_min":            "Total REM sleep duration [minutes]",
    "rem_fraction":                "Fraction of recording spent in REM",
    "w_duration_min":              "Wake duration [minutes]",
    "w_fraction":                  "Fraction of recording spent awake",
    "n1_duration_min":             "N1 (light sleep) duration [minutes]",
    "n1_fraction":                 "Fraction of recording in N1",
    "n2_duration_min":             "N2 (medium sleep) duration [minutes]",
    "n2_fraction":                 "Fraction of recording in N2",
    "n3_duration_min":             "N3 (deep sleep) duration [minutes]",
    "n3_fraction":                 "Fraction of recording in N3",
    "n_rem_epochs":                "Number of distinct REM episodes across the night",
    "rem_epoch_count":             "Number of distinct REM epochs",
    "rem_epoch_mean_duration_min": "Mean REM epoch duration [minutes]",
    "rem_epoch_std_duration_min":  "Std of REM epoch durations [minutes] - NaN values indicate only a patient has only one REM epoch",
    "rem_epoch_min_duration_min":  "Shortest REM epoch duration [minutes]",
    "rem_epoch_max_duration_min":  "Longest REM epoch duration [minutes]",

    # EOG amplitude
    "rem_loc_mean_abs_uv":       "Mean |LOC| amplitude during REM [uV]",
    "rem_roc_mean_abs_uv":       "Mean |ROC| amplitude during REM [uV]",
    "rem_loc_std_uv":            "Std of LOC signal during REM [uV]",
    "rem_roc_std_uv":            "Std of ROC signal during REM [uV]",
    "rem_loc_p95_uv":            "95th percentile of |LOC| during REM [uV]",
    "rem_roc_p95_uv":            "95th percentile of |ROC| during REM [uV]",

    # REM events
    "rem_event_count":           "Total detected REM eye-movement events",
    "rem_event_rate_per_min":    "REM events per minute of REM sleep",
    "rem_event_mean_duration_s": "Mean event duration [seconds]",
    "rem_event_median_duration_s": "Median event duration [seconds]",
    "rem_event_mean_loc_amp_uv": "Mean LOC peak amplitude across events [uV]",
    "rem_event_mean_roc_amp_uv": "Mean ROC peak amplitude across events [uV]",
    "rem_event_mean_loc_rise_slope": "Mean LOC rise slope [uV/s]",
    "rem_event_mean_roc_rise_slope": "Mean ROC rise slope [uV/s]",

    # EM classification
    "sem_count_rem_sleep":       "Number of Slow Eye Movements (SEMs) during REM",
    "rem_em_count_rem_sleep":    "Number of Rapid Eye Movements during REM",
    "sem_rate_per_min":          "SEMs per minute of REM sleep",
    "rem_em_rate_per_min":       "Rapid EMs per minute of REM sleep",
    "sem_fraction":              "SEMs / total EMs — slow-movement dominance",
    "rem_em_fraction":           "Rapid EMs / total EMs",
    "sem_mean_duration_s":       "Mean SEM duration [seconds]",
    "rem_em_mean_duration_s":    "Mean rapid EM duration [seconds]",
    "sem_mean_amp_uv":           "Mean SEM peak amplitude [uV]",
    "rem_em_mean_amp_uv":        "Mean rapid EM peak amplitude [uV]",

    # EM stage counts
    "em_count_n1":               "Total EM count during N1 sleep",
    "em_count_n2":               "Total EM count during N2 sleep",
    "em_count_n3":               "Total EM count during N3 sleep",
    "em_count_rem":              "Total EM count during REM sleep",
    "em_count_wake":             "Total EM count during Wake",

    # Phasic / Tonic
    "phasic_epoch_count":        "Number of 4s sub-epochs classified as Phasic",
    "tonic_epoch_count":         "Number of 4s sub-epochs classified as Tonic",
    "phasic_fraction":           "Phasic / (Phasic + Tonic) — key RBD biomarker",
    "tonic_fraction":            "Tonic / (Phasic + Tonic)",

    # GSSC probabilities
    "rem_mean_prob_rem":         "Mean GSSC prob(REM) during REM — high = confident staging",
    "rem_mean_prob_w":           "Mean prob(Wake) during REM — high = unstable REM",
    "rem_mean_prob_n1":          "Mean prob(N1) during REM",
    "rem_mean_prob_n2":          "Mean prob(N2) during REM",
    "rem_mean_prob_n3":          "Mean prob(N3) during REM",
    "rem_certainty":             "Fraction of REM where prob_rem > 0.5 — key RBD marker (Cesari et al.)",
    "rem_mean_prob_nrem":        "Mean combined NREM probability during REM",
    "rem_high_wake_prob_frac":   "Fraction of REM where prob_wake > 0.2 — wake intrusions",

    # REM stability
    "rem_stability_index":       "Mean prob_rem in REM epochs — higher = more stable REM",
    "rem_fragmentation_index":   "REM-to-nonREM transitions per hour — higher = more fragmented",
    "rem_w_transition_frac":     "Fraction of REM exits going directly to Wake — elevated in RBD - NaN values indicate a patient has zero REM-to-Wake transitions",
    "amount_of_rem":             "Fraction of ALL samples where prob_rem > 0.5 (Cesari definition)",

    # EEG features
    "eeg_loc__rem__delta":        "EEG (LOC) delta power during REM [µV²/Hz]",
  "eeg_loc__rem__theta":        "EEG (LOC) theta power during REM [µV²/Hz]",
  "eeg_loc__rem__alpha":        "EEG (LOC) alpha power during REM [µV²/Hz]",
  "eeg_loc__rem__beta":         "EEG (LOC) beta power during REM [µV²/Hz]",
  "eeg_loc__rem__total":        "EEG (LOC) total band power during REM [µV²/Hz]",
  "eeg_loc__rem__theta_ratio":  "EEG (LOC) theta / total power ratio during REM",
  "eeg_roc__rem__delta":        "EEG (ROC) delta power during REM [µV²/Hz]",
}

# Feature grouping for the cheat sheet
FEATURE_GROUPS = [
    ("Sleep Structure",     ["total_recording_min", "rem_duration_min", "rem_fraction",
                             "w_duration_min", "w_fraction", 
                             "n1_duration_min", "n1_fraction",
                             "n2_duration_min", "n2_fraction", 
                             "n3_duration_min", "n3_fraction",
                             "n_rem_epochs",
                             "rem_epoch_count", "rem_epoch_mean_duration_min",
                             "rem_epoch_std_duration_min", "rem_epoch_min_duration_min",
                             "rem_epoch_max_duration_min"]),

    ("EOG Amplitude (REM)", ["rem_loc_mean_abs_uv", "rem_roc_mean_abs_uv",
                             "rem_loc_std_uv", "rem_roc_std_uv",
                             "rem_loc_p95_uv", "rem_roc_p95_uv"]),

    ("REM Events - NaN values indicate a patient has zero detected REM events",          
                            ["rem_event_count", "rem_event_rate_per_min",
                             "rem_event_mean_duration_s", "rem_event_median_duration_s",
                             "rem_event_mean_loc_amp_uv", "rem_event_mean_roc_amp_uv",
                             "rem_event_mean_loc_rise_slope", "rem_event_mean_roc_rise_slope"]),

    ("EM Classification - NaN values indicate a patient has zero eye movements detected during REM",   
                            ["sem_count_rem_sleep", "rem_em_count_rem_sleep",
                             "sem_rate_per_min", "rem_em_rate_per_min",
                             "sem_fraction", "rem_em_fraction",
                             "sem_mean_duration_s", "rem_em_mean_duration_s",
                             "sem_mean_amp_uv", "rem_em_mean_amp_uv"]),

    ("EM Stage Counts",     ["em_count_n1", "em_count_n2", "em_count_n3",
                             "em_count_rem", "em_count_wake"]),

    ("Phasic / Tonic",      ["phasic_epoch_count", "tonic_epoch_count",
                             "phasic_fraction", "tonic_fraction"]),

    ("GSSC Probabilities",  ["rem_mean_prob_rem", "rem_mean_prob_w",
                             "rem_mean_prob_n1", "rem_mean_prob_n2", "rem_mean_prob_n3",
                             "rem_certainty", "rem_mean_prob_nrem", "rem_high_wake_prob_frac"]),

    ("REM Stability",       ["rem_stability_index", "rem_fragmentation_index",
                             "rem_w_transition_frac", "amount_of_rem"]),

    ("EEG Band Power (REM)", ["eeg_loc__rem__delta", "eeg_loc__rem__theta",
                              "eeg_loc__rem__alpha", "eeg_loc__rem__beta",
                              "eeg_loc__rem__total", "eeg_loc__rem__theta_ratio",
                              "eeg_roc__rem__delta", "eeg_roc__rem__theta",
                              "eeg_roc__rem__alpha", "eeg_roc__rem__beta",
                              "eeg_roc__rem__total", "eeg_roc__rem__theta_ratio",]),
]


def generate_report(
        combined_df: pd.DataFrame,
        output_path: Path,
        title: str,
) -> None:
    """
    Generate a warm-paper themed, self-contained HTML feature report.

    Parameters
    ----------
    combined_df : pd.DataFrame
        DataFrame containing all extracted features for each subject.
    output_path : Path
        File path to save the generated HTML report.
    title : str
        Title to display in the report header.
    """

    numeric_cols = [
        c for c in combined_df.columns
        if c != "subject_id" and pd.api.types.is_numeric_dtype(combined_df[c])
    ]

    n_subjects = len(combined_df)
    n_features = len(numeric_cols)
    nan_total = int(combined_df[numeric_cols].isna().sum().sum())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- Summary stats table ----
    stats = combined_df[numeric_cols].describe().T.round(4)
    stats["nan_count"] = combined_df[numeric_cols].isna().sum().values

    stats_rows = []
    for feat, row in stats.iterrows():
        badge = f' <span class="nan-badge">{int(row["nan_count"])} NaN</span>' if row["nan_count"] > 0 else ""
        stats_rows.append(
            f"<tr>"
            f"<td><strong>{feat}</strong>{badge}</td>"
            f"<td>{row.get('count', 0):.0f}</td>"
            f"<td>{row.get('mean', 0):.4f}</td>"
            f"<td>{row.get('std', 0):.4f}</td>"
            f"<td>{row.get('min', 0):.4f}</td>"
            f"<td>{row.get('25%', 0):.4f}</td>"
            f"<td>{row.get('50%', 0):.4f}</td>"
            f"<td>{row.get('75%', 0):.4f}</td>"
            f"<td>{row.get('max', 0):.4f}</td>"
            f"</tr>"
        )

    # ---- Cheat sheet ----
    cheat_rows = []
    seen = set()
    for group_name, group_cols in FEATURE_GROUPS:
        present = [c for c in group_cols if c in numeric_cols]
        if not present:
            continue
        cheat_rows.append(f'<tr class="group-header"><td colspan="2">{group_name}</td></tr>')
        for c in present:
            desc = FEATURE_DESCRIPTIONS.get(c, "—")
            cheat_rows.append(f'<tr><td><code>{c}</code></td><td>{desc}</td></tr>')
            seen.add(c)

    ungrouped = [c for c in numeric_cols if c not in seen]
    if ungrouped:
        cheat_rows.append('<tr class="group-header"><td colspan="2">Other</td></tr>')
        for c in ungrouped:
            desc = FEATURE_DESCRIPTIONS.get(c, "—")
            cheat_rows.append(f'<tr><td><code>{c}</code></td><td>{desc}</td></tr>')

    # ---- Plot cards ----
    plot_cards = []
    for col in numeric_cols:
        vals = combined_df[col].dropna().tolist()
        x_label = FEATURE_DESCRIPTIONS.get(col, col)
        hist_svg = _build_histogram_svg(vals, x_label=x_label)
        box_svg = _build_boxplot_svg(vals, x_label=x_label)
        m = np.mean(vals) if vals else float("nan")
        s = np.std(vals) if vals else float("nan")
        med = np.median(vals) if vals else float("nan")
        plot_cards.append(
            f'<div class="plot-card">'
            f'  <div class="plot-header">'
            f'    <div class="plot-title">{col}</div>'
            f'    <div class="plot-n">n = {len(vals)}</div>'
            f'  </div>'
            f'  <div class="plot-stats">mean {m:.4g} &nbsp;·&nbsp; median {med:.4g} &nbsp;·&nbsp; std {s:.4g}</div>'
            f'  <div class="svg-hist">{hist_svg}</div>'
            f'  <div class="svg-box" style="display:none">{box_svg}</div>'
            f'</div>'
        )

    # ---- Assemble HTML ----
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feature Report — {title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #FBF8F1;
    color: #1C2A3A;
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    max-width: 1240px;
    margin: 0 auto;
    padding: 28px 22px;
  }}

  /* ---- Header ---- */
  header {{
    background: #fff;
    border: 1px solid #F0E9D7;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0E7490 0%, #14B8A6 50%, #F59E0B 100%);
  }}
  .header-top {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
  }}
  header .eyebrow {{
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #0E7490;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  header h1 {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 28px;
    font-weight: 500;
    color: #1C2A3A;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
  }}
  header .subtitle {{
    color: #6B7C8E;
    font-size: 13px;
  }}
  header .date {{
    text-align: right;
    font-size: 11px;
    color: #94A3B8;
    white-space: nowrap;
    padding-top: 4px;
    line-height: 1.4;
  }}
  .meta {{
    display: flex;
    gap: 36px;
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px dashed #E5DDC8;
  }}
  .meta-item .val {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 32px;
    font-weight: 500;
    line-height: 1;
    color: #0E7490;
  }}
  .meta-item.nan .val {{ color: #C2410C; }}
  .meta-item .lbl {{
    font-size: 10px;
    color: #8899A8;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 6px;
    font-weight: 600;
  }}

  /* ---- Cards ---- */
  .card {{
    background: #fff;
    border: 1px solid #F0E9D7;
    border-radius: 14px;
    padding: 26px 30px;
    margin-bottom: 18px;
  }}
  .card h2 {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 20px;
    font-weight: 500;
    color: #1C2A3A;
    letter-spacing: -0.01em;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px dashed #E5DDC8;
  }}

  /* ---- Stats table ---- */
  .stats-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    display: block;
    overflow-x: auto;
  }}
  .stats-table th {{
    text-align: left;
    padding: 9px 12px;
    background: #FBF8F1;
    color: #6B7C8E;
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 2px solid #E5DDC8;
    position: sticky;
    top: 0;
  }}
  .stats-table td {{
    padding: 7px 12px;
    border-bottom: 1px solid #F5EDD8;
  }}
  .stats-table tr:hover td {{ background: #FDFAF3; }}
  .nan-badge {{
    background: #FEF2F2;
    color: #B91C1C;
    font-size: 10px;
    padding: 1px 7px;
    border-radius: 4px;
    font-weight: 600;
    margin-left: 6px;
    border: 1px solid #FECACA;
  }}

  /* ---- Controls ---- */
  .controls {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 18px;
    flex-wrap: wrap;
  }}
  .search-bar {{
    background: #FBF8F1;
    border: 1px solid #E5DDC8;
    color: #1C2A3A;
    padding: 9px 14px;
    font-size: 13px;
    border-radius: 8px;
    width: 300px;
    outline: none;
    font-family: inherit;
  }}
  .search-bar::placeholder {{ color: #94A3B8; }}
  .search-bar:focus {{
    border-color: #0E7490;
    background: #fff;
    box-shadow: 0 0 0 3px rgba(14, 116, 144, 0.12);
  }}
  .toggle-group {{
    display: inline-flex;
    background: #FBF8F1;
    border: 1px solid #E5DDC8;
    border-radius: 8px;
    padding: 3px;
  }}
  .toggle-btn {{
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    background: transparent;
    color: #6B7C8E;
    transition: all 0.15s;
    font-family: inherit;
  }}
  .toggle-btn.active {{
    background: #0E7490;
    color: #fff;
  }}
  .toggle-btn:hover:not(.active) {{ color: #1C2A3A; }}

  /* ---- Plot grid ---- */
  .plot-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 18px;
  }}
  .plot-card {{
    background: #fff;
    border: 1px solid #F0E9D7;
    border-radius: 12px;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
  }}
  .plot-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 4px;
  }}
  .plot-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 15px;
    font-weight: 500;
    color: #1C2A3A;
    word-break: break-all;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .plot-n {{
    font-size: 11px;
    color: #94A3B8;
    font-style: italic;
    white-space: nowrap;
    margin-left: 10px;
  }}
  .plot-stats {{
    font-size: 11px;
    color: #6B7C8E;
    margin-bottom: 10px;
    font-style: italic;
  }}
  .plot-card svg {{ width: 100%; height: auto; }}

  /* ---- Cheat sheet ---- */
  .cheat-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  .cheat-table td {{
    padding: 7px 12px;
    border-bottom: 1px solid #F5EDD8;
    vertical-align: top;
  }}
  .cheat-table code {{
    background: #F0FDFA;
    color: #0E7490;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 12px;
    font-family: 'SF Mono', Consolas, Monaco, monospace;
  }}
  .cheat-table .group-header td {{
    background: #F0FDFA;
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    color: #0E7490;
    font-size: 14px;
    letter-spacing: 0;
    padding: 12px 12px 8px;
    border-bottom: 1px solid #CFFAFE;
    border-top: 1px solid #CFFAFE;
  }}
  .cheat-description {{
    color: #6B7C8E;
    font-size: 13px;
    margin-bottom: 14px;
    font-style: italic;
  }}

  footer {{
    padding: 24px 0 8px;
    color: #94A3B8;
    font-size: 11px;
    text-align: center;
    font-style: italic;
  }}
</style>
</head>
<body>

<header>
  <div class="header-top">
    <div>
      <div class="eyebrow">RBD · Feature Extraction</div>
      <h1>{title}</h1>
      <p class="subtitle">EOG + GSSC + EEG features extracted from merged PSG recordings</p>
    </div>
    <div class="date">{timestamp}</div>
  </div>
  <div class="meta">
    <div class="meta-item"><div class="val">{n_subjects}</div><div class="lbl">Subjects</div></div>
    <div class="meta-item"><div class="val">{n_features}</div><div class="lbl">Features</div></div>
    <div class="meta-item nan"><div class="val">{nan_total}</div><div class="lbl">NaN values</div></div>
  </div>
</header>

<div class="card">
  <h2>Summary Statistics</h2>
  <table class="stats-table">
    <thead>
      <tr><th>Feature</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>50%</th><th>75%</th><th>Max</th></tr>
    </thead>
    <tbody>
      {"".join(stats_rows)}
    </tbody>
  </table>
</div>

<div class="card">
  <h2>Feature Distributions</h2>
  <div class="controls">
    <input type="text" class="search-bar" id="searchInput" placeholder="Search features..." oninput="filterPlots()">
    <div class="toggle-group">
      <button class="toggle-btn active" id="btnHist" onclick="setPlotType('hist')">Histogram</button>
      <button class="toggle-btn" id="btnBox" onclick="setPlotType('box')">Boxplot</button>
    </div>
  </div>
  <div class="plot-grid" id="plotGrid">
    {"".join(plot_cards)}
  </div>
</div>

<div class="card">
  <h2>Features</h2>
  <p class="cheat-description">What each feature means — grouped by category.</p>
  <table class="cheat-table">
    {"".join(cheat_rows)}
  </table>
</div>

<footer>Generated by feat_report.py &nbsp;·&nbsp; {timestamp}</footer>

<script>
function setPlotType(type) {{
  document.getElementById('btnHist').classList.toggle('active', type === 'hist');
  document.getElementById('btnBox').classList.toggle('active', type === 'box');
  document.querySelectorAll('.plot-card').forEach(card => {{
    card.querySelector('.svg-hist').style.display = type === 'hist' ? '' : 'none';
    card.querySelector('.svg-box').style.display = type === 'box' ? '' : 'none';
  }});
}}

function filterPlots() {{
  const q = document.getElementById('searchInput').value.toLowerCase();
  document.querySelectorAll('.plot-card').forEach(c => {{
    c.style.display = c.querySelector('.plot-title').textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
</script>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# =====================================================================
# 3)  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simple feature report — uses your existing feature functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          python simple_feat_report.py merged_csv_eog/
          python simple_feat_report.py merged_csv_eog/ --output reports/my_report.html
          python simple_feat_report.py merged_csv_eog/ --fs 128 --pattern "*.csv"
          python simple_feat_report.py merged_csv_eog/ --csv features_csv/simple.csv
        """,
    )
    parser.add_argument("merged_dir", type=str, help="Directory with merged CSV files")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling frequency [Hz] (default: 250)")
    parser.add_argument("--pattern", type=str, default="*_merged.csv", help="Glob pattern (default: *_merged.csv)")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path (default: reports/simple_report.html)")
    parser.add_argument("--csv", type=str, default=None, help="Also save feature table as CSV")

    args = parser.parse_args()
    merged_dir = Path(args.merged_dir)

    if not merged_dir.is_dir():
        print(f"Error: '{merged_dir}' is not a directory.")
        sys.exit(1)

    # ---- Collect features ----
    combined = collect_features(merged_dir, fs=args.fs, pattern=args.pattern)

    if combined.empty:
        print("Error: No features extracted.")
        sys.exit(1)

    # ---- Optionally save CSV ----
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(csv_path, index=False)
        print(f"Feature CSV saved -> {csv_path}")

    # ---- Generate HTML report ----
    output_path = Path(args.output) if args.output else Path("reports/features_report.html")
    generate_report(combined, output_path, title=merged_dir.name)
    print(f"HTML report saved -> {output_path}")
    print(f"\nDone! Open {output_path} in your browser.\n")


if __name__ == "__main__":
    main()