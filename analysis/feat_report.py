# Filename: feat_report.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Collects EOG and GSSC features from all merged CSVs and generates
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


# =====================================================================
# 1)  COLLECT FEATURES 
# =====================================================================

def collect_features(
    merged_dir: str | Path,
    fs: float = 250.0,
    pattern: str = "*_merged.csv",
) -> pd.DataFrame:
    """
    Run extract_features() and extract_gssc_features() on every merged CSV
    in a directory and join them into a single DataFrame.

    """
    merged_dir = Path(merged_dir)
    files = sorted(merged_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {merged_dir}")

    print(f"\nFound {len(files)} merged CSV(s) in {merged_dir}\n")

    # ---- EOG features ----
    eog_rows = []
    for f in files:
        try:
            row = extract_features(f, fs=fs)
            eog_rows.append(row)
        except Exception as e:
            print(f"  [SKIP EOG] {f.name} - {e}")

    eog_df = pd.DataFrame(eog_rows)
    print(f"\nEOG features: {eog_df.shape[0]} subjects | {eog_df.shape[1] - 1} features")

    # ---- GSSC features ----
    gssc_rows = []
    for f in files:
        try:
            row = extract_gssc_features(f, fs=fs)
            gssc_rows.append(row)
        except Exception as e:
            print(f"  [SKIP GSSC] {f.name} — {e}")

    gssc_df = pd.DataFrame(gssc_rows)
    print(f"GSSC features: {gssc_df.shape[0]} subjects | {gssc_df.shape[1] - 1} features")

    # ---- Join on subject_id ----
    combined = pd.merge(eog_df, gssc_df, on="subject_id", how="outer", suffixes=("_eog", "_gssc"))
    print(f"Combined: {combined.shape[0]} subjects | {combined.shape[1] - 1} features\n")

    return combined


# =====================================================================
# 2)  HTML REPORT  
# =====================================================================
def _svg_no_data(width: int = 360, height: int = 150) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<text x="50%" y="50%" text-anchor="middle" fill="#aaa" font-size="12">'
        f'Not enough data</text></svg>'
        )

def _build_histogram_svg(
        values: list[float], 
        width: int = 320, 
        height: int = 160
        ) -> str:
    """Build a simple SVG histogram for a list of values."""
    if not values or len(values) < 2:
        return (
            f'<svg width="{width}" height="{height}">'
            f'<text x="50%" y="50%" text-anchor="middle" fill="#999" font-size="12">'
            f'Not enough data</text></svg>'
        )

    n_bins = min(20, max(5, len(values) // 3))
    counts, edges = np.histogram(values, bins=n_bins)
    max_count = max(counts) if max(counts) > 0 else 1

    pad_l, pad_r, pad_t, pad_b = 40, 12, 12, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    bar_w = plot_w / n_bins
    gap = max(1, bar_w * 0.1)

    parts = []

    # --- Y-axis gridlines + labels ---
    for i in range(4):
        frac = i/3
        y = pad_t + plot_h * (1 - frac)
        val = int(max_count * frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="#edf2f7" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{y + 3:.1f}" font-size="9" fill="#a0aec0" '
            f'text-anchor="end">{val}</text>'
            )


    # --- Bars ---
    for i, c in enumerate(counts):
        bh = (c / max_count) * plot_h if max_count > 0 else 0
        x = pad_l + i * bar_w + gap / 2
        y = pad_t + plot_h - bh
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - gap:.1f}" '
            f'height="{bh:.1f}" fill="#4f8ef7" rx="2" opacity="0.85"/>'
            )
    
    # --- X-axis labels ---
    vmin, vmax = edges[0], edges[-1]
    vmid = (vmin + vmax) / 2
    ly = height - 4

    parts.append(f'<text x="{pad_l}" y="{ly}" font-size="9" fill="#a0aec0" text-anchor="start">{vmin:.3g}</text>')
    parts.append(f'<text x="{pad_l + plot_w / 2}" y="{ly}" font-size="9" fill="#a0aec0" text-anchor="middle">{vmid:.3g}</text>')
    parts.append(f'<text x="{width - pad_r}" y="{ly}" font-size="9" fill="#a0aec0" text-anchor="end">{vmax:.3g}</text>')
 
    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

def _build_boxplot_svg(
        values: list[float], 
        width: int = 360, 
        height: int = 150
        ) -> str:
    """Build a clean horizontal SVG boxplot with individual data points."""
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
  
    pad_l, pad_r, pad_t, pad_b = 12, 12, 20, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def xpos(v):
      return pad_l + (v - vmin) / (vmax - vmin) * plot_w
 
    box_top = pad_t + plot_h * 0.25
    box_bot = pad_t + plot_h * 0.75
    box_mid = pad_t + plot_h * 0.5
    box_h = box_bot - box_top

    parts = []

    # Whisker line
    parts.append(
        f'<line x1="{xpos(whisker_lo):.1f}" y1="{box_mid:.1f}" '
        f'x2="{xpos(whisker_hi):.1f}" y2="{box_mid:.1f}" stroke="#718096" stroke-width="1.5"/>'
        )
    
    # Whisker caps
    for wv in [whisker_lo, whisker_hi]:
      x = xpos(wv)
      parts.append(
          f'<line x1="{x:.1f}" y1="{box_top + 6:.1f}" x2="{x:.1f}" y2="{box_bot - 6:.1f}" '
          f'stroke="#718096" stroke-width="1.5"/>'
          )
 
    # IQR box
    bx = xpos(q1)
    bw = xpos(q3) - xpos(q1)
    parts.append(
        f'<rect x="{bx:.1f}" y="{box_top:.1f}" width="{bw:.1f}" height="{box_h:.1f}" '
        f'fill="#ebf4ff" stroke="#4f8ef7" stroke-width="1.5" rx="3"/>'
        )
 
    # Median line
    mx = xpos(med)
    parts.append(
        f'<line x1="{mx:.1f}" y1="{box_top:.1f}" x2="{mx:.1f}" y2="{box_bot:.1f}" '
        f'stroke="#e53e3e" stroke-width="2"/>'
        )
 
    # Individual data points (jittered)
    rng = np.random.RandomState(42)
    jitter = rng.uniform(-plot_h * 0.12, plot_h * 0.12, size=len(arr))
    for i, v in enumerate(arr):
      cx = xpos(v)
      cy = box_mid + jitter[i]
      parts.append(
          f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="#4f8ef7" opacity="0.4"/>'
          )
 
    # Outliers
    for v in outliers:
      cx = xpos(v)
      parts.append(
          f'<circle cx="{cx:.1f}" cy="{box_mid:.1f}" r="3" fill="none" stroke="#e53e3e" stroke-width="1.2"/>'
          )
 
    # X-axis labels
    ly = height - 4
    vmid_label = (vmin + vmax) / 2
    parts.append(f'<text x="{pad_l}" y="{ly}" font-size="9" fill="#a0aec0" text-anchor="start">{vmin:.3g}</text>')
    parts.append(f'<text x="{pad_l + plot_w / 2}" y="{ly}" font-size="9" fill="#a0aec0" text-anchor="middle">{vmid_label:.3g}</text>')
    parts.append(f'<text x="{width - pad_r}" y="{ly}" font-size="9" fill="#a0aec0" text-anchor="end">{vmax:.3g}</text>')

    # Stat labels at top
    parts.append(f'<text x="{xpos(q1):.1f}" y="{pad_t - 4}" font-size="8" fill="#718096" text-anchor="middle">Q1={q1:.2g}</text>')
    parts.append(f'<text x="{xpos(med):.1f}" y="{pad_t - 4}" font-size="8" fill="#e53e3e" text-anchor="middle">med={med:.2g}</text>')
    parts.append(f'<text x="{xpos(q3):.1f}" y="{pad_t - 4}" font-size="8" fill="#718096" text-anchor="middle">Q3={q3:.2g}</text>')

    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

# ---- Feature descriptions ----
FEATURE_DESCRIPTIONS = {
    # Sleep structure
    "total_recording_min":       "Total recording duration [minutes]",
    "rem_duration_min":          "Total REM sleep duration [minutes]",
    "rem_fraction":              "Fraction of recording spent in REM",
    "w_duration_min":            "Wake duration [minutes]",
    "w_fraction":                "Fraction of recording spent awake",
    "n1_duration_min":           "N1 (light sleep) duration [minutes]",
    "n1_fraction":               "Fraction of recording in N1",
    "n2_duration_min":           "N2 (medium sleep) duration [minutes]",
    "n2_fraction":               "Fraction of recording in N2",
    "n3_duration_min":           "N3 (deep sleep) duration [minutes]",
    "n3_fraction":               "Fraction of recording in N3",
    "n_rem_epochs":              "Number of distinct REM episodes across the night",

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
    "rem_w_transition_frac":     "Fraction of REM exits going directly to Wake — elevated in RBD",
    "amount_of_rem":             "Fraction of ALL samples where prob_rem > 0.5 (Cesari definition)",
}

# Feature grouping for the cheat sheet
FEATURE_GROUPS = [
    ("Sleep Structure",     ["total_recording_min", "rem_duration_min", "rem_fraction",
                             "w_duration_min", "w_fraction", "n1_duration_min", "n1_fraction",
                             "n2_duration_min", "n2_fraction", "n3_duration_min", "n3_fraction",
                             "n_rem_epochs"]),

    ("EOG Amplitude (REM)", ["rem_loc_mean_abs_uv", "rem_roc_mean_abs_uv",
                             "rem_loc_std_uv", "rem_roc_std_uv",
                             "rem_loc_p95_uv", "rem_roc_p95_uv"]),

    ("REM Events",          ["rem_event_count", "rem_event_rate_per_min",
                             "rem_event_mean_duration_s", "rem_event_median_duration_s",
                             "rem_event_mean_loc_amp_uv", "rem_event_mean_roc_amp_uv",
                             "rem_event_mean_loc_rise_slope", "rem_event_mean_roc_rise_slope"]),

    ("EM Classification",   ["sem_count_rem_sleep", "rem_em_count_rem_sleep",
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
]


def generate_report(
        combined_df: pd.DataFrame, 
        output_path: Path, 
        title: str
        ) -> None:
    """Generate a clean, self-contained HTML report."""

    numeric_cols = [
        c for c in combined_df.columns
        if c != "subject_id" and pd.api.types.is_numeric_dtype(combined_df[c])
    ]

    n_subjects = len(combined_df)
    n_features = len(numeric_cols)
    nan_total = int(combined_df[numeric_cols].isna().sum().sum())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- Build summary stats table ----
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

    # ---- Build cheat sheet table ----
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

    # Catch any features not in the predefined groups
    ungrouped = [c for c in numeric_cols if c not in seen]
    if ungrouped:
        cheat_rows.append('<tr class="group-header"><td colspan="2">Other</td></tr>')
        for c in ungrouped:
            desc = FEATURE_DESCRIPTIONS.get(c, "—")
            cheat_rows.append(f'<tr><td><code>{c}</code></td><td>{desc}</td></tr>')

    # ---- Build plot cards ----
    plot_cards = []
    for col in numeric_cols:
        vals = combined_df[col].dropna().tolist()
        hist_svg = _build_histogram_svg(vals)
        box_svg = _build_boxplot_svg(vals)
        m = np.mean(vals) if vals else float("nan")
        s = np.std(vals) if vals else float("nan")
        med = np.median(vals) if vals else float("nan")
        plot_cards.append(
            f'<div class="plot-card">'
            f'  <div class="plot-title">{col}</div>'
            f'  <div class="plot-stats">mean: {m:.4g} &nbsp;|&nbsp; median: {med:.4g} &nbsp;|&nbsp; std: {s:.4g} &nbsp;|&nbsp; n: {len(vals)}</div>'
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
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #f8f9fb; 
    color: #2d3748;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 14px; 
    line-height: 1.6;
    max-width: 1200px;
    margin: 0 auto; 
    padding: 20px;
  }}
  header {{
    background: #fff; 
    border: 1px solid #e2e8f0; 
    border-radius: 12px;
    padding: 32px 36px 24px; 
    margin-bottom: 20px;
  }}
  header h1 {{ font-size: 22px; font-weight: 700; color: #1a365d; margin-bottom: 4px; }}
  header .subtitle {{ color: #718096; font-size: 13px; margin-bottom: 16px; }}
  .meta {{ display: flex; gap: 32px; }}
  .meta-item {{ display: flex; flex-direction: column; }}
  .meta-item .val {{ font-size: 28px; font-weight: 700; color: #2d3748; }}
  .meta-item .lbl {{ font-size: 11px; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }}
  .card {{
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 28px 32px; margin-bottom: 20px;
  }}
  .card h2 {{
    font-size: 15px; 
    font-weight: 700; 
    color: #2b6cb0; 
    text-transform: uppercase;
    letter-spacing: 0.06em; 
    margin-bottom: 16px; 
    padding-bottom: 8px;
    border-bottom: 2px solid #ebf4ff;
  }}
  .stats-table {{ width: 100%; border-collapse: collapse; font-size: 12px; overflow-x: auto; display: block; }}
  .stats-table th {{
    text-align: left; 
    padding: 8px 10px; 
    background: #f7fafc; 
    color: #718096;
    font-weight: 600; 
    font-size: 11px; 
    text-transform: uppercase; 
    letter-spacing: 0.05em;
    border-bottom: 2px solid #e2e8f0; 
    position: sticky; top: 0;
  }}
  .stats-table td {{ padding: 6px 10px; border-bottom: 1px solid #edf2f7; }}
  .stats-table tr:hover td {{ background: #f7fafc; }}
  .nan-badge {{
    background: #fed7d7; 
    color: #c53030; 
    font-size: 10px;
    padding: 1px 6px; 
    border-radius: 3px; 
    font-weight: 600; 
    margin-left: 6px;
  }}

  /* ---- PLot grid ---- */
  .plot-grid {{ 
    display: grid; 
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
    gap: 12px; 
  }}

  .plot-card {{ 
    background: #fff; 
    border: 1px solid #e2e8f0;
    border-radius: 10px; 
    padding: 14px 16px;
    display: flex; 
    flex-direction: column;
  }}
  .plot-title {{ 
    font-size: 11px; 
    font-weight: 700; 
    color: #2d3748; 
    margin-bottom: 2px; 
    word-break: break-all;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis; 
  }}
  .plot-stats {{ font-size: 11px; color: #a0aec0; margin-bottom: 8px; }}
  .plot-card svg {{ width: 100%; height: auto; }}

  .cheat-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .cheat-table td {{ padding: 6px 10px; border-bottom: 1px solid #edf2f7; vertical-align: top; }}
  .cheat-table code {{ background: #edf2f7; padding: 1px 5px; border-radius: 3px; font-size: 12px; color: #2b6cb0; }}
  .cheat-table .group-header td {{
    background: #ebf8ff; font-weight: 700; color: #2c5282; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.06em; padding: 10px 10px 6px;
    border-bottom: 2px solid #bee3f8;
  }}

  /* ---- Controls bar ---- */
  .controls {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .search-bar {{
    background: #fff; border: 1px solid #e2e8f0; color: #2d3748;
    padding: 8px 14px; font-size: 13px; border-radius: 8px; width: 300px;
    outline: none; margin-bottom: 16px;
  }}
  .search-bar:focus {{ border-color: #4299e1; box-shadow: 0 0 0 3px rgba(66,153,225,0.15); }}

  .toggle-btn{{
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    cursor: pointer;
    background: #fff;
    color: #718096;
    transition: all 0.15s;
  }}
  toggle-btn.active {{
    background: #4f8ef7;
    color: #fff;
    border-color: #4f8ef7;
  }}
  .toggle-btn:hover:not(.active) {{ background: #f7fafc; }}

  footer {{ padding: 20px 0; color: #a0aec0; font-size: 11px; text-align: center; }}
</style>
</head>
<body>

<header>
  <h1>Feature Report — {title}</h1>
  <p class="subtitle">EOG + GSSC features extracted from merged PSG recordings &nbsp;·&nbsp; {timestamp}</p>
  <div class="meta">
    <div class="meta-item"><span class="val">{n_subjects}</span><span class="lbl">Subjects</span></div>
    <div class="meta-item"><span class="val">{n_features}</span><span class="lbl">Features</span></div>
    <div class="meta-item"><span class="val">{nan_total}</span><span class="lbl">NaN values</span></div>
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
  <div class ="controls">
    <input type="text" class="search-bar" id="searchInput" placeholder="Search features..." oninput="filterPlots()">
    <button class="toggle-btn active" id="btnHist" onclick="setPlotType('hist')">"Histogram"</button>
    <button class="toggle-btn" id="btnBox" onclick="setPlotType('box')">"Boxplot"</button>
  </div>
  <div class="plot-grid" id="plotGrid">
    {"".join(plot_cards)}
  </div>
</div>

<div class="card">
  <h2>Features</h2>
  <p style="color:#718096; font-size:13px; margin-bottom:14px;">What each feature means - grouped by category.</p>
  <table class="cheat-table">
    {"".join(cheat_rows)}
  </table>
</div>

<footer>Generated by simple_feat_report.py &nbsp;·&nbsp; {timestamp}</footer>

<script>
let currentPlot = 'hist';

function setPlotType(type) {{
  currentPlot = type;
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