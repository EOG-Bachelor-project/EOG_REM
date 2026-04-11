# Filename: feat_report.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Collects EOG and GSSC features from all merged CSVs in a directory,
#              joins them into a single feature table, saves as CSV, and generates
#              an HTML report with distribution plots per feature.


# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path

from features.eog_feats import extract_features
from features.gssc_feats import extract_gssc_features

# =====================================================================
# Constants
# =====================================================================
FEATURES_DIR = Path("features_csv")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Collection
# =====================================================================

def collect_all_features(
        merged_dir:  str | Path,
        fs:          float = 250.0,
        pattern:     str   = "*_merged.csv",
        output_stem: str   = "all_features",
        ) -> pd.DataFrame:
    """
    Run EOG and GSSC feature extraction on every merged CSV in a directory,
    join the two feature tables on subject_id, save as CSV, and generate
    an HTML report with distribution plots.

    Parameters
    ----------
    merged_dir : str | Path
        Directory containing merged CSV files (output of merge_all).
    fs : float
        Sampling frequency in [Hz]. Default is **250.0 Hz**.
    pattern : str
        Glob pattern to match merged CSVs. Default is ``'*_merged.csv'``.
    output_stem : str
        Stem for output filenames. Saves to:
        - ``features_csv/{output_stem}.csv``
        - ``reports/{output_stem}.html``

    Returns
    -------
    pd.DataFrame
        Combined feature table with one row per subject.
    """
    merged_dir = Path(merged_dir)
    files = sorted(merged_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {merged_dir}"
        )

    print(f"\nFound {len(files)} merged CSV(s) in {merged_dir}")
    print(f"Output stem: {output_stem}")

    # ---- 1) Extract EOG features ----
    print("\n" + "=" * 60)
    print("Extracting EOG features...")
    eog_rows = []
    for f in files:
        try:
            row = extract_features(f, fs=fs)
            eog_rows.append(row)
        except Exception as e:
            print(f"  [SKIP EOG] {f.name} — {e}")

    eog_df = pd.DataFrame(eog_rows)
    print(f"\nEOG features: {eog_df.shape[0]} subjects | {eog_df.shape[1]-1} features")

    # ---- 2) Extract GSSC features ----
    print("\n" + "=" * 60)
    print("Extracting GSSC features...")
    gssc_rows = []
    for f in files:
        try:
            row = extract_gssc_features(f, fs=fs)
            gssc_rows.append(row)
        except Exception as e:
            print(f"  [SKIP GSSC] {f.name} — {e}")

    gssc_df = pd.DataFrame(gssc_rows)
    print(f"\nGSSC features: {gssc_df.shape[0]} subjects | {gssc_df.shape[1]-1} features")

    # ---- 3) Join on subject_id ----
    print("\n" + "=" * 60)
    print("Joining feature tables...")
    combined_df = pd.merge(eog_df, gssc_df, on="subject_id", how="outer", suffixes=("_eog", "_gssc"))
    print(f"Combined: {combined_df.shape[0]} subjects | {combined_df.shape[1]-1} features")

    # ---- 4) Save CSV ----
    csv_path = FEATURES_DIR / f"{output_stem}.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\nFeature table saved → {csv_path}")

    # ---- 5) Generate HTML report ----
    html_path = REPORTS_DIR / f"{output_stem}.html"
    _generate_html_report(combined_df, html_path, output_stem)
    print(f"HTML report saved  → {html_path}")

    return combined_df


# =====================================================================
# HTML report
# =====================================================================

def _generate_html_report(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Generate a self-contained HTML report with summary stats and distribution plots."""

    numeric_cols = [
        c for c in df.columns
        if c != "subject_id" and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Build summary stats table
    stats = df[numeric_cols].describe().T.round(4)
    stats["nan_count"] = df[numeric_cols].isna().sum()
    stats_html = stats.to_html(classes="stats-table", border=0)

    # Serialize data for JS plots
    plot_data = {}
    for col in numeric_cols:
        vals = df[col].dropna().tolist()
        plot_data[col] = vals

    plot_data_json = json.dumps(plot_data)
    cols_json      = json.dumps(numeric_cols)

    n_subjects = len(df)
    n_features = len(numeric_cols)
    nan_total  = int(df[numeric_cols].isna().sum().sum())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>

  /* =====================================================================
     DESIGN TOKENS
     Change colours here to restyle the entire report.
     --bg       : page background
     --surface  : card / table background
     --border   : subtle divider lines
     --accent   : primary highlight colour (headings, bars, box borders)
     --accent2  : secondary highlight (median line in boxplot)
     --text     : main body text
     --muted    : labels, captions, axis ticks
     --green    : available for positive indicators
     --red      : NaN badge background
     ===================================================================== */
  :root {{
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --border:   #2a2d3a;
    --accent:   #4f8ef7;
    --accent2:  #f7a24f;
    --text:     #e2e8f0;
    --muted:    #8892a4;
    --green:    #4ade80;
    --red:      #f87171;
  }}

  /* =====================================================================
     RESET & BASE
     font-family : change the typeface for the whole page here
     font-size   : base size that all other sizes scale from
     ===================================================================== */
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.6;
  }}

  /* =====================================================================
     HEADER
     The top banner containing the report title, subtitle, and summary metrics.
     padding : controls whitespace around the header content
     h1      : report title — change font-size, color, letter-spacing here
     p       : subtitle line below the title
     ===================================================================== */
  header {{
    padding: 40px 48px 24px;
    border-bottom: 1px solid var(--border);
  }}

  header h1 {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--accent);
    text-transform: uppercase;
  }}

  header p {{
    color: var(--muted);
    margin-top: 6px;
    font-size: 12px;
  }}

  /* =====================================================================
     HEADER METRICS ROW
     The three big numbers (Subjects / Features / NaN values) in the header.
     .meta      : flex container — change gap to spread numbers out
     .val       : the big number — change font-size to make it larger/smaller
     .lbl       : the small label below the number
     ===================================================================== */
  .meta {{
    display: flex;
    gap: 32px;
    margin-top: 16px;
  }}

  .meta-item {{
    display: flex;
    flex-direction: column;
  }}

  .meta-item .val {{
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }}

  .meta-item .lbl {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }}

  /* =====================================================================
     SECTION WRAPPERS
     Each major block (stats table, distributions) sits inside a <section>.
     padding : inner whitespace of each block
     h2      : section heading style
     ===================================================================== */
  section {{
    padding: 32px 48px;
    border-bottom: 1px solid var(--border);
  }}

  section h2 {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
  }}

  /* =====================================================================
     SUMMARY STATISTICS TABLE
     The describe() output table at the top of the report.
     th : column header cells — change background, font-size, padding here
     td : data cells
     tr:hover : row highlight on mouse-over
     ===================================================================== */
  .stats-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}

  .stats-table th {{
    text-align: left;
    padding: 8px 12px;
    background: var(--surface);
    color: var(--muted);
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 11px;
    border-bottom: 1px solid var(--border);
  }}

  .stats-table td {{
    padding: 7px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}

  .stats-table tr:hover td {{
    background: var(--surface);
  }}

  /* =====================================================================
     PLOT CONTROLS BAR
     The search box and plot-type dropdown above the plot grid.
     gap          : spacing between controls
     label        : control labels (Search / Plot type)
     select       : the dropdown element
     select:focus : focused state border colour
     ===================================================================== */
  .plot-controls {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }}

  .plot-controls label {{
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  .plot-controls select {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    font-family: inherit;
    font-size: 12px;
    cursor: pointer;
    outline: none;
  }}

  .plot-controls select:focus {{
    border-color: var(--accent);
  }}

  /* =====================================================================
     PLOT GRID
     The responsive grid that holds all the feature plot cards.
     grid-template-columns : controls how many cards per row.
       minmax(320px, 1fr)  → min card width 320px, expands to fill row.
       Change 320px to make cards wider/narrower.
     gap : spacing between cards
     ===================================================================== */
  .plot-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }}

  /* =====================================================================
     PLOT CARD
     Individual card wrapping each feature's plot.
     background : card background colour (defaults to --surface)
     border     : card border
     padding    : inner whitespace
     h3         : feature name label at the top of each card
     ===================================================================== */
  .plot-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 16px;
  }}

  .plot-card h3 {{
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  /* =====================================================================
     CANVAS
     The drawing surface inside each plot card.
     height is set dynamically in JS (120px histogram / 80px boxplot).
     ===================================================================== */
  canvas {{
    width: 100% !important;
    display: block;
  }}

  /* =====================================================================
     PLOT FOOTER STATS
     The n= / mean= / std= line below each plot.
     font-size : size of the stat labels
     color     : text colour of the stat labels
     ===================================================================== */
  .plot-stats {{
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    font-size: 11px;
    color: var(--muted);
  }}

  /* =====================================================================
     NaN BADGE
     Red pill shown on cards where some subjects have missing values.
     background   : badge background colour
     border-radius: roundness — set to 0 for square, 99px for pill
     ===================================================================== */
  .nan-badge {{
    background: var(--red);
    color: #fff;
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 2px;
    font-weight: 700;
  }}

  /* =====================================================================
     SEARCH BAR
     The text input for filtering features by name.
     width        : how wide the input is
     padding      : inner spacing
     border:focus : border colour when the input is focused
     ===================================================================== */
  .search-bar {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    font-family: inherit;
    font-size: 12px;
    width: 280px;
    outline: none;
  }}

  .search-bar:focus {{ border-color: var(--accent); }}

  /* =====================================================================
     FOOTER
     Bottom bar with the generated-by line.
     padding : whitespace around footer text
     color   : footer text colour
     ===================================================================== */
  footer {{
    padding: 24px 48px;
    color: var(--muted);
    font-size: 11px;
  }}

</style>
</head>
<body>

<!-- =====================================================================
     HEADER BLOCK
     Title, subtitle, and the three summary metric numbers.
     To change the title text: edit the <h1> content.
     To add more metrics: copy a .meta-item div and update the values.
     ===================================================================== -->
<header>
  <h1>Feature Report — {title}</h1>
  <p>EOG + GSSC features extracted from merged PSG recordings</p>
  <div class="meta">
    <div class="meta-item"><span class="val">{n_subjects}</span><span class="lbl">Subjects</span></div>
    <div class="meta-item"><span class="val">{n_features}</span><span class="lbl">Features</span></div>
    <div class="meta-item"><span class="val">{nan_total}</span><span class="lbl">NaN values</span></div>
  </div>
</header>

<!-- =====================================================================
     SUMMARY STATISTICS TABLE
     Pandas describe() output rendered as an HTML table.
     The table is injected by Python as {stats_html}.
     To change table styling: edit .stats-table in the CSS above.
     ===================================================================== -->
<section>
  <h2>Summary Statistics</h2>
  {stats_html}
</section>

<!-- =====================================================================
     FEATURE DISTRIBUTIONS SECTION
     Contains the search bar, plot-type dropdown, and the plot grid.
     The grid itself (#plotGrid) is populated dynamically by JS below.
     ===================================================================== -->
<section>
  <h2>Feature Distributions</h2>

  <!-- Controls bar: search input + plot type selector -->
  <div class="plot-controls">
    <label>Search</label>
    <input class="search-bar" type="text" id="search" placeholder="Filter features..." oninput="filterPlots()">
    <label>Plot type</label>
    <select id="plotType" onchange="renderAll()">
      <option value="histogram">Histogram</option>
      <option value="boxplot">Box plot</option>
    </select>
  </div>

  <!-- Plot cards are injected here by renderAll() in JS -->
  <div class="plot-grid" id="plotGrid"></div>
</section>

<!-- =====================================================================
     FOOTER
     ===================================================================== -->
<footer>
  Generated by feature_report.py — EOG_REM project
</footer>

<script>
/* =====================================================================
   JS DATA
   Feature values injected by Python. Do not edit these manually.
   DATA : dict of {{ feature_name: [val1, val2, ...] }}
   COLS : list of feature names in display order
   ===================================================================== */
const DATA = {plot_data_json};
const COLS = {cols_json};

/* =====================================================================
   COLOUR CONSTANTS
   These mirror the CSS variables above.
   Change here if you want different plot colours without touching CSS.
   ACCENT  : bar fill / box border colour
   ACCENT2 : median line colour in boxplot
   MUTED   : axis tick and label colour
   ===================================================================== */
const ACCENT  = '#4f8ef7';
const ACCENT2 = '#f7a24f';
const MUTED   = '#8892a4';
const BG      = '#1a1d27';
const TEXT    = '#e2e8f0';

/* =====================================================================
   HISTOGRAM (Chart.js bar chart)
   Called for each feature card when plot type = histogram.
   bins        : number of bars — currently auto-calculated from sqrt(n)
   backgroundColor : bar fill colour — change ACCENT to any hex
   borderColor     : bar outline colour
   grid color      : '#2a2d3a' — change to match your --border token
   ===================================================================== */
function buildHistogramChart(vals, ctx, canvas) {{
  if (vals.length === 0) return null;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const bins = Math.min(20, Math.max(5, Math.ceil(Math.sqrt(vals.length))));
  const step = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  vals.forEach(v => {{
    let i = Math.floor((v - min) / step);
    if (i >= bins) i = bins - 1;
    counts[i]++;
  }});
  const labels = counts.map((_, i) => (min + i * step).toFixed(2));

  return new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ data: counts, backgroundColor: ACCENT + 'bb', borderColor: ACCENT, borderWidth: 1 }}]
    }},
    options: {{
      responsive: false,
      animation: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: MUTED, font: {{ size: 9 }}, maxRotation: 0 }}, grid: {{ color: '#2a2d3a' }} }},
        y: {{ ticks: {{ color: MUTED, font: {{ size: 9 }} }}, grid: {{ color: '#2a2d3a' }} }}
      }}
    }}
  }});
}}

/* =====================================================================
   BOXPLOT (drawn manually on HTML canvas)
   Called for each feature card when plot type = boxplot.
   bh          : box height in pixels — change to make the box taller
   ACCENT      : box fill and border colour
   ACCENT2     : median line colour
   '#f87171'   : outlier dot colour — change to any hex
   Font        : axis label font — currently '10px Courier New'
   ===================================================================== */
function buildBoxplot(vals, ctx) {{
  if (vals.length === 0) return;
  const sorted = [...vals].sort((a,b) => a-b);
  const q1  = sorted[Math.floor(sorted.length * 0.25)];
  const med = sorted[Math.floor(sorted.length * 0.50)];
  const q3  = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1;
  const lo  = Math.max(sorted[0],  q1 - 1.5*iqr);
  const hi  = Math.min(sorted[sorted.length-1], q3 + 1.5*iqr);
  const outliers = sorted.filter(v => v < lo || v > hi);

  const canvas = ctx.canvas;
  const W = canvas.width, H = canvas.height;
  const c = ctx;
  c.clearRect(0,0,W,H);

  const allVals = [lo, q1, med, q3, hi, ...outliers];
  const vmin = Math.min(...allVals), vmax = Math.max(...allVals);
  const scale = v => 20 + (v - vmin) / (vmax - vmin + 1e-9) * (W - 40);
  const midY = H / 2;

  // Whiskers
  c.strokeStyle = MUTED; c.lineWidth = 1.5;
  c.beginPath(); c.moveTo(scale(lo), midY); c.lineTo(scale(q1), midY); c.stroke();
  c.beginPath(); c.moveTo(scale(q3), midY); c.lineTo(scale(hi), midY); c.stroke();
  [lo, hi].forEach(v => {{
    c.beginPath(); c.moveTo(scale(v), midY-8); c.lineTo(scale(v), midY+8); c.stroke();
  }});

  // Box body
  c.fillStyle = ACCENT + '33';
  c.strokeStyle = ACCENT; c.lineWidth = 2;
  const bh = 24;
  c.fillRect(scale(q1), midY-bh/2, scale(q3)-scale(q1), bh);
  c.strokeRect(scale(q1), midY-bh/2, scale(q3)-scale(q1), bh);

  // Median line
  c.strokeStyle = ACCENT2; c.lineWidth = 2.5;
  c.beginPath(); c.moveTo(scale(med), midY-bh/2); c.lineTo(scale(med), midY+bh/2); c.stroke();

  // Outlier dots
  c.fillStyle = '#f87171';
  outliers.forEach(v => {{
    c.beginPath(); c.arc(scale(v), midY, 3, 0, Math.PI*2); c.fill();
  }});

  // Q1 / Median / Q3 labels
  c.fillStyle = MUTED; c.font = '10px Courier New'; c.textAlign = 'center';
  c.fillText(`Q1: ${{q1.toFixed(3)}}`, scale(q1), midY + bh/2 + 14);
  c.fillText(`Med: ${{med.toFixed(3)}}`, scale(med), midY - bh/2 - 6);
  c.fillText(`Q3: ${{q3.toFixed(3)}}`, scale(q3), midY + bh/2 + 14);
}}

/* =====================================================================
   CARD BUILDER
   Creates the HTML for one feature card (title + canvas + stats footer).
   canvas height : 120px for histogram, set to 80px for boxplot in renderAll
   nan-badge     : only shown if nanCount > 0
   plot-stats    : n= / mean= / std= line below the chart
   ===================================================================== */
const chartInstances = {{}};

function renderCard(col) {{
  const vals     = DATA[col] || [];
  const nanCount = {n_subjects} - vals.length;

  const card = document.createElement('div');
  card.className = 'plot-card';
  card.dataset.col = col;

  const nanBadge = nanCount > 0 ? `<span class="nan-badge">${{nanCount}} NaN</span>` : '';
  card.innerHTML = `
    <h3>${{col}} ${{nanBadge}}</h3>
    <canvas id="canvas_${{col}}" height="120"></canvas>
    <div class="plot-stats">
      <span>n=${{vals.length}}</span>
      <span>mean=${{vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(3) : 'N/A'}}</span>
      <span>std=${{vals.length > 1 ? Math.sqrt(vals.map(v=>(v-(vals.reduce((a,b)=>a+b,0)/vals.length))**2).reduce((a,b)=>a+b,0)/(vals.length-1)).toFixed(3) : 'N/A'}}</span>
    </div>`;

  return card;
}}

/* =====================================================================
   RENDER ALL
   Clears the grid and redraws all visible feature cards.
   Called on page load, search input, and plot type change.
   ===================================================================== */
function renderAll() {{
  const grid     = document.getElementById('plotGrid');
  const search   = document.getElementById('search').value.toLowerCase();
  const plotType = document.getElementById('plotType').value;

  Object.values(chartInstances).forEach(c => {{ try {{ c.destroy(); }} catch(e) {{}} }});
  Object.keys(chartInstances).forEach(k => delete chartInstances[k]);
  grid.innerHTML = '';

  const visibleCols = COLS.filter(c => c.toLowerCase().includes(search));
  visibleCols.forEach(col => grid.appendChild(renderCard(col)));

  requestAnimationFrame(() => {{
    visibleCols.forEach(col => {{
      const vals   = DATA[col] || [];
      const canvas = document.getElementById(`canvas_${{col}}`);
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (plotType === 'histogram') {{
        const chart = buildHistogramChart(vals, ctx, canvas);
        if (chart) chartInstances[col] = chart;
      }} else {{
        canvas.height = 80;
        buildBoxplot(vals, ctx);
      }}
    }});
  }});
}}

/* =====================================================================
   SEARCH FILTER
   Triggered by the search input oninput event.
   ===================================================================== */
function filterPlots() {{ renderAll(); }}

/* =====================================================================
   CHART.JS LOADER
   Loads Chart.js from CDN then triggers the initial render.
   To pin a different Chart.js version: change the version in the URL.
   ===================================================================== */
const script = document.createElement('script');
script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js';
script.onload = () => renderAll();
document.head.appendChild(script);
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_report.py <merged_dir> [fs] [output_stem]")
        print("")
        print("Examples:")
        print("  # Default merged CSVs")
        print("  python feature_report.py merged_csv_eog/ 250.0 default_features")
        print("")
        print("  # Umaer merged CSVs")
        print("  python feature_report.py merged_csv_eog/ 250.0 umaer_features --pattern '*_merged_Umaer.csv'")
        sys.exit(1)

    merged_dir   = Path(sys.argv[1])
    fs           = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
    output_stem  = sys.argv[3] if len(sys.argv) > 3 else "all_features"
    pattern      = "*_merged_Umaer.csv" if "--pattern" in sys.argv and "Umaer" in sys.argv else "*_merged.csv"
    # Exclude Umaer files from default run
    if "Umaer" not in output_stem:
        pattern = "*_merged.csv"

    df = collect_all_features(
        merged_dir  = merged_dir,
        fs          = fs,
        pattern     = pattern,
        output_stem = output_stem,
    )

    print(f"\nDone. {df.shape[0]} subjects × {df.shape[1]-1} features.")