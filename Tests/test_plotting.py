from __future__ import annotations

import mne
import torch
import gssc.networks
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from preprocessing.plot import plot_eog_epochs
torch.serialization.add_safe_globals([gssc.networks.ResSleep])
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def plot_eog_gssc(edf_path, csv_path, timestamp_col='timestamp', stage_col='stage', epoch_sec=30):
    """
    Plot EOG signals (LOC, ROC) using NeuroKit2 with GSSC staging hypnogram.

    Parameters
    ----------
    edf_path      : path to .edf file
    csv_path      : path to GSSC output CSV
    timestamp_col : name of timestamp column in CSV
    stage_col     : name of stage label column in CSV
    epoch_sec     : epoch length in seconds (default 30)
    """

    # ── 1. Load EDF and extract LOC/ROC ──────────────────────────────────────
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs  = int(raw.info['sfreq'])

    eog_channels = [ch for ch in raw.ch_names
                    if any(k in ch.upper() for k in ['LOC', 'ROC', 'EOGH', 'EOGV', 'E1', 'E2'])]
    print(f"Found EOG channels: {eog_channels}")

    eog_data, times = raw[eog_channels]  # (n_ch, n_samples)

    # ── 2. NeuroKit2 EOG cleaning ─────────────────────────────────────────────
    eog_cleaned = []
    for sig in eog_data:
        cleaned = nk.eog_clean(sig, sampling_rate=fs, method='neurokit')
        eog_cleaned.append(cleaned)

    # ── 3. Load GSSC CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    t0 = df[timestamp_col].iloc[0]
    df['t_sec'] = (df[timestamp_col] - t0).dt.total_seconds()

    staging  = df[stage_col].astype(str).tolist()
    t_epochs = df['t_sec'].values

    # ── 4. Staging style ──────────────────────────────────────────────────────
    stage_order   = ['W', 'N1', 'N2', 'N3', 'R']
    stage_colors  = {'W': '#e74c3c', 'N1': '#f39c12',
                     'N2': '#2ecc71', 'N3': '#2980b9',
                     'R':  '#9b59b6', 'U':  '#95a5a6'}
    stage_y       = {s: i for i, s in enumerate(stage_order)}
    stage_numeric = np.array([stage_y.get(s, -1) for s in staging])

    # ── 5. Build figure ───────────────────────────────────────────────────────
    n_ch  = len(eog_channels)
    fig   = plt.figure(figsize=(18, 3 * (n_ch + 1)))
    gs    = GridSpec(n_ch + 1, 1, figure=fig, hspace=0.15,
                     height_ratios=[1.3] + [1] * n_ch)

    axes = [fig.add_subplot(gs[i]) for i in range(n_ch + 1)]
    for ax in axes[:-1]:
        ax.sharex(axes[-1])

    # ── 6. Hypnogram ──────────────────────────────────────────────────────────
    ax_hyp = axes[0]
    for stage, t_ep in zip(staging, t_epochs):
        color = stage_colors.get(stage, '#95a5a6')
        ax_hyp.barh(0, epoch_sec, left=t_ep,
                    height=0.6, color=color, alpha=0.85, edgecolor='none')

    ax_hyp.step(t_epochs, stage_numeric, where='post',
                color='k', linewidth=1.3, zorder=5)
    ax_hyp.set_yticks(range(len(stage_order)))
    ax_hyp.set_yticklabels(stage_order, fontsize=9)
    ax_hyp.invert_yaxis()
    ax_hyp.set_ylabel("Stage", fontsize=10)
    ax_hyp.set_title("EOG Signal with GSSC Sleep Staging", fontsize=13, fontweight='bold')
    ax_hyp.tick_params(labelbottom=False)

    patches = [mpatches.Patch(color=stage_colors[s], label=s) for s in stage_order]
    ax_hyp.legend(handles=patches, ncol=5, fontsize=8,
                  loc='upper right', framealpha=0.7)

    # ── 7. EOG channels ───────────────────────────────────────────────────────
    for i, (ax, ch_name, sig) in enumerate(zip(axes[1:], eog_channels, eog_cleaned)):

        # stage background shading
        for stage, t_ep in zip(staging, t_epochs):
            ax.axvspan(t_ep, t_ep + epoch_sec,
                       alpha=0.07, color=stage_colors.get(stage, '#95a5a6'), linewidth=0)

        # plot cleaned EOG via neurokit
        nk.signal_plot(sig, sampling_rate=fs, ax=ax, show=False)
        ax.get_lines()[0].set(color='#2c3e50', linewidth=0.5)  # restyle nk line
        ax.set_ylabel(ch_name, fontsize=10)
        ax.set_title('')
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        if i < n_ch - 1:
            ax.tick_params(labelbottom=False)

    # x-axis in hours
    axes[-1].set_xlabel("Time", fontsize=11)
    formatter = plt.FuncFormatter(lambda x, _: f"{x/3600:.1f}h")
    axes[-1].xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    return fig, axes


# ── Usage ─────────────────────────────────────────────────────────────────────

EDF_FILE = Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_1_a\contiguous.edf")
GSSC_FILE = Path("C:/Users/AKLO0022/EOG_REM/gssc_csv/DCSM_1_a_gssc.csv")

fig, axes = plot_eog_gssc(
    edf_path= EDF_FILE,
    csv_path=GSSC_FILE,
    timestamp_col="time_sec",    # adjust to your CSV column name
    stage_col="stage",           # adjust to your CSV column name
)
plt.savefig("eog_gssc_plot.png", dpi=150, bbox_inches='tight')
plt.show()