# Filename: univariate_stats.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Performs univariate statistical testing for all features and
#              compares the resulting rankings with machine learning feature
#              importance metrics (MDI and permutation importance).
#
#              The script automatically selects between Welch's t-test and
#              Mann-Whitney U based on Shapiro-Wilk normality testing,
#              computes effect sizes and AUC values, applies FDR correction,
#              and generates summary plots and CSV tables.

# =========================================================================================================
# Imports
# =========================================================================================================
from __future__ import annotations

import argparse
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# ==========================================================================================================
# Helpers
# ==========================================================================================================

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_sd
 
 
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    try:
        U, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        return np.nan
    return 2.0 * U / (nx * ny) - 1.0
 
 
def feature_auc(values: np.ndarray, labels: np.ndarray) -> float:
    try:
        auc = roc_auc_score(labels, values)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan
 
 
def interpret_d(d: float) -> str:
    a = abs(d)
    if np.isnan(a): return "n/a"
    if a < 0.2: return "ubetydelig"
    if a < 0.5: return "lille"
    if a < 0.8: return "medium"
    return "stor"
 
 
def interpret_cliffs(delta: float) -> str:
    a = abs(delta)
    if np.isnan(a): return "n/a"
    if a < 0.147: return "ubetydelig"
    if a < 0.33: return "lille"
    if a < 0.474: return "medium"
    return "stor"
 
 
# ==========================================================================================================
# FDR-korrektion
# ==========================================================================================================
 
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    valid = ~np.isnan(pvals)
    q = np.full(n, np.nan)
    if valid.sum() == 0:
        return q
    p_valid = pvals[valid]
    order = np.argsort(p_valid)
    ranks = np.arange(1, len(p_valid) + 1)
    q_sorted = p_valid[order] * len(p_valid) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q_valid = np.empty_like(p_valid)
    q_valid[order] = q_sorted
    q[valid] = q_valid
    return q
 
 
# ==========================================================================================================
# Data loading
# ==========================================================================================================
 
# Samme exclude-konvention som prepare_data.py
EXCLUDE_NON_FEATURE = {"subject_id", "DCSM_ID", "Control", "PD(-RBD)", "PD(+RBD)",
                       "iRBD", "PLM", "group"}
 
 
def load_data(csv_path: str, label_col: str):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label-kolonne '{label_col}' findes ikke i CSV.")
    y = df[label_col]
    drop_cols = [c for c in df.columns if c in EXCLUDE_NON_FEATURE]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    return X, y
 
 
def parse_class(val: str):
    if val.isdigit():
        return int(val)
    try:
        return float(val)
    except ValueError:
        return val
 
 
def build_binary_labels(y: pd.Series, positive_class, negative_class=None) -> pd.Series:
    if negative_class is None:
        return (y == positive_class).astype(int)
    out = pd.Series(np.nan, index=y.index)
    out[y == negative_class] = 0
    out[y == positive_class] = 1
    return out
 
 
# ==========================================================================================================
# Test selection
# ==========================================================================================================
 
def choose_test(values_pos, values_neg, mode="auto") -> str:
    if mode in ("t", "u"):
        return mode
    rng = np.random.default_rng(0)
    for vals in (values_pos, values_neg):
        v = np.asarray(vals)
        v = v[~np.isnan(v)]
        if len(v) < 3:
            return "u"
        sample = v if len(v) <= 5000 else rng.choice(v, 5000, replace=False)
        try:
            _, p = stats.shapiro(sample)
            if p < 0.05:
                return "u"
        except Exception:
            return "u"
    return "t"
 
 
def run_univariate(X: pd.DataFrame, y_bin: pd.Series, test_mode: str) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        vals_pos = X.loc[y_bin == 1, col].dropna().values
        vals_neg = X.loc[y_bin == 0, col].dropna().values
 
        if len(vals_pos) < 2 or len(vals_neg) < 2:
            rows.append({"feature": col, "test": "n/a", "stat": np.nan, "p": np.nan,
                         "cohens_d": np.nan, "cliffs_delta": np.nan, "auc": np.nan,
                         "n_pos": len(vals_pos), "n_neg": len(vals_neg)})
            continue
 
        chosen = choose_test(vals_pos, vals_neg, test_mode)
        if chosen == "t":
            try:
                stat, p = stats.ttest_ind(vals_pos, vals_neg, equal_var=False, nan_policy="omit")
            except Exception:
                stat, p = np.nan, np.nan
            test_name = "welch_t"
        else:
            try:
                stat, p = stats.mannwhitneyu(vals_pos, vals_neg, alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            test_name = "mann_whitney_u"
 
        d = cohens_d(vals_pos, vals_neg)
        delta = cliffs_delta(vals_pos, vals_neg)
 
        mask = X[col].notna()
        auc = feature_auc(X.loc[mask, col].values, y_bin.loc[mask].values)
 
        rows.append({
            "feature": col,
            "test": test_name,
            "stat": stat,
            "p": p,
            "cohens_d": d,
            "abs_cohens_d": abs(d) if not np.isnan(d) else np.nan,
            "cohens_d_tolkning": interpret_d(d),
            "cliffs_delta": delta,
            "abs_cliffs_delta": abs(delta) if not np.isnan(delta) else np.nan,
            "cliffs_tolkning": interpret_cliffs(delta),
            "auc": auc,
            "n_pos": len(vals_pos),
            "n_neg": len(vals_neg),
        })
 
    df = pd.DataFrame(rows)
    df["p_bh_fdr"] = benjamini_hochberg(df["p"].values)
    df["signifikant_fdr_0.05"] = df["p_bh_fdr"] < 0.05
    return df
 
 
# ==========================================================================================================
# Merge importance
# ==========================================================================================================
 
def merge_importance(stats_df: pd.DataFrame,
                     mdi_csv: str | None,
                     perm_csv: str | None) -> pd.DataFrame:
    """
    Læser pipeline-output:
      mdi_csv:  [feature, importance]  -> renames importance -> mdi
      perm_csv: [feature, mean, std]   -> renames mean -> permutation, std -> permutation_std
    """
    merged = stats_df.copy()
 
    if mdi_csv is not None:
        mdi = pd.read_csv(mdi_csv)
        if "importance" not in mdi.columns:
            raise ValueError(f"MDI-CSV mangler 'importance'-kolonne: {mdi_csv}")
        mdi = mdi.rename(columns={"importance": "mdi"})[["feature", "mdi"]]
        merged = merged.merge(mdi, on="feature", how="left")
 
    if perm_csv is not None:
        perm = pd.read_csv(perm_csv)
        if "mean" not in perm.columns:
            raise ValueError(f"Permutation-CSV mangler 'mean'-kolonne: {perm_csv}")
        perm = perm.rename(columns={"mean": "permutation", "std": "permutation_std"})
        keep = ["feature", "permutation"] + (["permutation_std"] if "permutation_std" in perm.columns else [])
        merged = merged.merge(perm[keep], on="feature", how="left")
 
    if "mdi" in merged.columns:
        merged["rank_mdi"] = merged["mdi"].rank(ascending=False, method="min")
    if "permutation" in merged.columns:
        merged["rank_permutation"] = merged["permutation"].rank(ascending=False, method="min")
    merged["rank_abs_d"]     = merged["abs_cohens_d"].rank(ascending=False, method="min")
    merged["rank_abs_delta"] = merged["abs_cliffs_delta"].rank(ascending=False, method="min")
    merged["rank_auc"]       = merged["auc"].rank(ascending=False, method="min")
    merged["rank_p"]         = merged["p"].rank(ascending=True, method="min")
    return merged
 
 
def rank_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    if len(rank_cols) < 2:
        return pd.DataFrame()
    sub = merged[rank_cols].dropna()
    if sub.empty:
        return pd.DataFrame()
    return sub.corr(method="spearman")
 
 
# ==========================================================================================================
# Plots 
# ==========================================================================================================
 
def plot_top_features(merged: pd.DataFrame, metric: str, title: str, out_path: Path, top_n=20):
    if metric not in merged.columns:
        return
    df = merged.dropna(subset=[metric]).copy()
    if df.empty:
        return
    df = df.reindex(df[metric].abs().sort_values(ascending=False).index).head(top_n)
 
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in df[metric]]
    ax.barh(df["feature"], df[metric], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_rank_scatter(merged: pd.DataFrame, x_col: str, y_col: str,
                       out_path: Path, label_top=10):
    if x_col not in merged.columns or y_col not in merged.columns:
        return
    df = merged.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return
 
    rho, p = stats.spearmanr(df[x_col], df[y_col])
 
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=40)
 
    df["combined"] = df[x_col] + df[y_col]
    top = df.nsmallest(label_top, "combined")
    for _, row in top.iterrows():
        ax.annotate(row["feature"], (row[x_col], row[y_col]),
                    fontsize=8, alpha=0.8, xytext=(3, 3), textcoords="offset points")
 
    lim = max(df[x_col].max(), df[y_col].max()) + 1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel(x_col + "  (1 = vigtigst/mest signifikant)")
    ax.set_ylabel(y_col + "  (1 = vigtigst/mest signifikant)")
    ax.set_title(f"{x_col}  vs  {y_col}\nSpearman ρ = {rho:.3f}  (p = {p:.3g})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_volcano(merged: pd.DataFrame, out_path: Path):
    df = merged.dropna(subset=["cohens_d", "p"]).copy()
    if df.empty:
        return
    df["neglog10p"] = -np.log10(df["p"].clip(lower=1e-300))
 
    fig, ax = plt.subplots(figsize=(8, 6))
    sig = df["p_bh_fdr"] < 0.05
    ax.scatter(df.loc[~sig, "cohens_d"], df.loc[~sig, "neglog10p"],
               alpha=0.5, s=30, color="gray", label="Ikke signifikant (FDR)")
    ax.scatter(df.loc[sig, "cohens_d"], df.loc[sig, "neglog10p"],
               alpha=0.8, s=40, color="tab:red", label="Signifikant (FDR<0.05)")
 
    top = df.reindex(df["neglog10p"].sort_values(ascending=False).index).head(10)
    for _, row in top.iterrows():
        ax.annotate(row["feature"], (row["cohens_d"], row["neglog10p"]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
 
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Volcano plot: effektstørrelse vs signifikans")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
# ==========================================================================================================
# CLI 
# ==========================================================================================================
 
def main():
    parser = argparse.ArgumentParser(description="Univariat statistik + sammenligning med ML importance.")
    parser.add_argument("--csv", required=True, help="Feature-CSV (samme som I bruger til ML).")
    parser.add_argument("--label-col", required=True,
                        help="Navn på label-kolonne (fx 'group' for jeres pipeline).")
    parser.add_argument("--positive-class", required=True,
                        help="Værdi der udgør positiv klasse (fx 'iRBD', 'PD(-RBD)', 1).")
    parser.add_argument("--negative-class", default=None,
                        help="Værdi der udgør negativ klasse (fx 'Control'). "
                             "Hvis ikke angivet: alle andre rækker bliver negative (one-vs-rest).")
    parser.add_argument("--mdi-csv", default=None,
                        help="Sti til ..._mdi.csv fra pipelinen.")
    parser.add_argument("--permutation-csv", default=None,
                        help="Sti til ..._permutation.csv fra pipelinen.")
    parser.add_argument("--test", choices=["auto", "t", "u"], default="auto")
    parser.add_argument("--out-dir", default="stats_output")
    args = parser.parse_args()
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"Indlæser {args.csv} ...")
    X, y = load_data(args.csv, args.label_col)
 
    pos_val = parse_class(args.positive_class)
    neg_val = parse_class(args.negative_class) if args.negative_class is not None else None
 
    y_bin = build_binary_labels(y, pos_val, neg_val)
 
    valid = y_bin.notna()
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"  Dropper {n_dropped} rækker uden gyldig label.")
    X = X.loc[valid].reset_index(drop=True)
    y_bin = y_bin.loc[valid].astype(int).reset_index(drop=True)
 
    label_str = f"'{pos_val}' vs " + (f"'{neg_val}'" if neg_val is not None else "rest")
    print(f"Sammenligner: {label_str}")
    print(f"  Positiv: {(y_bin == 1).sum()} samples")
    print(f"  Negativ: {(y_bin == 0).sum()} samples")
    print(f"  Features: {X.shape[1]}")
 
    print(f"\nKører univariate tests (mode='{args.test}') ...")
    stats_df = run_univariate(X, y_bin, args.test)
 
    n_t = (stats_df["test"] == "welch_t").sum()
    n_u = (stats_df["test"] == "mann_whitney_u").sum()
    print(f"  Welch t-test: {n_t}  |  Mann-Whitney U: {n_u}")
    print(f"  Signifikante efter FDR-korrektion (q<0.05): {stats_df['signifikant_fdr_0.05'].sum()}")
 
    if args.mdi_csv:
        print(f"  Indlæser MDI:         {args.mdi_csv}")
    if args.permutation_csv:
        print(f"  Indlæser permutation: {args.permutation_csv}")
 
    merged = merge_importance(stats_df, args.mdi_csv, args.permutation_csv)
 
    main_csv = out_dir / "univariate_stats.csv"
    merged.to_csv(main_csv, index=False)
    print(f"\n  Hovedtabel: {main_csv}")
 
    rank_corr = rank_correlations(merged)
    if not rank_corr.empty:
        corr_csv = out_dir / "rank_correlations.csv"
        rank_corr.to_csv(corr_csv)
        print(f"  Rank-korrelationer: {corr_csv}")
        print("\nSpearman-korrelationer mellem rangeringer:")
        print(rank_corr.round(3).to_string())
 
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    print(f"\nLaver plots i {plot_dir} ...")
 
    plot_top_features(merged, "abs_cohens_d", "Top features — |Cohen's d|",
                      plot_dir / "top_cohens_d.png")
    plot_top_features(merged, "abs_cliffs_delta", "Top features — |Cliff's delta|",
                      plot_dir / "top_cliffs_delta.png")
    plot_top_features(merged, "auc", "Top features — AUC (univariat)",
                      plot_dir / "top_auc.png")
 
    if "mdi" in merged.columns:
        plot_top_features(merged, "mdi", "Top features — MDI",
                          plot_dir / "top_mdi.png")
    if "permutation" in merged.columns:
        plot_top_features(merged, "permutation", "Top features — Permutation importance",
                          plot_dir / "top_permutation.png")
 
    plot_volcano(merged, plot_dir / "volcano.png")
 
    if "rank_mdi" in merged.columns:
        plot_rank_scatter(merged, "rank_mdi", "rank_abs_d",
                          plot_dir / "scatter_mdi_vs_cohens_d.png")
        plot_rank_scatter(merged, "rank_mdi", "rank_abs_delta",
                          plot_dir / "scatter_mdi_vs_cliffs.png")
        plot_rank_scatter(merged, "rank_mdi", "rank_auc",
                          plot_dir / "scatter_mdi_vs_auc.png")
    if "rank_permutation" in merged.columns:
        plot_rank_scatter(merged, "rank_permutation", "rank_abs_d",
                          plot_dir / "scatter_permutation_vs_cohens_d.png")
        plot_rank_scatter(merged, "rank_permutation", "rank_abs_delta",
                          plot_dir / "scatter_permutation_vs_cliffs.png")
        plot_rank_scatter(merged, "rank_permutation", "rank_auc",
                          plot_dir / "scatter_permutation_vs_auc.png")
    if "rank_mdi" in merged.columns and "rank_permutation" in merged.columns:
        plot_rank_scatter(merged, "rank_mdi", "rank_permutation",
                          plot_dir / "scatter_mdi_vs_permutation.png")
 
    print("\nFærdig.")
    print(f"\nAnbefalet workflow til rapporten:")
    print("  1. Kig på univariate_stats.csv — sortér efter p_bh_fdr eller abs_cohens_d.")
    print("  2. Kig på rank_correlations.csv — høj korrelation = metoderne er enige.")
    print("  3. Brug scatter-plots til at finde features med 'uenighed':")
    print("     - Høj univariat rank, lav importance rank = redundant (korreleret med andre).")
    print("     - Lav univariat rank, høj importance rank = interaktionseffekt — godt fund!")
    print("  4. Volcano-plot er fin til rapporten: viser både effekt og signifikans samlet.")
 
 
if __name__ == "__main__":
    main()