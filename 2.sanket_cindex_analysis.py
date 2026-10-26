#####################################################################
################## Combined dataset #################################
#####################################################################
# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from lifelines.utils import concordance_index
# from statsmodels.stats.multitest import multipletests
# from tqdm import tqdm
#
# from pacman.config import DATA_DIR, RESULTS_DIR
#
# print(7 * "=" * 7)
# print("C-indices of mitotic features for GBMLGG (combined)")
# print(7 * "=" * 7)
#
# PERM_N = 1000  # Number of permutations for significance testing
#
#
# def adjust_p_values(pval_df: pd.DataFrame, alpha: float = 0.05, method: str = "fdr_bh") -> pd.DataFrame:
#     """Benjamini–Hochberg FDR correction; returns boolean significance matrix."""
#     pvals = pval_df.values.flatten()
#     mask = ~pd.isnull(pvals)
#     if mask.sum() == 0:
#         return pval_df * False
#
#     rejected, _, _, _ = multipletests(pvals[mask], alpha=alpha, method=method)
#
#     significant = np.full(pvals.shape, False)
#     significant[mask] = rejected
#
#     sig_matrix = pd.DataFrame(
#         significant.reshape(pval_df.shape),
#         index=pval_df.index,
#         columns=pval_df.columns,
#     )
#     return sig_matrix
#
# def permutation_c_index_test(
#     group: pd.DataFrame,
#     feature: str,
#     time_col: str,
#     event_col: str,
#     n_permutations: int = 1000,
# ) -> tuple[float, float]:
#     """Permutation test: is C-index of feature vs survival different from 0.5?"""
#
#     # Convert to numpy and SQUEEZE to 1D in case of duplicate columns (n,2) -> (n,)
#     times = np.asarray(group[time_col], dtype=float)
#     events = np.asarray(group[event_col], dtype=float)
#     scores = np.asarray(group[feature], dtype=float)
#
#     if times.ndim > 1:
#         times = times[:, 0]
#     if events.ndim > 1:
#         events = events[:, 0]
#     if scores.ndim > 1:
#         scores = scores[:, 0]
#
#     scores = -scores  # higher value = higher risk
#
#     # Drop rows with NaN in any of these
#     mask = (~np.isnan(times)) & (~np.isnan(events)) & (~np.isnan(scores))
#     times, events, scores = times[mask], events[mask].astype(int), scores[mask]
#
#     # Need at least some events and non-events
#     if len(times) < 10 or events.sum() == 0 or events.sum() == len(events):
#         return np.nan, np.nan
#
#     obs_ci = concordance_index(times, scores, event_observed=events)
#
#     null_cis = []
#     rng = np.random.RandomState(0)
#     for _ in tqdm(range(n_permutations), total=n_permutations, desc=f"{feature}"):
#         shuffled = rng.permutation(scores)
#         try:
#             null_ci = concordance_index(times, shuffled, event_observed=events)
#             null_cis.append(null_ci)
#         except Exception:
#             continue
#
#     null_cis = np.asarray(null_cis, dtype=float)
#     if null_cis.size == 0:
#         return np.nan, np.nan
#
#     p_value = np.mean(np.abs(null_cis - 0.5) >= np.abs(obs_ci - 0.5))
#     return obs_ci, p_value
#
#
# def plot_cindex_heatmap_with_significance(
#     df: pd.DataFrame,
#     sig_matrix: pd.DataFrame,
#     figsize: tuple[float, float] = (6, 2.5),
#     cmap: str = "coolwarm",
# ):
#     """Heatmap of C-indices with * for FDR-significant cells."""
#     annot = df.round(2).astype(str)
#     # Only show numbers where we have a C-index
#     annot = annot.mask(df.isna(), "")
#     # Add * where significant
#     annot = annot.where(~sig_matrix, annot + "*")
#
#     fig, ax = plt.subplots(figsize=figsize)
#     sns.heatmap(
#         df,
#         annot=annot,
#         fmt="",
#         cmap=cmap,
#         linewidths=0.5,
#         linecolor="gray",
#         vmin=0.2,
#         vmax=0.8,
#         ax=ax,
#         cbar_kws={"label": "C-index"},
#     )
#     ax.set_ylabel("Cancer Type")
#     ax.set_xlabel("Feature")
#     return fig, ax
#
#
# def compute_cindex_and_significance(
#     df: pd.DataFrame,
#     features: list[str],
#     time_col: str,
#     event_col: str,
#     type_col: str,
#     alpha: float = 0.05,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     cindex_data: dict[str, dict[str, float]] = {}
#     pval_data: dict[str, dict[str, float]] = {}
#
#     for cancer_type, group in df.groupby(type_col):
#         print(f"Processing {cancer_type} (n={len(group)})")
#         cindex_row: dict[str, float] = {}
#         pval_row: dict[str, float] = {}
#
#         for feat in features:
#             if feat not in group.columns:
#                 cindex_row[feat] = np.nan
#                 pval_row[feat] = np.nan
#                 continue
#             ci, p = permutation_c_index_test(group, feat, time_col, event_col, n_permutations=PERM_N)
#             cindex_row[feat] = ci
#             pval_row[feat] = p
#
#         cindex_data[cancer_type] = cindex_row
#         pval_data[cancer_type] = pval_row
#
#     cindex_df = pd.DataFrame(cindex_data).T
#     pval_df = pd.DataFrame(pval_data).T
#     sig_df = adjust_p_values(pval_df, alpha=alpha, method="fdr_bh")
#     return cindex_df, sig_df
#
#
# def main():
#     save_dir = os.path.join(RESULTS_DIR, "survival", "cindex_gbmlgg_combined")
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Original mitotic feature set used in the paper
#     features = [
#         "HSC",
#         "mean(ND)",
#         "cv(ND)",
#         "mean(CL)",
#         "mean(HC)",
#         "AMH",
#         "AMAH",
#         "AFW",
#     ]
#     censor_at = 120  # months
#
#     # Load your processed file
#     df = pd.read_excel(os.path.join(DATA_DIR, "heatmap_cin_raw_num.xlsx"))
#
#     # Fix duplicate 'type' columns:
#     # one is IDH status, the other is cancer type (GBMLGG)
#     # From your screenshot: last two columns are 'type', 'type'
#     # pandas will name them 'type' and 'type.1'
#     cols = list(df.columns)
#     if "type.1" in cols:
#         df = df.rename(columns={"type": "idh_status", "type.1": "type"})
#     else:
#         # if you already cleaned it, assume 'type' is cancer type
#         df["idh_status"] = np.nan
#
#     # Combine everything as one GBMLGG cohort
#     df["type"] = "GBMLGG"
#
#     # Create derived feature names to match script expectations
#     df = df.rename(
#         columns={
#             "mit_hotspot_score": "HSC",
#             "mit_nodeDegrees_mean": "mean(ND)",
#             "mit_nodeDegrees_cv": "cv(ND)",
#             "mit_clusterCoff_mean": "mean(CL)",
#             "mit_cenHarmonic_mean": "mean(HC)",
#             "aty_hotspot_ratio": "AMH",
#             "aty_ahotspot_ratio": "AMAH",
#             "aty_wsi_ratio": "AFW",
#         }
#     )
#
#     for event_col in ["DSS", "PFI", "OS", "DFI"]:
#         time_col = f"{event_col}.time"
#         print(f"\n=== Endpoint: {event_col} ===")
#
#         # Keep only rows that have this endpoint + time and at least one feature value
#         subset_cols = [event_col, time_col] + features
#         df_evt = df.dropna(subset=[event_col, time_col]).copy()
#
#         if df_evt.empty:
#             print(f"No data for {event_col}, skipping.")
#             continue
#
#         # Cast event to int, time to months (capped)
#         df_evt[event_col] = df_evt[event_col].astype(int)
#         df_evt[time_col] = (df_evt[time_col] / 30.4).astype(int)
#
#         if censor_at > 0:
#             mask_censor = df_evt[time_col] > censor_at
#             df_evt.loc[mask_censor, event_col] = 0
#             df_evt.loc[mask_censor, time_col] = censor_at
#
#         c_index_table, significance_table = compute_cindex_and_significance(
#             df_evt, features, time_col, event_col, type_col="type"
#         )
#
#         num_groups = len(c_index_table.index)
#         fig_size = (7, 0.5 * num_groups + 2)
#         print(f"Figure size: {fig_size}")
#
#         fig, ax = plot_cindex_heatmap_with_significance(
#             c_index_table, significance_table, figsize=fig_size, cmap="coolwarm"
#         )
#         ax.set_title(f"C-index for {event_col} (* : permutation test's p-value < 0.05)")
#
#         fig.savefig(
#             os.path.join(save_dir, f"cindex_heatmap_{event_col}_censor{censor_at}.png"),
#             dpi=600,
#             bbox_inches="tight",
#             pad_inches=0.01,
#         )
#
#         c_index_table.to_csv(
#             os.path.join(save_dir, f"cindex_table_{event_col}_censor{censor_at}.csv"),
#             index=True,
#         )
#         significance_table.to_csv(
#             os.path.join(save_dir, f"pvalue_table_{event_col}_censor{censor_at}.csv"),
#             index=True,
#         )
#
#         plt.close(fig)
#
#
# if __name__ == "__main__":
#     main()


#####################################################################
########### separated idh-wildtype & idh-mutant dataset ###############
#####################################################################

#
# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from lifelines.utils import concordance_index
# from statsmodels.stats.multitest import multipletests
# from tqdm import tqdm
#
# from pacman.config import DATA_DIR, RESULTS_DIR
#
# print(7 * "=" * 7)
# print("C-indices of mitotic features for IDH subgroups (GBMLGG)")
# print(7 * "=" * 7)
#
# PERM_N = 1000  # permutations
#
#
# # -------------------------------------------------------------------
# # Utilities
# # -------------------------------------------------------------------
# def adjust_p_values(pval_df: pd.DataFrame, alpha: float = 0.05, method: str = "fdr_bh") -> pd.DataFrame:
#     """FDR-correct p-values and return boolean significance matrix."""
#     pvals = pval_df.values.flatten()
#     mask = ~pd.isnull(pvals)
#     if mask.sum() == 0:
#         return pval_df * False
#
#     rejected, _, _, _ = multipletests(pvals[mask], alpha=alpha, method=method)
#
#     sig = np.full(pvals.shape, False)
#     sig[mask] = rejected
#     return pd.DataFrame(sig.reshape(pval_df.shape), index=pval_df.index, columns=pval_df.columns)
#
#
# def permutation_c_index_test(
#     group: pd.DataFrame,
#     feature: str,
#     time_col: str,
#     event_col: str,
#     n_permutations: int = 1000,
# ) -> tuple[float, float]:
#     """Permutation test: is C-index of feature vs survival different from 0.5?"""
#
#     times = np.asarray(group[time_col], dtype=float)
#     events = np.asarray(group[event_col], dtype=float)
#     scores = np.asarray(group[feature], dtype=float)
#
#     # squeeze in case of duplicate columns (shape (n,2))
#     if times.ndim > 1:
#         times = times[:, 0]
#     if events.ndim > 1:
#         events = events[:, 0]
#     if scores.ndim > 1:
#         scores = scores[:, 0]
#
#     scores = -scores  # higher = riskier
#
#     mask = (~np.isnan(times)) & (~np.isnan(events)) & (~np.isnan(scores))
#     times, events, scores = times[mask], events[mask].astype(int), scores[mask]
#
#     # need both events and censored cases
#     if len(times) < 10 or events.sum() == 0 or events.sum() == len(events):
#         return np.nan, np.nan
#
#     obs_ci = concordance_index(times, scores, event_observed=events)
#
#     null_cis = []
#     rng = np.random.RandomState(0)
#     for _ in tqdm(range(n_permutations), total=n_permutations, desc=f"{feature}"):
#         shuffled = rng.permutation(scores)
#         try:
#             null_ci = concordance_index(times, shuffled, event_observed=events)
#             null_cis.append(null_ci)
#         except Exception:
#             continue
#
#     null_cis = np.asarray(null_cis, dtype=float)
#     if null_cis.size == 0:
#         return np.nan, np.nan
#
#     p_value = np.mean(np.abs(null_cis - 0.5) >= np.abs(obs_ci - 0.5))
#     return obs_ci, p_value
#
#
# def plot_cindex_heatmap_with_significance(
#     df: pd.DataFrame,
#     sig_matrix: pd.DataFrame,
#     figsize=(6, 3),
#     cmap="coolwarm",
# ):
#     """Heatmap of C-index with * where FDR-significant."""
#     annot = df.round(2).astype(str)
#     annot = annot.mask(df.isna(), "")
#     annot = annot.where(~sig_matrix, annot + "*")
#
#     fig, ax = plt.subplots(figsize=figsize)
#     sns.heatmap(
#         df,
#         annot=annot,
#         fmt="",
#         cmap=cmap,
#         linewidths=0.5,
#         linecolor="gray",
#         vmin=0.2,
#         vmax=0.8,
#         cbar_kws={"label": "C-index"},
#         ax=ax,
#     )
#     ax.set_ylabel("IDH subgroup")
#     ax.set_xlabel("Feature")
#     return fig, ax
#
#
# def compute_cindex_and_significance(
#     df: pd.DataFrame,
#     features: list[str],
#     time_col: str,
#     event_col: str,
#     type_col: str,
#     alpha: float = 0.05,
# ):
#     cindex_data = {}
#     pval_data = {}
#     diag_rows = []  # for diagnostics
#
#     for grp, group in df.groupby(type_col):
#         print(f"Processing {grp} (n={len(group)})")
#
#         # basic survival stats per group
#         events = np.asarray(group[event_col], dtype=float)
#         if events.ndim > 1:
#             events = events[:, 0]
#         n_events = int(np.nansum(events))
#         n_total = int(np.sum(~np.isnan(events)))
#         n_cens = n_total - n_events
#
#         c_row = {}
#         p_row = {}
#
#         for feat in features:
#             if feat not in group.columns:
#                 c_row[feat] = np.nan
#                 p_row[feat] = np.nan
#                 continue
#
#             # --- diagnostics for this feature in this group ---
#             col = group[feat]
#             # if there are duplicate columns, group[feat] is a DataFrame -> take first
#             if isinstance(col, pd.DataFrame):
#                 col = col.iloc[:, 0]
#
#             vals = pd.to_numeric(col, errors="coerce")
#
#             diag_rows.append(
#                 {
#                     "group": grp,
#                     "feature": feat,
#                     "n": vals.notna().sum(),
#                     "mean": vals.mean(),
#                     "std": vals.std(),
#                     "min": vals.min(),
#                     "max": vals.max(),
#                     "n_events": n_events,
#                     "n_censored": n_cens,
#                 }
#             )
#
#             # --- C-index + permutation p-value ---
#             ci, p = permutation_c_index_test(
#                 group, feat, time_col, event_col, n_permutations=PERM_N
#             )
#             c_row[feat] = ci
#             p_row[feat] = p
#
#         cindex_data[grp] = c_row
#         pval_data[grp] = p_row
#
#     cindex_df = pd.DataFrame(cindex_data).T
#     pval_df = pd.DataFrame(pval_data).T
#     sig_df = adjust_p_values(pval_df, alpha=alpha, method="fdr_bh")
#
#     diag_df = pd.DataFrame(diag_rows)
#     return cindex_df, sig_df, diag_df
#
#
# # -------------------------------------------------------------------
# # Main
# # -------------------------------------------------------------------
# def main():
#     save_dir = os.path.join(RESULTS_DIR, "survival", "cindex_idh_subgroups")
#     os.makedirs(save_dir, exist_ok=True)
#
#     features = [
#         "HSC",
#         "mean(ND)",
#         "cv(ND)",
#         "mean(CL)",
#         "mean(HC)",
#         "AMH",
#         "AMAH",
#         "AFW",
#     ]
#     censor_at = 120  # months
#
#     df = pd.read_excel(os.path.join(DATA_DIR, "heatmap_cin_raw_num.xlsx"))
#
#     # handle duplicate 'type' columns: IDH status + cancer type
#     cols = list(df.columns)
#     if "type.1" in cols:
#         df = df.rename(columns={"type": "idh_status", "type.1": "cancer_type"})
#     else:
#         df = df.rename(columns={"type": "idh_status"})
#         df["cancer_type"] = "GBMLGG"
#
#     # restrict to GBMLGG only (should already be true, but explicit)
#     df = df[df["cancer_type"] == "GBMLGG"].copy()
#
#     # rename mitotic features into the names used in the paper
#     df = df.rename(
#         columns={
#             "mit_hotspot_score": "HSC",
#             "mit_nodeDegrees_mean": "mean(ND)",
#             "mit_nodeDegrees_cv": "cv(ND)",
#             "mit_clusterCoff_mean": "mean(CL)",
#             "mit_cenHarmonic_mean": "mean(HC)",
#             "aty_hotspot_ratio": "AMH",
#             "aty_ahotspot_ratio": "AMAH",
#             "aty_wsi_ratio": "AFW",
#         }
#     )
#
#     for event_col in ["DSS", "PFI", "OS", "DFI"]:
#         time_col = f"{event_col}.time"
#         print(f"\n=== Endpoint: {event_col} ===")
#
#         df_evt = df.dropna(subset=[event_col, time_col]).copy()
#         if df_evt.empty:
#             print(f"No data for {event_col}, skipping.")
#             continue
#
#         # ensure numeric and in months
#         df_evt[event_col] = df_evt[event_col].astype(int)
#         df_evt[time_col] = (df_evt[time_col] / 30.4).astype(int)
#
#         if censor_at > 0:
#             mask_cens = df_evt[time_col] > censor_at
#             df_evt.loc[mask_cens, event_col] = 0
#             df_evt.loc[mask_cens, time_col] = censor_at
#
#         c_index_table, sig_table, diag_df = compute_cindex_and_significance(
#             df_evt,
#             features,
#             time_col,
#             event_col,
#             type_col="idh_status",  # << split by IDH
#         )
#
#         # heatmap
#         num_groups = len(c_index_table.index)
#         fig_size = (7, 0.5 * num_groups + 2)
#         fig, ax = plot_cindex_heatmap_with_significance(
#             c_index_table, sig_table, figsize=fig_size, cmap="coolwarm"
#         )
#         ax.set_title(f"C-index for {event_col} (* : permutation test's p-value < 0.05)")
#
#         fig.savefig(
#             os.path.join(save_dir, f"idh_cindex_heatmap_{event_col}_censor{censor_at}.png"),
#             dpi=600,
#             bbox_inches="tight",
#             pad_inches=0.01,
#         )
#         plt.close(fig)
#
#         # tables
#         c_index_table.to_csv(
#             os.path.join(save_dir, f"idh_cindex_table_{event_col}_censor{censor_at}.csv"),
#             index=True,
#         )
#         sig_table.to_csv(
#             os.path.join(save_dir, f"idh_pvalue_table_{event_col}_censor{censor_at}.csv"),
#             index=True,
#         )
#         diag_df.to_csv(
#             os.path.join(save_dir, f"idh_feature_diagnostics_{event_col}_censor{censor_at}.csv"),
#             index=False,
#         )
#         print("Saved C-index, p-values, and diagnostics.")
#
#
# if __name__ == "__main__":
#     main()


################################################################################
################################################################################
#--------------------- Calculate the cindex for 6 requirements
#   On top of IDH mutant row ( create 6 row plot)
# 1- IDH wildtype,
# 2- IDH -mutant,
# 3 - grade 4 astro ,
# 4 - grade 2&3 astro,
# 5 - oligo ,
# 6 - grade 2&3 astro + oligo. ------------------
################################################################################
################################################################################
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines.utils import concordance_index
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from pacman.config import DATA_DIR, RESULTS_DIR

print(7 * "=" * 7)
print("C-indices of mitotic features for requested GBMLGG subgroups")
print(7 * "=" * 7)

PERM_N = 1000  # permutations


def adjust_p_values(
    pval_df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """FDR-correct p-values and return boolean significance matrix."""
    pvals = pval_df.values.flatten()
    mask = ~pd.isnull(pvals)

    if mask.sum() == 0:
        return pd.DataFrame(
            np.full(pval_df.shape, False),
            index=pval_df.index,
            columns=pval_df.columns,
        )

    rejected, _, _, _ = multipletests(pvals[mask], alpha=alpha, method=method)

    sig = np.full(pvals.shape, False)
    sig[mask] = rejected
    return pd.DataFrame(sig.reshape(pval_df.shape), index=pval_df.index, columns=pval_df.columns)


def permutation_c_index_test(
    group: pd.DataFrame,
    feature: str,
    time_col: str,
    event_col: str,
    n_permutations: int = 1000,
) -> tuple[float, float]:
    """Permutation test: is C-index of feature vs survival different from 0.5?"""
    times = np.asarray(group[time_col], dtype=float)
    events = np.asarray(group[event_col], dtype=float)
    scores = np.asarray(group[feature], dtype=float)

    if times.ndim > 1:
        times = times[:, 0]
    if events.ndim > 1:
        events = events[:, 0]
    if scores.ndim > 1:
        scores = scores[:, 0]

    scores = -scores  # higher feature value = higher risk

    mask = (~np.isnan(times)) & (~np.isnan(events)) & (~np.isnan(scores))
    times = times[mask]
    events = events[mask].astype(int)
    scores = scores[mask]

    if len(times) < 10 or events.sum() == 0 or events.sum() == len(events):
        return np.nan, np.nan

    try:
        obs_ci = concordance_index(times, scores, event_observed=events)
    except Exception:
        return np.nan, np.nan

    null_cis = []
    rng = np.random.RandomState(0)

    for _ in tqdm(range(n_permutations), total=n_permutations, desc=f"{feature}", leave=False):
        shuffled = rng.permutation(scores)
        try:
            null_ci = concordance_index(times, shuffled, event_observed=events)
            null_cis.append(null_ci)
        except Exception:
            continue

    null_cis = np.asarray(null_cis, dtype=float)
    if null_cis.size == 0:
        return np.nan, np.nan

    p_value = np.mean(np.abs(null_cis - 0.5) >= np.abs(obs_ci - 0.5))
    return obs_ci, p_value


def plot_cindex_heatmap_with_significance(
    df: pd.DataFrame,
    sig_matrix: pd.DataFrame,
    figsize: tuple[float, float] = (8, 4),
    cmap: str = "coolwarm",
):
    """Heatmap of C-index with * where FDR-significant."""
    annot = df.round(2).astype(str)
    annot = annot.mask(df.isna(), "")
    annot = annot.where(~sig_matrix, annot + "*")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        annot=annot,
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray",
        vmin=0.2,
        vmax=0.8,
        cbar_kws={"label": "C-index"},
        ax=ax,
    )
    ax.set_ylabel("Subtype group")
    ax.set_xlabel("Feature")
    return fig, ax


def compute_cindex_and_significance(
    df: pd.DataFrame,
    features: list[str],
    time_col: str,
    event_col: str,
    group_col: str,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cindex_data: dict[str, dict[str, float]] = {}
    pval_data: dict[str, dict[str, float]] = {}
    diag_rows: list[dict] = []

    for grp_name, group in df.groupby(group_col, sort=False):
        print(f"Processing {grp_name} (n={len(group)})")

        events_arr = np.asarray(group[event_col], dtype=float)
        if events_arr.ndim > 1:
            events_arr = events_arr[:, 0]

        n_events = int(np.nansum(events_arr))
        n_total = int(np.sum(~np.isnan(events_arr)))
        n_censored = n_total - n_events

        c_row: dict[str, float] = {}
        p_row: dict[str, float] = {}

        for feat in features:
            if feat not in group.columns:
                c_row[feat] = np.nan
                p_row[feat] = np.nan
                continue

            col = group[feat]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]

            vals = pd.to_numeric(col, errors="coerce")

            diag_rows.append(
                {
                    "group": grp_name,
                    "feature": feat,
                    "n": int(vals.notna().sum()),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "min": vals.min(),
                    "max": vals.max(),
                    "n_events": n_events,
                    "n_censored": n_censored,
                }
            )

            ci, p = permutation_c_index_test(
                group,
                feat,
                time_col,
                event_col,
                n_permutations=PERM_N,
            )
            c_row[feat] = ci
            p_row[feat] = p

        cindex_data[grp_name] = c_row
        pval_data[grp_name] = p_row

    cindex_df = pd.DataFrame(cindex_data).T
    pval_df = pd.DataFrame(pval_data).T
    sig_df = adjust_p_values(pval_df, alpha=alpha, method="fdr_bh")
    diag_df = pd.DataFrame(diag_rows)

    return cindex_df, sig_df, diag_df


def normalize_string_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Strip spaces and normalize missing strings."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return df


def add_requested_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the six requested groups:
    1) IDH wildtype
    2) IDH mutant
    3) grade 4 astro
    4) grade 2&3 astro
    5) oligo
    6) grade 2&3 astro + oligo
    """
    hist_col = "histology"
    idh_col = "idh_status"
    grade_col = "grade"

    group_conditions = {
        "IDH-wildtype": df[idh_col].eq("IDH-wildtype"),
        "IDH-mutant": df[idh_col].eq("IDH-mutant"),
        "Grade 4 astro": (
            df[hist_col].eq("Astrocytoma") & df[grade_col].eq("Grade 4")
        ),
        "Grade 2&3 astro": (
            df[hist_col].eq("Astrocytoma") & df[grade_col].isin(["Grade 2", "Grade 3"])
        ),
        "Oligo": df[hist_col].eq("Oligodendroglioma"),
        "Grade 2&3 astro + oligo": (
            (df[hist_col].eq("Astrocytoma") & df[grade_col].isin(["Grade 2", "Grade 3"]))
            | df[hist_col].eq("Oligodendroglioma")
        ),
    }

    parts = []
    for group_name, mask in group_conditions.items():
        tmp = df.loc[mask].copy()
        tmp["analysis_group"] = group_name
        parts.append(tmp)

    out = pd.concat(parts, axis=0, ignore_index=True)

    desired_order = [
        "IDH-wildtype",
        "IDH-mutant",
        "Grade 4 astro",
        "Grade 2&3 astro",
        "Oligo",
        "Grade 2&3 astro + oligo",
    ]
    out["analysis_group"] = pd.Categorical(
        out["analysis_group"],
        categories=desired_order,
        ordered=True,
    )
    out = out.sort_values("analysis_group").reset_index(drop=True)
    return out


def main():
    save_dir = os.path.join(RESULTS_DIR, "survival", "cindex_requested_6groups")
    os.makedirs(save_dir, exist_ok=True)

    features = [
        "HSC",
        "mean(ND)",
        "cv(ND)",
        "mean(CL)",
        "mean(HC)",
        "AMH",
        "AMAH",
        "AFW",
    ]
    censor_at = 120  # months

    df = pd.read_excel(os.path.join(DATA_DIR, "heatmap_cin_raw_num.xlsx"))

    # Normalize group columns prepared in the revised preprocessing file
    expected_group_cols = ["histology", "idh_status", "grade"]
    missing_group_cols = [c for c in expected_group_cols if c not in df.columns]
    if missing_group_cols:
        raise ValueError(
            f"Missing required grouping columns in heatmap_cin_raw_num.xlsx: {missing_group_cols}"
        )

    df = normalize_string_columns(df, expected_group_cols)

    # Rename mitotic features to names used in the paper/script
    df = df.rename(
        columns={
            "mit_hotspot_score": "HSC",
            "mit_nodeDegrees_mean": "mean(ND)",
            "mit_nodeDegrees_cv": "cv(ND)",
            "mit_clusterCoff_mean": "mean(CL)",
            "mit_cenHarmonic_mean": "mean(HC)",
            "aty_hotspot_ratio": "AMH",
            "aty_ahotspot_ratio": "AMAH",
            "aty_wsi_ratio": "AFW",
        }
    )

    # Build the six requested groups
    df = add_requested_groups(df)

    for event_col in ["DSS", "PFI", "OS", "DFI"]:
        time_col = f"{event_col}.time"
        print(f"\n=== Endpoint: {event_col} ===")

        if event_col not in df.columns or time_col not in df.columns:
            print(f"Skipping {event_col}: missing columns.")
            continue

        df_evt = df.dropna(subset=[event_col, time_col]).copy()
        if df_evt.empty:
            print(f"No data for {event_col}, skipping.")
            continue

        df_evt[event_col] = pd.to_numeric(df_evt[event_col], errors="coerce")
        df_evt[time_col] = pd.to_numeric(df_evt[time_col], errors="coerce")
        df_evt = df_evt.dropna(subset=[event_col, time_col]).copy()

        if df_evt.empty:
            print(f"No valid numeric data for {event_col}, skipping.")
            continue

        df_evt[event_col] = df_evt[event_col].astype(int)
        df_evt[time_col] = (df_evt[time_col] / 30.4).astype(int)

        if censor_at > 0:
            mask_cens = df_evt[time_col] > censor_at
            df_evt.loc[mask_cens, event_col] = 0
            df_evt.loc[mask_cens, time_col] = censor_at

        c_index_table, sig_table, diag_df = compute_cindex_and_significance(
            df_evt,
            features,
            time_col,
            event_col,
            group_col="analysis_group",
        )

        desired_order = [
            "IDH-wildtype",
            "IDH-mutant",
            "Grade 4 astro",
            "Grade 2&3 astro",
            "Oligo",
            "Grade 2&3 astro + oligo",
        ]
        c_index_table = c_index_table.reindex(desired_order)
        sig_table = sig_table.reindex(desired_order)

        fig_size = (8, 0.55 * len(c_index_table.index) + 2)
        fig, ax = plot_cindex_heatmap_with_significance(
            c_index_table,
            sig_table,
            figsize=fig_size,
            cmap="coolwarm",
        )
        ax.set_title(
            f"C-index for {event_col} (* : permutation test FDR-adjusted p-value < 0.05)"
        )

        fig.savefig(
            os.path.join(save_dir, f"cindex_6groups_heatmap_{event_col}_censor{censor_at}.png"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.01,
        )
        plt.close(fig)

        c_index_table.to_csv(
            os.path.join(save_dir, f"cindex_6groups_table_{event_col}_censor{censor_at}.csv"),
            index=True,
        )
        sig_table.to_csv(
            os.path.join(save_dir, f"cindex_6groups_pvalue_table_{event_col}_censor{censor_at}.csv"),
            index=True,
        )
        diag_df.to_csv(
            os.path.join(save_dir, f"cindex_6groups_feature_diagnostics_{event_col}_censor{censor_at}.csv"),
            index=False,
        )

        # Save group counts for transparency
        group_counts = (
            df_evt.groupby("analysis_group", observed=False)
            .size()
            .reindex(desired_order)
            .rename("n_cases")
            .reset_index()
        )
        group_counts.to_csv(
            os.path.join(save_dir, f"cindex_6groups_counts_{event_col}_censor{censor_at}.csv"),
            index=False,
        )

        print("Saved heatmap, C-index table, p-values, diagnostics, and group counts.")


if __name__ == "__main__":
    main()
