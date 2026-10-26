# import os
# import warnings
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from lifelines import KaplanMeierFitter
# from lifelines.statistics import logrank_test
# from matplotlib.offsetbox import AnchoredText
# from tqdm import tqdm
#
# from pacman.config import DATA_DIR, RESULTS_DIR
# from pacman.survival.utils import add_at_risk_counts, survival_stratification_analysis
#
# warnings.filterwarnings("ignore")
#
# # ----------------- CONFIG -----------------
# CV_FOLDS = 5
# PERM_REPS = 30  # increase to 1000 for final runs if needed
# CENSOR_AT = 120
# FONT_SIZE = 12
# FIG_SIZE = 5
#
# SPLITS_ROOT = "./splits"
# BRAIN_FILE = os.path.join(DATA_DIR, "heatmap_cin_raw_num.xlsx")
# RESULTS_ROOT = os.path.join(RESULTS_DIR, "survival", "kmplots_cv_brain_all_features")
# os.makedirs(RESULTS_ROOT, exist_ok=True)
#
# ENDPOINTS = ["OS", "PFI", "DSS", "DFI"]
#
# ID_COLS = ["slide_id", "bcr_patient_barcode", "case_id"]
# GROUP_COLS = ["histology", "idh_status", "grade"]
# SURVIVAL_COLS = ["OS", "OS.time", "DSS", "DSS.time", "DFI", "DFI.time", "PFI", "PFI.time"]
# EXCLUDE_COLS = set(ID_COLS + GROUP_COLS + SURVIVAL_COLS)
# # ------------------------------------------
#
#
# def smart_text_location(ax):
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     lines = ax.get_lines()
#     x_mid = (xlim[1] - xlim[0]) * 0.5
#     y_threshold = (ylim[1] - ylim[0]) * 0.5
#
#     lower_left_count = 0
#     upper_right_count = 0
#     for line in lines:
#         xdata, ydata = line.get_data()
#         lower_left_count += np.sum((xdata < x_mid) & (ydata < y_threshold))
#         upper_right_count += np.sum((xdata > x_mid) & (ydata > y_threshold))
#
#     return "lower left" if lower_left_count < upper_right_count else "upper right"
#
#
# def sanitize_name(text: str) -> str:
#     return (
#         str(text)
#         .replace("/", "_")
#         .replace("\\", "_")
#         .replace(" ", "_")
#         .replace("(", "")
#         .replace(")", "")
#         .replace("&", "and")
#     )
#
#
# def get_all_numeric_features(df: pd.DataFrame) -> list[str]:
#     feature_cols = []
#     for col in df.columns:
#         if col in EXCLUDE_COLS:
#             continue
#         if pd.api.types.is_numeric_dtype(df[col]):
#             feature_cols.append(col)
#     return feature_cols
#
#
# def build_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
#     cohorts = {
#         "GBMLGG": df.copy(),
#         "IDH-mutant": df[df["idh_status"] == "IDH-mutant"].copy(),
#         "IDH-wildtype": df[df["idh_status"] == "IDH-wildtype"].copy(),
#         "Grade 4 astro": df[
#             (df["histology"] == "Astrocytoma") & (df["grade"] == "Grade 4")
#         ].copy(),
#         "Grade 2&3 astro": df[
#             (df["histology"] == "Astrocytoma") & (df["grade"].isin(["Grade 2", "Grade 3"]))
#         ].copy(),
#         "Oligo": df[df["histology"] == "Oligodendroglioma"].copy(),
#         "Grade 2&3 astro + oligo": df[
#             ((df["histology"] == "Astrocytoma") & (df["grade"].isin(["Grade 2", "Grade 3"])))
#             | (df["histology"] == "Oligodendroglioma")
#         ].copy(),
#     }
#     return cohorts
#
#
# def cv_helper(
#     discov_df,
#     feat_name,
#     event_col,
#     time_col,
#     split_folder,
#     km_plot=False,
#     add_counts=False,
#     shuffle_data=False,
#     cutoff_mode="median",
#     cutoff_point=-1,
# ):
#     EE = discov_df[event_col].to_numpy()
#     rng = np.random.RandomState(0)
#
#     c_indices = []
#     p_values = []
#
#     T_lower_test = T_upper_test = None
#     E_lower_test = E_upper_test = None
#     test_hazard_ratios = None
#
#     for run in range(CV_FOLDS):
#         split_path = os.path.join(split_folder, f"splits_{run}.csv")
#         if not os.path.exists(split_path):
#             continue
#
#         split_df = pd.read_csv(split_path)
#
#         train_set = discov_df[
#             discov_df["bcr_patient_barcode"].isin(split_df["train"])
#         ].copy()
#         test_set = discov_df[
#             discov_df["bcr_patient_barcode"].isin(split_df["val"])
#         ].copy()
#
#         train_set.reset_index(drop=True, inplace=True)
#         test_set.reset_index(drop=True, inplace=True)
#
#         if len(train_set) == 0 or len(test_set) == 0:
#             continue
#
#         if shuffle_data:
#             idx_tr = rng.permutation(len(train_set))
#             train_set[event_col] = train_set[event_col].iloc[idx_tr].to_numpy()
#             train_set[time_col] = train_set[time_col].iloc[idx_tr].to_numpy()
#
#             idx_te = rng.permutation(len(test_set))
#             test_set[event_col] = test_set[event_col].iloc[idx_te].to_numpy()
#             test_set[time_col] = test_set[time_col].iloc[idx_te].to_numpy()
#
#         output = survival_stratification_analysis(
#             train_set,
#             test_set,
#             feat_name,
#             time_col,
#             event_col,
#             CENSOR_AT,
#             cutoff_mode,
#             cutoff_point,
#         )
#
#         if output == -1:
#             continue
#
#         (
#             _T_lower_test,
#             _T_upper_test,
#             _E_lower_test,
#             _E_upper_test,
#             cindex_test,
#             pvalue_test,
#             temp_hr,
#         ) = output
#
#         if T_lower_test is None:
#             T_lower_test, T_upper_test = _T_lower_test, _T_upper_test
#             E_lower_test, E_upper_test = _E_lower_test, _E_upper_test
#             test_hazard_ratios = temp_hr
#         else:
#             T_lower_test = pd.concat([T_lower_test, _T_lower_test], ignore_index=True)
#             T_upper_test = pd.concat([T_upper_test, _T_upper_test], ignore_index=True)
#             E_lower_test = pd.concat([E_lower_test, _E_lower_test], ignore_index=True)
#             E_upper_test = pd.concat([E_upper_test, _E_upper_test], ignore_index=True)
#             test_hazard_ratios = pd.concat(
#                 [test_hazard_ratios, temp_hr], axis=1, ignore_index=True
#             )
#
#         c_indices.append(cindex_test)
#         p_values.append(pvalue_test)
#
#     if T_lower_test is None or T_upper_test is None:
#         return None, None, None, None
#
#     df_to_save = test_hazard_ratios.T
#     df_to_save["c_index"] = c_indices
#     df_to_save["p_value"] = p_values
#
#     logrank_results = logrank_test(
#         T_lower_test, T_upper_test, E_lower_test, E_upper_test
#     )
#
#     if not km_plot:
#         return df_to_save, logrank_results, None, None
#
#     fig = plt.figure(figsize=(FIG_SIZE - 1.8, FIG_SIZE - 2))
#     ax = fig.add_subplot(111)
#     ax.set_xlabel("", fontsize=FONT_SIZE)
#     ax.set_ylabel("", fontsize=FONT_SIZE)
#     ax.tick_params(axis="x", labelsize=FONT_SIZE)
#     ax.tick_params(axis="y", labelsize=FONT_SIZE)
#
#     km_high = KaplanMeierFitter()
#     km_low = KaplanMeierFitter()
#
#     upper_label = f"{feat_name}-High"
#     lower_label = f"{feat_name}-Low"
#
#     ax = km_high.fit(
#         T_upper_test, event_observed=E_upper_test, label=upper_label
#     ).plot_survival_function(
#         ax=ax,
#         show_censors=True,
#         censor_styles={"ms": 4},
#         color="r",
#         ci_show=False,
#         xlabel="Months",
#         ylabel=None,
#     )
#     ax = km_low.fit(
#         T_lower_test, event_observed=E_lower_test, label=lower_label
#     ).plot_survival_function(
#         ax=ax,
#         show_censors=True,
#         censor_styles={"ms": 4},
#         color="b",
#         ci_show=False,
#         xlabel="Months",
#         ylabel=None,
#     )
#     if ax.get_legend() is not None:
#         ax.get_legend().remove()
#
#     fig_counts = None
#     if add_counts:
#         fig_counts = plt.figure(figsize=(FIG_SIZE - 1, FIG_SIZE - 2))
#         ax_c = fig_counts.add_subplot(111)
#         ax_c.set_xlabel("", fontsize=FONT_SIZE)
#         ax_c.set_ylabel("", fontsize=FONT_SIZE)
#         ax_c.tick_params(axis="x", labelsize=FONT_SIZE)
#         ax_c.tick_params(axis="y", labelsize=FONT_SIZE)
#
#         ax_c = km_high.fit(
#             T_upper_test, event_observed=E_upper_test, label=upper_label
#         ).plot_survival_function(
#             ax=ax_c,
#             show_censors=True,
#             censor_styles={"ms": 5},
#             color="r",
#             ci_show=False,
#             xlabel="Months",
#             ylabel=None,
#         )
#         ax_c = km_low.fit(
#             T_lower_test, event_observed=E_lower_test, label=lower_label
#         ).plot_survival_function(
#             ax=ax_c,
#             show_censors=True,
#             censor_styles={"ms": 5},
#             color="b",
#             ci_show=False,
#             xlabel="Months",
#             ylabel=None,
#         )
#
#         add_at_risk_counts(
#             km_low,
#             km_high,
#             rows_to_show=["At risk"],
#             xticks=[0, 25, 50, 75, 100],
#             ypos=-0.6,
#             colors=["blue", "red"],
#             ax=ax_c,
#             fig=fig_counts,
#             fontsize=int(FONT_SIZE * 1),
#         )
#         fig_counts.subplots_adjust(bottom=0.4, left=0.2)
#         if ax_c.get_legend() is not None:
#             ax_c.get_legend().remove()
#
#     return df_to_save, logrank_results, fig, fig_counts
#
#
# def run_for_feature_and_cohort(label, df, feature_name, event_type):
#     time_col = f"{event_type}.time"
#     event_col = event_type
#
#     required = [event_col, time_col, feature_name, "bcr_patient_barcode"]
#     df = df.dropna(subset=required).copy()
#
#     if len(df) < 10:
#         print(f"Skipping {label} | {feature_name} | {event_type}: too few rows")
#         return
#
#     df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
#     df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
#     df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")
#     df = df.dropna(subset=[event_col, time_col, feature_name]).copy()
#
#     if len(df) < 10:
#         print(f"Skipping {label} | {feature_name} | {event_type}: too few valid rows")
#         return
#
#     df[event_col] = df[event_col].astype(int)
#     df[time_col] = (df[time_col] / 30.4).astype(int)
#     df.loc[df[time_col] > CENSOR_AT, event_col] = 0
#     df.loc[df[time_col] > CENSOR_AT, time_col] = CENSOR_AT
#
#     if df[event_col].sum() == 0:
#         print(f"Skipping {label} | {feature_name} | {event_type}: no events")
#         return
#
#     split_folder = os.path.join(SPLITS_ROOT, event_type, "GBMLGG")
#
#     out = cv_helper(
#         df,
#         feature_name,
#         event_col,
#         time_col,
#         split_folder,
#         km_plot=True,
#         add_counts=True,
#         cutoff_mode="median",
#     )
#
#     if out[0] is None:
#         print(f"Skipping {label} | {feature_name} | {event_type}: CV failed")
#         return
#
#     df_cv, logrank_res, km_fig, km_fig_counts = out
#
#     ref_stat = logrank_res.test_statistic
#     num_greater = 0
#     num_valid = 0
#
#     for _ in range(PERM_REPS):
#         try:
#             perm_out = cv_helper(
#                 df,
#                 feature_name,
#                 event_col,
#                 time_col,
#                 split_folder,
#                 km_plot=False,
#                 add_counts=False,
#                 shuffle_data=True,
#                 cutoff_mode="median",
#             )
#             if perm_out[0] is None:
#                 continue
#             _, logrep, _, _ = perm_out
#         except Exception:
#             continue
#
#         if logrep.test_statistic > ref_stat:
#             num_greater += 1
#         num_valid += 1
#
#     p_corr = num_greater / max(num_valid, 1)
#     avg_ci = df_cv["c_index"].mean()
#
#     if p_corr < 0.0001:
#         p_txt = "p < 0.0001"
#     else:
#         p_txt = f"p = {p_corr:.4f}"
#
#     for fig in [km_fig, km_fig_counts]:
#         if fig is None:
#             continue
#         ax = fig.axes[0]
#         loc = smart_text_location(ax)
#         ax.add_artist(
#             AnchoredText(p_txt, loc=loc, frameon=False, prop=dict(size=FONT_SIZE))
#         )
#         ax.set_ylabel(f"{event_type} Probability")
#         ax.set_ylim(0, 1)
#         ax.set_xticks([0, 25, 50, 75, 100])
#         ax.set_xlim(0, CENSOR_AT + 1)
#         ax.set_title(label, fontsize=FONT_SIZE + 2)
#         ax.spines[["right", "top"]].set_visible(False)
#
#     endpoint_dir = os.path.join(RESULTS_ROOT, sanitize_name(event_type))
#     cohort_dir = os.path.join(endpoint_dir, sanitize_name(label))
#     os.makedirs(cohort_dir, exist_ok=True)
#
#     feat_safe = sanitize_name(feature_name)
#     save_prefix = os.path.join(
#         cohort_dir,
#         f"km_cv_{feat_safe}_{sanitize_name(label)}_{event_type}_censor{CENSOR_AT}_cindex{avg_ci:.2f}_p{p_corr:.3f}",
#     )
#
#     df_cv.to_csv(save_prefix + ".csv", index=False)
#
#     # Save ONLY KM plot with risk table
#     if km_fig_counts is not None:
#         save_prefix2 = save_prefix.replace("km_cv_", "risked_km_cv_")
#         km_fig_counts.savefig(
#             save_prefix2 + ".png", dpi=600, bbox_inches="tight", pad_inches=0.01
#         )
#         km_fig_counts.savefig(
#             save_prefix2 + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.01
#         )
#         plt.close(km_fig_counts)
#
#     print(
#         f"Done | cohort={label} | endpoint={event_type} | feature={feature_name} | "
#         f"n={len(df)} | c-index={avg_ci:.3f} | p={p_corr:.4g}"
#     )
#
#
# def main():
#     df = pd.read_excel(BRAIN_FILE)
#
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
#     for col in ["histology", "idh_status", "grade"]:
#         if col in df.columns:
#             df[col] = df[col].astype(str).str.strip()
#
#     features_to_run = get_all_numeric_features(df)
#     print("Features to run:")
#     print(features_to_run)
#
#     cohorts = build_cohorts(df)
#
#     total_jobs = len(ENDPOINTS) * len(cohorts) * len(features_to_run)
#     print(f"Total jobs: {total_jobs}")
#
#     job_idx = 0
#     for event_type in ENDPOINTS:
#         for cohort_name, cohort_df in cohorts.items():
#             for feature_name in features_to_run:
#                 job_idx += 1
#                 print(
#                     f"\n[{job_idx}/{total_jobs}] "
#                     f"endpoint={event_type} | cohort={cohort_name} | feature={feature_name}"
#                 )
#                 try:
#                     run_for_feature_and_cohort(
#                         cohort_name,
#                         cohort_df.copy(),
#                         feature_name,
#                         event_type,
#                     )
#                 except Exception as e:
#                     print(
#                         f"FAILED | endpoint={event_type} | cohort={cohort_name} | "
#                         f"feature={feature_name} | error={e}"
#                     )
#
#
# if __name__ == "__main__":
#     main()
#


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredText

from pacman.config import DATA_DIR, RESULTS_DIR
from pacman.survival.utils import add_at_risk_counts

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
BRAIN_FILE = os.path.join(DATA_DIR, "heatmap_cin_raw_num.xlsx")
SAVE_ROOT = os.path.join(RESULTS_DIR, "survival", "km_high_astro_vs_gbm")
os.makedirs(SAVE_ROOT, exist_ok=True)

ENDPOINTS = ["OS", "PFI", "DSS", "DFI"]
CENSOR_AT = 120  # months
FONT_SIZE = 12
FIG_W = 6.5
FIG_H = 4.2

GROUP_A_NAME = "Grade 2&3 Astro"
GROUP_B_NAME = "GBM"

# =========================================================
# HELPERS
# =========================================================
def sanitize_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("&", "and")
    )


def smart_text_location(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lines = ax.get_lines()

    x_mid = (xlim[1] - xlim[0]) * 0.5
    y_mid = (ylim[1] - ylim[0]) * 0.5

    lower_left_count = 0
    upper_right_count = 0

    for line in lines:
        xdata, ydata = line.get_data()
        lower_left_count += np.sum((xdata < x_mid) & (ydata < y_mid))
        upper_right_count += np.sum((xdata > x_mid) & (ydata > y_mid))

    return "lower left" if lower_left_count < upper_right_count else "upper right"


def prepare_dataframe():
    df = pd.read_excel(BRAIN_FILE)

    # normalize grouping columns
    for col in ["histology", "idh_status", "grade"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # build requested cohorts
    # High astro = grade 2&3 astrocytoma
    df["comparison_group"] = np.nan

    mask_high_astro = (
        (df["histology"] == "Astrocytoma")
        & (df["grade"].isin(["Grade 2", "Grade 3"]))
    )

    # GBM from your cleaned file is typically Glioblastoma histology
    mask_gbm = df["histology"] == "Glioblastoma"

    df.loc[mask_high_astro, "comparison_group"] = GROUP_A_NAME
    df.loc[mask_gbm, "comparison_group"] = GROUP_B_NAME

    df = df.dropna(subset=["comparison_group"]).copy()

    return df


def prepare_endpoint_df(df: pd.DataFrame, event_col: str) -> pd.DataFrame:
    time_col = f"{event_col}.time"

    req_cols = ["comparison_group", event_col, time_col, "bcr_patient_barcode"]
    df_evt = df.dropna(subset=req_cols).copy()

    df_evt[event_col] = pd.to_numeric(df_evt[event_col], errors="coerce")
    df_evt[time_col] = pd.to_numeric(df_evt[time_col], errors="coerce")
    df_evt = df_evt.dropna(subset=[event_col, time_col]).copy()

    df_evt[event_col] = df_evt[event_col].astype(int)
    df_evt[time_col] = (df_evt[time_col] / 30.4).astype(int)

    if CENSOR_AT > 0:
        mask_cens = df_evt[time_col] > CENSOR_AT
        df_evt.loc[mask_cens, event_col] = 0
        df_evt.loc[mask_cens, time_col] = CENSOR_AT

    return df_evt


def make_km_plot(df_evt: pd.DataFrame, event_col: str):
    time_col = f"{event_col}.time"

    g1 = df_evt[df_evt["comparison_group"] == GROUP_A_NAME].copy()
    g2 = df_evt[df_evt["comparison_group"] == GROUP_B_NAME].copy()

    if len(g1) == 0 or len(g2) == 0:
        return None, None, None

    T1 = g1[time_col]
    E1 = g1[event_col]
    T2 = g2[time_col]
    E2 = g2[event_col]

    # log-rank
    logrank_res = logrank_test(T1, T2, E1, E2)
    p_val = logrank_res.p_value

    # KM with risk table
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_subplot(111)

    km1 = KaplanMeierFitter()
    km2 = KaplanMeierFitter()

    ax = km1.fit(
        T1,
        event_observed=E1,
        label=GROUP_A_NAME,
    ).plot_survival_function(
        ax=ax,
        show_censors=True,
        censor_styles={"ms": 5},
        ci_show=False,
        color="b",
        xlabel="Months",
        ylabel=None,
    )

    ax = km2.fit(
        T2,
        event_observed=E2,
        label=GROUP_B_NAME,
    ).plot_survival_function(
        ax=ax,
        show_censors=True,
        censor_styles={"ms": 5},
        ci_show=False,
        color="r",
        xlabel="Months",
        ylabel=None,
    )

    add_at_risk_counts(
        km1,
        km2,
        rows_to_show=["At risk"],
        xticks=[0, 25, 50, 75, 100],
        ypos=-0.55,
        colors=["blue", "red"],
        ax=ax,
        fig=fig,
        fontsize=FONT_SIZE,
    )

    fig.subplots_adjust(bottom=0.33, left=0.16)

    if p_val < 0.0001:
        p_txt = "p < 0.0001"
    else:
        p_txt = f"p = {p_val:.4f}"

    loc = smart_text_location(ax)
    ax.add_artist(
        AnchoredText(p_txt, loc=loc, frameon=False, prop=dict(size=FONT_SIZE))
    )

    ax.set_ylabel(f"{event_col} Probability", fontsize=FONT_SIZE)
    ax.set_xlabel("Months", fontsize=FONT_SIZE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, CENSOR_AT + 1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title(f"{GROUP_A_NAME} vs {GROUP_B_NAME}", fontsize=FONT_SIZE + 2)
    ax.spines[["right", "top"]].set_visible(False)

    summary = {
        "endpoint": event_col,
        "group_a": GROUP_A_NAME,
        "group_b": GROUP_B_NAME,
        "n_group_a": len(g1),
        "n_group_b": len(g2),
        "events_group_a": int(E1.sum()),
        "events_group_b": int(E2.sum()),
        "p_value": p_val,
        "median_group_a_months": km1.median_survival_time_,
        "median_group_b_months": km2.median_survival_time_,
    }

    return fig, logrank_res, summary


# =========================================================
# MAIN
# =========================================================
def main():
    df = prepare_dataframe()

    all_summaries = []

    for event_col in ENDPOINTS:
        print(f"Running {event_col} ...")

        df_evt = prepare_endpoint_df(df, event_col)

        fig, logrank_res, summary = make_km_plot(df_evt, event_col)

        if fig is None:
            print(f"Skipping {event_col}: missing data in one of the groups.")
            continue

        all_summaries.append(summary)

        save_prefix = os.path.join(
            SAVE_ROOT,
            f"risked_km_{sanitize_name(GROUP_A_NAME)}_vs_{sanitize_name(GROUP_B_NAME)}_{event_col}_censor{CENSOR_AT}"
        )

        fig.savefig(save_prefix + ".png", dpi=600, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(save_prefix + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

        print(
            f"Saved {event_col} | "
            f"n({GROUP_A_NAME})={summary['n_group_a']} | "
            f"n({GROUP_B_NAME})={summary['n_group_b']} | "
            f"p={summary['p_value']:.4g}"
        )

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            os.path.join(
                SAVE_ROOT,
                f"summary_{sanitize_name(GROUP_A_NAME)}_vs_{sanitize_name(GROUP_B_NAME)}.csv"
            ),
            index=False,
        )

    print("Done.")


if __name__ == "__main__":
    main()
