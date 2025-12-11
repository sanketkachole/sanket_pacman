import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

from pacman.config import DATA_DIR, RESULTS_DIR
from pacman.survival.utils import add_at_risk_counts, survival_stratification_analysis

warnings.filterwarnings("ignore")

# ----------------- CONFIG -----------------
FEATURE_NAME = "mean(ND)"       # stratification feature
IDH_COL = "idh_group"           # column with "IDH-mutant" / "IDH-wildtype"
CV_FOLDS = 5
PERM_REPS = 1000
CENSOR_AT = 120                 # months, as in paper
FONT_SIZE = 12
FIG_SIZE = 5
SPLITS_ROOT = "./splits"        # same as repo
BRAIN_FILE = os.path.join(DATA_DIR, "heatmap_cin_raw_num.xlsx")
RESULTS_ROOT = os.path.join(RESULTS_DIR, "survival", "kmplots_cv_brain")
os.makedirs(RESULTS_ROOT, exist_ok=True)
IDH_COL = "idh_group"
# ------------------------------------------


def smart_text_location(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lines = ax.get_lines()
    x_mid = (xlim[1] - xlim[0]) * 0.5
    y_threshold = (ylim[1] - ylim[0]) * 0.5

    lower_left_count = 0
    upper_right_count = 0
    for line in lines:
        xdata, ydata = line.get_data()
        lower_left_count += np.sum((xdata < x_mid) & (ydata < y_threshold))
        upper_right_count += np.sum((xdata > x_mid) & (ydata > y_threshold))

    return "lower left" if lower_left_count < upper_right_count else "upper right"


def cv_helper(
    discov_df,
    feat_name,
    event_col,
    time_col,
    split_folder,
    km_plot=False,
    add_counts=False,
    shuffle_data=False,
    cutoff_mode="median",
    cutoff_point=-1,
):
    """One CV + (optionally) KM plotting run for a single cohort."""
    EE = discov_df[event_col].to_numpy()
    rng = np.random.RandomState()

    # storage over folds
    c_indices = []
    p_values = []

    T_lower_test = T_upper_test = None
    E_lower_test = E_upper_test = None
    test_hazard_ratios = None

    for run in range(CV_FOLDS):
        # train/val split based on provided split files (same as PACMAN)
        split_path = os.path.join(split_folder, f"splits_{run}.csv")
        split_df = pd.read_csv(split_path)

        train_set = discov_df[
            discov_df["bcr_patient_barcode"].isin(split_df["train"])
        ].copy()
        test_set = discov_df[
            discov_df["bcr_patient_barcode"].isin(split_df["val"])
        ].copy()

        train_set.reset_index(drop=True, inplace=True)
        test_set.reset_index(drop=True, inplace=True)

        if shuffle_data:  # permutation test
            idx_tr = rng.permutation(len(train_set))
            train_set[event_col] = train_set[event_col].iloc[idx_tr].to_numpy()
            train_set[time_col] = train_set[time_col].iloc[idx_tr].to_numpy()

            idx_te = rng.permutation(len(test_set))
            test_set[event_col] = test_set[event_col].iloc[idx_te].to_numpy()
            test_set[time_col] = test_set[time_col].iloc[idx_te].to_numpy()

        output = survival_stratification_analysis(
            train_set,
            test_set,
            feat_name,        # raw feature, NOT Cox
            time_col,
            event_col,
            CENSOR_AT,
            cutoff_mode,
            cutoff_point,
        )

        if output == -1:
            continue

        (
            _T_lower_test,
            _T_upper_test,
            _E_lower_test,
            _E_upper_test,
            cindex_test,
            pvalue_test,
            temp_hr,
        ) = output

        if T_lower_test is None:
            T_lower_test, T_upper_test = _T_lower_test, _T_upper_test
            E_lower_test, E_upper_test = _E_lower_test, _E_upper_test
            test_hazard_ratios = temp_hr
        else:
            T_lower_test = pd.concat(
                [T_lower_test, _T_lower_test], ignore_index=True
            )
            T_upper_test = pd.concat(
                [T_upper_test, _T_upper_test], ignore_index=True
            )
            E_lower_test = pd.concat(
                [E_lower_test, _E_lower_test], ignore_index=True
            )
            E_upper_test = pd.concat(
                [E_upper_test, _E_upper_test], ignore_index=True
            )
            test_hazard_ratios = pd.concat(
                [test_hazard_ratios, temp_hr], axis=1, ignore_index=True
            )

        c_indices.append(cindex_test)
        p_values.append(pvalue_test)

    df_to_save = test_hazard_ratios.T
    df_to_save["c_index"] = c_indices
    df_to_save["p_value"] = p_values

    logrank_results = logrank_test(
        T_lower_test, T_upper_test, E_lower_test, E_upper_test
    )

    if not km_plot:
        return df_to_save, logrank_results, None, None

    # ========= KM plotting =========
    fig = plt.figure(figsize=(FIG_SIZE - 1.8, FIG_SIZE - 2))
    ax = fig.add_subplot(111)
    ax.set_xlabel("", fontsize=FONT_SIZE)
    ax.set_ylabel("", fontsize=FONT_SIZE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)

    km_high = KaplanMeierFitter()
    km_low = KaplanMeierFitter()

    upper_label = f"{feat_name}-High"
    lower_label = f"{feat_name}-Low"

    ax = km_high.fit(
        T_upper_test, event_observed=E_upper_test, label=upper_label
    ).plot_survival_function(
        ax=ax,
        show_censors=True,
        censor_styles={"ms": 4},
        color="r",
        ci_show=False,
        xlabel="Months",
        ylabel=None,
    )
    ax = km_low.fit(
        T_lower_test, event_observed=E_lower_test, label=lower_label
    ).plot_survival_function(
        ax=ax,
        show_censors=True,
        censor_styles={"ms": 4},
        color="b",
        ci_show=False,
        xlabel="Months",
        ylabel=None,
    )
    ax.get_legend().remove()

    fig_counts = None
    if add_counts:
        fig_counts = plt.figure(figsize=(FIG_SIZE - 1, FIG_SIZE - 2))
        ax_c = fig_counts.add_subplot(111)
        ax_c.set_xlabel("", fontsize=FONT_SIZE)
        ax_c.set_ylabel("", fontsize=FONT_SIZE)
        ax_c.tick_params(axis="x", labelsize=FONT_SIZE)
        ax_c.tick_params(axis="y", labelsize=FONT_SIZE)

        ax_c = km_high.fit(
            T_upper_test, event_observed=E_upper_test, label=upper_label
        ).plot_survival_function(
            ax=ax_c,
            show_censors=True,
            censor_styles={"ms": 5},
            color="r",
            ci_show=False,
            xlabel="Months",
            ylabel=None,
        )
        ax_c = km_low.fit(
            T_lower_test, event_observed=E_lower_test, label=lower_label
        ).plot_survival_function(
            ax=ax_c,
            show_censors=True,
            censor_styles={"ms": 5},
            color="b",
            ci_show=False,
            xlabel="Months",
            ylabel=None,
        )

        add_at_risk_counts(
            km_low,
            km_high,
            rows_to_show=["At risk"],
            xticks=[0, 25, 50, 75, 100],
            ypos=-0.6,
            colors=["blue", "red"],
            ax=ax_c,
            fig=fig_counts,
            fontsize=int(FONT_SIZE * 1),
        )
        fig_counts.subplots_adjust(bottom=0.4, left=0.2)
        ax_c.get_legend().remove()

    return df_to_save, logrank_results, fig, fig_counts


def run_for_cohort(label, df, event_type="OS"):
    """Run CV+perm KM for one cohort (combined, IDH-mutant, IDH-wildtype)."""
    time_col = f"{event_type}.time"
    event_col = event_type

    # drop NA and censor at 120 months
    df = df.dropna(subset=[event_col, time_col, FEATURE_NAME]).copy()
    df[event_col] = df[event_col].astype(int)
    df[time_col] = (df[time_col] / 30.4).astype(int)
    df.loc[df[time_col] > CENSOR_AT, event_col] = 0
    df.loc[df[time_col] > CENSOR_AT, time_col] = CENSOR_AT

    print(f"\n*** {label}: {event_type}, n={len(df)} ***")

    split_folder = os.path.join(SPLITS_ROOT, event_type, "GBMLGG")

    df_cv, logrank_res, km_fig, km_fig_counts = cv_helper(
        df,
        FEATURE_NAME,
        event_col,
        time_col,
        split_folder,
        km_plot=True,
        add_counts=True,
        cutoff_mode="median",
    )

    ref_stat = logrank_res.test_statistic
    num_greater = 0
    num_valid = 0

    for _ in tqdm(range(PERM_REPS), desc=f"{label} perm tests"):
        try:
            _, logrep, _, _ = cv_helper(
                df,
                FEATURE_NAME,
                event_col,
                time_col,
                split_folder,
                km_plot=False,
                add_counts=False,
                shuffle_data=True,
                cutoff_mode="median",
            )
        except Exception:
            continue
        if logrep.test_statistic > ref_stat:
            num_greater += 1
        num_valid += 1

    p_corr = num_greater / max(num_valid, 1)
    avg_ci = df_cv["c_index"].mean()

    print(f"Average C-index: {avg_ci:.3f}")
    print(f"Corrected p-value: {p_corr:.4g}")

    # annotate and style figure
    if p_corr < 0.0001:
        p_txt = "p < 0.0001"
    else:
        p_txt = f"p = {p_corr:.4f}"

    for fig in [km_fig, km_fig_counts]:
        if fig is None:
            continue
        ax = fig.axes[0]
        loc = smart_text_location(ax)
        ax.add_artist(
            AnchoredText(p_txt, loc=loc, frameon=False, prop=dict(size=FONT_SIZE))
        )
        ax.set_ylabel(f"{event_type} Probability")
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlim(0, CENSOR_AT + 1)
        ax.set_title(label, fontsize=FONT_SIZE + 2)
        ax.spines[["right", "top"]].set_visible(False)

    # save
    save_prefix = os.path.join(
        RESULTS_ROOT,
        f"km_cv_{label.replace(' ', '_')}_{event_type}_censor{CENSOR_AT}_cindex{avg_ci:.2f}_p{p_corr:.3f}",
    )
    df_cv.to_csv(save_prefix + ".csv", index=False)
    km_fig.savefig(save_prefix + ".png", dpi=600, bbox_inches="tight", pad_inches=0.01)
    km_fig.savefig(save_prefix + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.01)

    if km_fig_counts is not None:
        save_prefix2 = save_prefix.replace("km_cv_", "risked_km_cv_")
        km_fig_counts.savefig(
            save_prefix2 + ".png", dpi=600, bbox_inches="tight", pad_inches=0.01
        )


def main():
    df = pd.read_excel(BRAIN_FILE)

    # ---- map raw columns to paper feature names ----
    # adjust if your column names differ
    df["HSC"] = df["mit_hotspot_score"]
    df["mean(ND)"] = df["mit_nodeDegrees_mean"]
    df["cv(ND)"] = df["mit_nodeDegrees_cv"]
    df["mean(CL)"] = df["mit_clusterCoff_mean"]
    df["mean(HC)"] = df["mit_cenHarmonic_mean"]

    # --- NEW: build IDH group column ---
    # If your IDH group labels live in the first 'type' column:
    df["idh_group"] = df["type"].astype(str).str.strip()
    # (if they’re in 'type.1' instead, change to df["type.1"]...)

    # combined GBMLGG
    run_for_cohort("GBMLGG", df, event_type="OS")
    # IDH-mutant
    run_for_cohort("IDH-mutant", df[df[IDH_COL] == "IDH-mutant"], event_type="OS")
    # IDH-wildtype
    run_for_cohort("IDH-wildtype", df[df[IDH_COL] == "IDH-wildtype"], event_type="OS")


if __name__ == "__main__":
    main()
