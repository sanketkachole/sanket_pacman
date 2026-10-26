import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.utils import concordance_index
from scipy.linalg import LinAlgWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import wandb
except Exception:
    wandb = None  # type: ignore

try:
    from pacman.config import DATA_DIR, RESULTS_DIR  # type: ignore

    DEFAULT_INPUT = str(Path(DATA_DIR) / "heatmap_cin_raw_num.xlsx")
    DEFAULT_OUTPUT = str(
        Path(RESULTS_DIR) / "survival" / "corrected_original_style_heatmaps"
    )
except Exception:
    DEFAULT_INPUT = "heatmap_cin_raw_num.xlsx"
    DEFAULT_OUTPUT = "corrected_original_style_heatmaps"


FEATURES: List[str] = [
    "HSC",
    "mean(ND)",
    "cv(ND)",
    "mean(CL)",
    "mean(HC)",
    "AMH",
    "AMAH",
    "AFW",
]

ENDPOINTS: List[str] = ["DSS", "PFI", "OS", "DFI"]

GROUP_ORDER: List[str] = [
    "IDH-wildtype",
    "IDH-mutant",
    "Grade 4 astro",
    "Grade 2&3 astro",
    "Oligo",
    "Grade 2&3 astro + oligo",
]

RENAME_MAP: Dict[str, str] = {
    "mit_hotspot_score": "HSC",
    "mit_nodeDegrees_mean": "mean(ND)",
    "mit_nodeDegrees_cv": "cv(ND)",
    "mit_clusterCoff_mean": "mean(CL)",
    "mit_cenHarmonic_mean": "mean(HC)",
    "aty_hotspot_ratio": "AMH",
    "aty_ahotspot_ratio": "AMAH",
    "aty_wsi_ratio": "AFW",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate corrected original-style heatmaps for univariate and "
            "multivariate survival C-index across the six requested groups."
        )
    )
    parser.add_argument(
        "--input_excel",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to heatmap_cin_raw_num.xlsx",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=ENDPOINTS,
        help="Endpoints to analyze",
    )
    parser.add_argument(
        "--censor_at_months",
        type=float,
        default=120.0,
        help="Administrative censoring time in months",
    )
    parser.add_argument(
        "--penalizer",
        type=float,
        default=0.1,
        help="Cox penalizer",
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.0,
        help="Cox l1 ratio",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Cross-validation folds for multivariate Cox",
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=10,
        help="Minimum rows required for analysis",
    )
    parser.add_argument(
        "--min_events",
        type=int,
        default=5,
        help="Minimum events required for analysis",
    )
    parser.add_argument(
        "--min_test_events",
        type=int,
        default=1,
        help="Minimum events required in a test fold",
    )
    parser.add_argument(
        "--standardize_features",
        action="store_true",
        help="Standardize features before Cox fitting",
    )
    parser.add_argument(
        "--perm_n",
        type=int,
        default=1000,
        help="Permutation count for univariate p-values",
    )
    parser.add_argument(
        "--fdr_alpha",
        type=float,
        default=0.05,
        help="FDR threshold",
    )
    parser.add_argument(
        "--plot_vmin",
        type=float,
        default=0.2,
        help="Heatmap min",
    )
    parser.add_argument(
        "--plot_vmax",
        type=float,
        default=0.8,
        help="Heatmap max",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="corrected-original-style-survival-heatmaps",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="W&B entity/team",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def normalize_string_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return df


def rename_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP)


def create_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "IDH-wildtype": df[df["idh_status"].eq("IDH-wildtype")].copy(),
        "IDH-mutant": df[df["idh_status"].eq("IDH-mutant")].copy(),
        "Grade 4 astro": df[
            df["histology"].eq("Astrocytoma") & df["grade"].eq("Grade 4")
        ].copy(),
        "Grade 2&3 astro": df[
            df["histology"].eq("Astrocytoma")
            & df["grade"].isin(["Grade 2", "Grade 3"])
        ].copy(),
        "Oligo": df[df["histology"].eq("Oligodendroglioma")].copy(),
        "Grade 2&3 astro + oligo": df[
            (
                df["histology"].eq("Astrocytoma")
                & df["grade"].isin(["Grade 2", "Grade 3"])
            )
            | df["histology"].eq("Oligodendroglioma")
        ].copy(),
    }


def preprocess_endpoint_df(
    df: pd.DataFrame,
    endpoint: str,
    censor_at_months: float,
) -> pd.DataFrame:
    time_col = f"{endpoint}.time"
    out = df.copy()

    out[endpoint] = pd.to_numeric(out[endpoint], errors="coerce")
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out = out.dropna(subset=[endpoint, time_col]).copy()

    if out.empty:
        return out

    out[endpoint] = out[endpoint].astype(int)
    out[time_col] = out[time_col] / 30.4

    if censor_at_months > 0:
        mask = out[time_col] > censor_at_months
        out.loc[mask, endpoint] = 0
        out.loc[mask, time_col] = float(censor_at_months)

    return out


def safe_concordance_index(
    times: np.ndarray,
    risk_scores: np.ndarray,
    events: np.ndarray,
) -> float:
    """
    concordance_index expects higher prediction = longer survival.
    Our feature values / Cox hazards are risk scores where higher = shorter survival.
    Therefore pass -risk_scores.
    """
    times = np.asarray(times, dtype=float)
    risk_scores = np.asarray(risk_scores, dtype=float)
    events = np.asarray(events, dtype=float)

    mask = np.isfinite(times) & np.isfinite(risk_scores) & np.isfinite(events)
    times = times[mask]
    risk_scores = risk_scores[mask]
    events = events[mask].astype(int)

    if len(times) < 3:
        return np.nan
    if events.sum() == 0 or events.sum() == len(events):
        return np.nan

    try:
        return float(concordance_index(times, -risk_scores, event_observed=events))
    except Exception:
        return np.nan


def permutation_c_index_test(
    group: pd.DataFrame,
    feature: str,
    time_col: str,
    event_col: str,
    n_permutations: int,
    random_state: int,
) -> Tuple[float, float]:
    times = np.asarray(group[time_col], dtype=float)
    events = np.asarray(group[event_col], dtype=float)
    scores = np.asarray(group[feature], dtype=float)

    if times.ndim > 1:
        times = times[:, 0]
    if events.ndim > 1:
        events = events[:, 0]
    if scores.ndim > 1:
        scores = scores[:, 0]

    mask = (~np.isnan(times)) & (~np.isnan(events)) & (~np.isnan(scores))
    times = times[mask]
    events = events[mask].astype(int)
    scores = scores[mask]

    if len(times) < 10 or events.sum() == 0 or events.sum() == len(events):
        return np.nan, np.nan

    obs_ci = safe_concordance_index(times, scores, events)
    if np.isnan(obs_ci):
        return np.nan, np.nan

    rng = np.random.RandomState(random_state)
    null_cis: List[float] = []

    for _ in range(n_permutations):
        shuffled = rng.permutation(scores)
        null_ci = safe_concordance_index(times, shuffled, events)
        if pd.notna(null_ci):
            null_cis.append(float(null_ci))

    if not null_cis:
        return obs_ci, np.nan

    null_arr = np.asarray(null_cis, dtype=float)
    p_value = float(np.mean(np.abs(null_arr - 0.5) >= np.abs(obs_ci - 0.5)))
    return obs_ci, p_value


def adjust_p_values(
    pval_df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> pd.DataFrame:
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


def prepare_design_matrix(
    df: pd.DataFrame,
    subset: Sequence[str],
    standardize_features: bool,
) -> pd.DataFrame:
    x = df[list(subset)].copy()
    for feat in subset:
        x[feat] = pd.to_numeric(x[feat], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    x_imp = pd.DataFrame(
        imputer.fit_transform(x),
        columns=list(subset),
        index=x.index,
    )

    keep_cols: List[str] = []
    for feat in subset:
        std_val = safe_float(x_imp[feat].std())
        if pd.notna(std_val) and std_val > 0:
            keep_cols.append(feat)

    if not keep_cols:
        return pd.DataFrame(index=x.index)

    x_imp = x_imp[keep_cols].copy()

    if standardize_features:
        scaler = StandardScaler()
        x_imp = pd.DataFrame(
            scaler.fit_transform(x_imp),
            columns=keep_cols,
            index=x_imp.index,
        )

    return x_imp.clip(-5.0, 5.0)


def fit_multivariate_cox(
    df: pd.DataFrame,
    endpoint: str,
    subset: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    standardize_features: bool,
) -> Tuple[Optional[CoxPHFitter], pd.DataFrame, str]:
    time_col = f"{endpoint}.time"
    core = df[[time_col, endpoint]].copy()
    x = prepare_design_matrix(df, subset, standardize_features)

    if x.empty:
        return None, pd.DataFrame(), "all_zero_variance"

    fit_df = pd.concat([core, x], axis=1)
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(fit_df) < 10:
        return None, fit_df, "too_few_rows"
    if fit_df[endpoint].sum() == 0 or fit_df[endpoint].sum() == len(fit_df):
        return None, fit_df, "invalid_event_distribution"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            warnings.filterwarnings("error", category=LinAlgWarning)
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            cph.fit(fit_df, duration_col=time_col, event_col=endpoint)
        return cph, fit_df, "ok"
    except (ConvergenceError, np.linalg.LinAlgError) as exc:
        return None, fit_df, f"failed:{type(exc).__name__}"
    except Exception as exc:
        return None, fit_df, f"failed:{type(exc).__name__}"


def fit_cv_cox_for_subset(
    df: pd.DataFrame,
    endpoint: str,
    subset: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    n_splits: int,
    min_test_events: int,
    random_state: int,
    standardize_features: bool,
) -> Dict[str, Any]:
    time_col = f"{endpoint}.time"
    base_df = df[[time_col, endpoint] + list(subset)].copy()

    for feat in subset:
        base_df[feat] = pd.to_numeric(base_df[feat], errors="coerce")

    base_df = base_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[time_col, endpoint]).copy()
    base_df = base_df.reset_index(drop=True)

    if len(base_df) < max(10, n_splits):
        return {"cv_mean_cindex": np.nan, "cv_std_cindex": np.nan, "valid_folds": 0, "status": "too_few_rows"}
    if base_df[endpoint].sum() == 0 or base_df[endpoint].sum() == len(base_df):
        return {"cv_mean_cindex": np.nan, "cv_std_cindex": np.nan, "valid_folds": 0, "status": "invalid_event_distribution"}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []

    for train_idx, test_idx in kf.split(base_df):
        train_df = base_df.iloc[train_idx].copy()
        test_df = base_df.iloc[test_idx].copy()

        if train_df[endpoint].sum() == 0 or train_df[endpoint].sum() == len(train_df):
            continue
        if test_df[endpoint].sum() < min_test_events or test_df[endpoint].sum() == len(test_df):
            continue

        train_core = train_df[[time_col, endpoint]].copy()
        test_core = test_df[[time_col, endpoint]].copy()

        train_x_raw = train_df[list(subset)].copy()
        test_x_raw = test_df[list(subset)].copy()

        for feat in subset:
            train_x_raw[feat] = pd.to_numeric(train_x_raw[feat], errors="coerce")
            test_x_raw[feat] = pd.to_numeric(test_x_raw[feat], errors="coerce")

        imputer = SimpleImputer(strategy="median")
        try:
            train_x = pd.DataFrame(
                imputer.fit_transform(train_x_raw),
                columns=list(subset),
                index=train_x_raw.index,
            )
            test_x = pd.DataFrame(
                imputer.transform(test_x_raw),
                columns=list(subset),
                index=test_x_raw.index,
            )
        except Exception:
            continue

        keep_cols: List[str] = []
        for feat in subset:
            std_val = safe_float(train_x[feat].std())
            if pd.notna(std_val) and std_val > 0:
                keep_cols.append(feat)

        if not keep_cols:
            continue

        train_x = train_x[keep_cols].copy()
        test_x = test_x[keep_cols].copy()

        if standardize_features:
            scaler = StandardScaler()
            try:
                train_x = pd.DataFrame(
                    scaler.fit_transform(train_x),
                    columns=keep_cols,
                    index=train_x.index,
                )
                test_x = pd.DataFrame(
                    scaler.transform(test_x),
                    columns=keep_cols,
                    index=test_x.index,
                )
            except Exception:
                continue

        train_fit = pd.concat([train_core, train_x], axis=1).dropna().copy()
        test_fit = pd.concat([test_core, test_x], axis=1).dropna().copy()

        if len(train_fit) < 10 or len(test_fit) < 3:
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                warnings.filterwarnings("error", category=LinAlgWarning)
                cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                cph.fit(train_fit, duration_col=time_col, event_col=endpoint)

            risk = cph.predict_partial_hazard(test_fit).to_numpy().reshape(-1)
            ci = safe_concordance_index(
                times=test_fit[time_col].to_numpy(),
                risk_scores=risk,
                events=test_fit[endpoint].to_numpy(),
            )
            if pd.notna(ci):
                fold_scores.append(float(ci))
        except Exception:
            continue

    if not fold_scores:
        return {"cv_mean_cindex": np.nan, "cv_std_cindex": np.nan, "valid_folds": 0, "status": "all_folds_failed"}

    return {
        "cv_mean_cindex": float(np.mean(fold_scores)),
        "cv_std_cindex": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else np.nan,
        "valid_folds": int(len(fold_scores)),
        "status": "ok",
    }


def compute_univariate_tables(
    groups: Dict[str, pd.DataFrame],
    endpoint: str,
    features: Sequence[str],
    censor_at_months: float,
    perm_n: int,
    fdr_alpha: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_col = f"{endpoint}.time"
    cindex_data: Dict[str, Dict[str, float]] = {}
    pval_data: Dict[str, Dict[str, float]] = {}
    diag_rows: List[Dict[str, Any]] = []

    for group_name in GROUP_ORDER:
        group_raw = groups[group_name]
        group = preprocess_endpoint_df(group_raw, endpoint, censor_at_months)

        c_row: Dict[str, float] = {}
        p_row: Dict[str, float] = {}

        n_events = int(group[endpoint].sum()) if len(group) > 0 else 0
        n_total = int(len(group))
        n_censored = n_total - n_events

        for feat in features:
            if feat not in group.columns:
                c_row[feat] = np.nan
                p_row[feat] = np.nan
                continue

            vals = pd.to_numeric(group[feat], errors="coerce")
            diag_rows.append(
                {
                    "group": group_name,
                    "endpoint": endpoint,
                    "feature": feat,
                    "n_total_after_endpoint_cleanup": n_total,
                    "n_non_missing_feature": int(vals.notna().sum()),
                    "mean": safe_float(vals.mean()),
                    "std": safe_float(vals.std()),
                    "min": safe_float(vals.min()),
                    "max": safe_float(vals.max()),
                    "n_events": n_events,
                    "n_censored": n_censored,
                }
            )

            sub = group[[time_col, endpoint, feat]].copy()
            sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
            sub = sub.dropna().copy()

            if len(sub) < 10 or sub[endpoint].sum() == 0 or sub[endpoint].sum() == len(sub):
                c_row[feat] = np.nan
                p_row[feat] = np.nan
                continue

            ci, p = permutation_c_index_test(
                group=sub,
                feature=feat,
                time_col=time_col,
                event_col=endpoint,
                n_permutations=perm_n,
                random_state=random_state,
            )
            c_row[feat] = ci
            p_row[feat] = p

        cindex_data[group_name] = c_row
        pval_data[group_name] = p_row

    cindex_df = pd.DataFrame(cindex_data).T.reindex(GROUP_ORDER)
    pval_df = pd.DataFrame(pval_data).T.reindex(GROUP_ORDER)
    sig_df = adjust_p_values(pval_df, alpha=fdr_alpha, method="fdr_bh")
    diag_df = pd.DataFrame(diag_rows)

    return cindex_df, sig_df, diag_df


def compute_multivariate_table(
    groups: Dict[str, pd.DataFrame],
    endpoint: str,
    features: Sequence[str],
    censor_at_months: float,
    penalizer: float,
    l1_ratio: float,
    n_splits: int,
    min_rows: int,
    min_events: int,
    min_test_events: int,
    random_state: int,
    standardize_features: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    in_sample_row: Dict[str, float] = {}
    cv_row: Dict[str, float] = {}
    diag_rows: List[Dict[str, Any]] = []

    for group_name in GROUP_ORDER:
        group_raw = groups[group_name]
        group = preprocess_endpoint_df(group_raw, endpoint, censor_at_months)
        n_events = int(group[endpoint].sum()) if len(group) > 0 else 0

        diag: Dict[str, Any] = {
            "group": group_name,
            "endpoint": endpoint,
            "n_rows_after_endpoint_cleanup": int(len(group)),
            "n_events": n_events,
            "fit_status": "not_run",
            "cv_status": "not_run",
            "n_rows_fit_df": np.nan,
            "n_events_fit_df": np.nan,
        }

        if len(group) < min_rows:
            in_sample_row[group_name] = np.nan
            cv_row[group_name] = np.nan
            diag["fit_status"] = "too_few_rows"
            diag["cv_status"] = "too_few_rows"
            diag_rows.append(diag)
            continue

        if n_events < min_events:
            in_sample_row[group_name] = np.nan
            cv_row[group_name] = np.nan
            diag["fit_status"] = "too_few_events"
            diag["cv_status"] = "too_few_events"
            diag_rows.append(diag)
            continue

        cph, fit_df, fit_status = fit_multivariate_cox(
            df=group,
            endpoint=endpoint,
            subset=features,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            standardize_features=standardize_features,
        )

        diag["fit_status"] = fit_status
        diag["n_rows_fit_df"] = int(len(fit_df))
        diag["n_events_fit_df"] = int(fit_df[endpoint].sum()) if not fit_df.empty else 0

        if cph is not None and not fit_df.empty:
            risk = cph.predict_partial_hazard(fit_df).to_numpy().reshape(-1)
            in_sample_row[group_name] = safe_concordance_index(
                times=fit_df[f"{endpoint}.time"].to_numpy(),
                risk_scores=risk,
                events=fit_df[endpoint].to_numpy(),
            )
        else:
            in_sample_row[group_name] = np.nan

        cv_info = fit_cv_cox_for_subset(
            df=group,
            endpoint=endpoint,
            subset=features,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            n_splits=n_splits,
            min_test_events=min_test_events,
            random_state=random_state,
            standardize_features=standardize_features,
        )
        cv_row[group_name] = safe_float(cv_info["cv_mean_cindex"])
        diag["cv_status"] = str(cv_info["status"])
        diag["cv_mean_cindex"] = safe_float(cv_info["cv_mean_cindex"])
        diag["cv_std_cindex"] = safe_float(cv_info["cv_std_cindex"])
        diag["cv_valid_folds"] = int(cv_info["valid_folds"])

        diag_rows.append(diag)

    mv_df = pd.DataFrame(
        {
            "In-sample Cox": in_sample_row,
            "CV Cox": cv_row,
        }
    ).T.reindex(columns=GROUP_ORDER)
    diag_df = pd.DataFrame(diag_rows)

    return mv_df, diag_df


def plot_heatmap_with_significance(
    df: pd.DataFrame,
    sig_matrix: pd.DataFrame,
    title: str,
    out_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    annot = df.round(2).astype(object)
    annot = annot.mask(df.isna(), "")
    annot = annot.where(~sig_matrix, annot.astype(str) + "*")

    fig_size = (8, 0.55 * len(df.index) + 2)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(
        df,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="gray",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "C-index"},
        ax=ax,
    )
    ax.set_ylabel("Subtype group")
    ax.set_xlabel("Feature")
    ax.set_title(title)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_multivariate_heatmap_original_style(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    annot = df.round(2).astype(object)
    annot = annot.mask(df.isna(), "")

    fig_size = (8, 0.55 * len(df.index) + 2)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(
        df,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="gray",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "C-index"},
        ax=ax,
    )
    ax.set_ylabel("Model")
    ax.set_xlabel("Subtype group")
    ax.set_title(title)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def init_wandb(args: argparse.Namespace) -> Optional[Any]:
    if not args.use_wandb or wandb is None:
        return None

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        config=vars(args),
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input_excel)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"

    ensure_dir(output_dir)
    ensure_dir(plots_dir)
    ensure_dir(tables_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logging.info("Loading data from %s", input_path)
    df = pd.read_excel(input_path)

    required_group_cols = ["histology", "idh_status", "grade"]
    missing_group_cols = [c for c in required_group_cols if c not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Missing required grouping columns: {missing_group_cols}")

    df = rename_feature_columns(df)
    df = normalize_string_columns(df, required_group_cols)

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns after renaming: {missing_features}")

    groups = create_groups(df)

    all_uni_diag: List[pd.DataFrame] = []
    all_mv_diag: List[pd.DataFrame] = []

    wandb_run = init_wandb(args)

    for endpoint in args.endpoints:
        logging.info("Processing endpoint: %s", endpoint)

        uni_df, sig_df, uni_diag_df = compute_univariate_tables(
            groups=groups,
            endpoint=endpoint,
            features=FEATURES,
            censor_at_months=args.censor_at_months,
            perm_n=args.perm_n,
            fdr_alpha=args.fdr_alpha,
            random_state=args.random_state,
        )

        mv_df, mv_diag_df = compute_multivariate_table(
            groups=groups,
            endpoint=endpoint,
            features=FEATURES,
            censor_at_months=args.censor_at_months,
            penalizer=args.penalizer,
            l1_ratio=args.l1_ratio,
            n_splits=args.n_splits,
            min_rows=args.min_rows,
            min_events=args.min_events,
            min_test_events=args.min_test_events,
            random_state=args.random_state,
            standardize_features=args.standardize_features,
        )

        uni_df.to_csv(tables_dir / f"univariate_cindex_table_{endpoint}.csv", index=True)
        sig_df.to_csv(tables_dir / f"univariate_fdr_significance_table_{endpoint}.csv", index=True)
        uni_diag_df.to_csv(tables_dir / f"univariate_feature_diagnostics_{endpoint}.csv", index=False)

        mv_df.to_csv(tables_dir / f"multivariate_cindex_table_{endpoint}.csv", index=True)
        mv_diag_df.to_csv(tables_dir / f"multivariate_diagnostics_{endpoint}.csv", index=False)

        plot_heatmap_with_significance(
            df=uni_df,
            sig_matrix=sig_df,
            title=f"C-index for {endpoint} (* : permutation test FDR-adjusted p-value < 0.05)",
            out_path=plots_dir / f"univariate_cindex_heatmap_{endpoint}.png",
            vmin=args.plot_vmin,
            vmax=args.plot_vmax,
        )

        plot_multivariate_heatmap_original_style(
            df=mv_df,
            title=f"Multivariate C-index ({endpoint})",
            out_path=plots_dir / f"multivariate_cindex_heatmap_{endpoint}.png",
            vmin=args.plot_vmin,
            vmax=args.plot_vmax,
        )

        all_uni_diag.append(uni_diag_df)
        all_mv_diag.append(mv_diag_df)

        if wandb_run is not None:
            wandb.log(
                {
                    f"{endpoint}/best_univariate_max": safe_float(np.nanmax(uni_df.values)),
                    f"{endpoint}/best_multivariate_cv_max": safe_float(
                        np.nanmax(mv_df.loc["CV Cox"].values)
                    ),
                }
            )

    if all_uni_diag:
        pd.concat(all_uni_diag, ignore_index=True).to_csv(
            tables_dir / "all_univariate_feature_diagnostics.csv",
            index=False,
        )

    if all_mv_diag:
        pd.concat(all_mv_diag, ignore_index=True).to_csv(
            tables_dir / "all_multivariate_diagnostics.csv",
            index=False,
        )

    if wandb_run is not None:
        for path in tables_dir.glob("*.csv"):
            wandb.save(str(path))
        for path in plots_dir.glob("*.png"):
            wandb.save(str(path))
        wandb.finish()

    logging.info("Finished. Outputs saved to: %s", output_dir)
    logging.info("Main plots:")
    for endpoint in args.endpoints:
        logging.info("- %s", plots_dir / f"univariate_cindex_heatmap_{endpoint}.png")
        logging.info("- %s", plots_dir / f"multivariate_cindex_heatmap_{endpoint}.png")


if __name__ == "__main__":
    main()