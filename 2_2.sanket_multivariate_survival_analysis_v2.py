import argparse
import itertools
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from scipy.linalg import LinAlgWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import wandb
except Exception:
    wandb = None  # type: ignore

# Optional pacman config fallback
try:
    from pacman.config import DATA_DIR, RESULTS_DIR  # type: ignore

    DEFAULT_INPUT = str(Path(DATA_DIR) / "heatmap_cin_raw_num.xlsx")
    DEFAULT_OUTPUT = str(Path(RESULTS_DIR) / "survival" / "multivariate_feature_search_v2")
except Exception:
    DEFAULT_INPUT = "heatmap_cin_raw_num.xlsx"
    DEFAULT_OUTPUT = "multivariate_feature_search_v2"


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

ENDPOINTS: List[str] = ["OS", "PFI", "DSS", "DFI"]

GROUP_ORDER: List[str] = [
    "IDH-wildtype",
    "IDH-mutant",
    "Grade 4 astro",
    "Grade 2&3 astro",
    "Oligo",
    "Grade 2&3 astro + oligo",
]

NUMERIC_WARNINGS = (
    RuntimeWarning,
    LinAlgWarning,
    UserWarning,
)


@dataclass(frozen=True)
class ExperimentConfig:
    max_features: int
    min_features: int
    top_k_univariate: int
    corr_threshold: float
    penalizers: Tuple[float, ...]
    l1_ratios: Tuple[float, ...]
    n_splits: int
    min_rows: int
    min_events: int
    random_state: int
    censor_at_months: int
    require_ph_ok: bool
    ph_p_threshold: float
    max_vif_allowed: float
    target_feature: Optional[str]
    force_include_target: bool
    min_test_events: int
    max_combinations_per_size: int
    min_unique_values: int
    max_abs_corr_skip: float
    standardize_features: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stable multivariate Cox subset search with stronger numerical safeguards. "
            "Searches small feature subsets, applies scaling, removes near-constant and "
            "highly collinear features, sweeps penalizers, and ranks models by CV C-index."
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
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=ENDPOINTS,
        help="Endpoints to analyze. Example: --endpoints OS DSS",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=GROUP_ORDER,
        help="Groups to analyze.",
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=1,
        help="Minimum subset size to test.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=3,
        help="Maximum subset size to test.",
    )
    parser.add_argument(
        "--top_k_univariate",
        type=int,
        default=5,
        help="Keep top-K univariate features as candidates before subset search.",
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.80,
        help="Absolute correlation threshold for redundancy filtering.",
    )
    parser.add_argument(
        "--penalizers",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="Penalizer values to sweep.",
    )
    parser.add_argument(
        "--l1_ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0],
        help="L1 ratios to sweep.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Cross-validation folds.",
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=20,
        help="Minimum rows required to attempt subset search.",
    )
    parser.add_argument(
        "--min_events",
        type=int,
        default=10,
        help="Minimum events required to attempt subset search.",
    )
    parser.add_argument(
        "--min_test_events",
        type=int,
        default=1,
        help="Minimum events required in a CV test fold.",
    )
    parser.add_argument(
        "--censor_at_months",
        type=int,
        default=120,
        help="Administrative censoring time in months.",
    )
    parser.add_argument(
        "--require_ph_ok",
        action="store_true",
        help="Drop features that fail univariate PH test before subset search.",
    )
    parser.add_argument(
        "--ph_p_threshold",
        type=float,
        default=0.05,
        help="PH p-value threshold below which a feature is considered to violate PH.",
    )
    parser.add_argument(
        "--max_vif_allowed",
        type=float,
        default=10.0,
        help="Discard subsets whose max VIF exceeds this threshold.",
    )
    parser.add_argument(
        "--target_feature",
        type=str,
        default="AFW",
        help="Feature to build around. Use '' to disable.",
    )
    parser.add_argument(
        "--force_include_target",
        action="store_true",
        help="Force every tested subset to include target_feature.",
    )
    parser.add_argument(
        "--max_combinations_per_size",
        type=int,
        default=200,
        help="Cap the number of subsets tested per subset size.",
    )
    parser.add_argument(
        "--min_unique_values",
        type=int,
        default=3,
        help="Minimum unique values required for a feature inside analyzed data.",
    )
    parser.add_argument(
        "--max_abs_corr_skip",
        type=float,
        default=0.95,
        help="Immediately skip a subset if any pairwise |corr| exceeds this value.",
    )
    parser.add_argument(
        "--no_standardize_features",
        action="store_true",
        help="Disable train-fold standardization. Not recommended.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="survival-multivariate-search",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="W&B entity/team. Leave empty if not needed.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    return (
        name.replace("&", "and")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def rename_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
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


def normalize_string_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return df


def create_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    hist_col = "histology"
    idh_col = "idh_status"
    grade_col = "grade"

    return {
        "IDH-wildtype": df[df[idh_col].eq("IDH-wildtype")].copy(),
        "IDH-mutant": df[df[idh_col].eq("IDH-mutant")].copy(),
        "Grade 4 astro": df[
            df[hist_col].eq("Astrocytoma") & df[grade_col].eq("Grade 4")
        ].copy(),
        "Grade 2&3 astro": df[
            df[hist_col].eq("Astrocytoma")
            & df[grade_col].isin(["Grade 2", "Grade 3"])
        ].copy(),
        "Oligo": df[df[hist_col].eq("Oligodendroglioma")].copy(),
        "Grade 2&3 astro + oligo": df[
            (
                df[hist_col].eq("Astrocytoma")
                & df[grade_col].isin(["Grade 2", "Grade 3"])
            )
            | df[hist_col].eq("Oligodendroglioma")
        ].copy(),
    }


def safe_concordance_index(
    times: np.ndarray,
    risk_scores: np.ndarray,
    events: np.ndarray,
) -> float:
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
        return float(concordance_index(times, risk_scores, event_observed=events))
    except Exception:
        return np.nan


def preprocess_endpoint_df(
    df: pd.DataFrame,
    endpoint: str,
    censor_at_months: int,
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
        mask = out[time_col] > float(censor_at_months)
        out.loc[mask, endpoint] = 0
        out.loc[mask, time_col] = float(censor_at_months)

    return out


def impute_numeric(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    num_df = df[list(features)].copy()
    for feat in features:
        num_df[feat] = pd.to_numeric(num_df[feat], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    arr = imputer.fit_transform(num_df)
    return pd.DataFrame(arr, columns=list(features), index=df.index)


def remove_low_information_features(
    df: pd.DataFrame,
    features: Sequence[str],
    min_unique_values: int,
) -> Tuple[List[str], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    kept: List[str] = []

    for feat in features:
        ser = pd.to_numeric(df[feat], errors="coerce")
        n_non_missing = int(ser.notna().sum())
        n_unique = int(ser.dropna().nunique())
        std_val = safe_float(ser.std())
        keep = n_non_missing >= 10 and n_unique >= min_unique_values and pd.notna(std_val) and std_val > 0

        rows.append(
            {
                "feature": feat,
                "n_non_missing": n_non_missing,
                "n_unique": n_unique,
                "std": std_val,
                "keep": keep,
            }
        )
        if keep:
            kept.append(feat)

    return kept, pd.DataFrame(rows)


def compute_corr_matrix(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    if len(features) == 0:
        return pd.DataFrame()
    imp = impute_numeric(df, features)
    return imp.corr()


def compute_vif_table(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    if len(features) <= 1:
        return pd.DataFrame({"feature": list(features), "vif": [1.0] * len(features)})

    imp = impute_numeric(df, features)
    rows: List[Dict[str, Any]] = []

    for i, feat in enumerate(features):
        try:
            vif_val = variance_inflation_factor(imp.values, i)
        except Exception:
            vif_val = np.nan
        rows.append({"feature": feat, "vif": safe_float(vif_val)})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def greedy_correlation_filter(
    corr_df: pd.DataFrame,
    ranked_features: Sequence[str],
    threshold: float,
) -> List[str]:
    kept: List[str] = []

    for feat in ranked_features:
        if feat not in corr_df.columns:
            kept.append(feat)
            continue

        keep = True
        for prev in kept:
            if prev in corr_df.columns:
                corr_val = safe_float(corr_df.loc[feat, prev])
                if pd.notna(corr_val) and abs(corr_val) >= threshold:
                    keep = False
                    break

        if keep:
            kept.append(feat)

    return kept


def max_abs_corr(corr_df: pd.DataFrame) -> float:
    if corr_df.empty or corr_df.shape[1] <= 1:
        return 0.0
    arr = corr_df.abs().to_numpy().copy()
    np.fill_diagonal(arr, np.nan)
    return float(np.nanmax(arr))


def compute_univariate_feature_screen(
    df: pd.DataFrame,
    endpoint: str,
    features: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    ph_p_threshold: float,
) -> pd.DataFrame:
    time_col = f"{endpoint}.time"
    rows: List[Dict[str, Any]] = []

    for feat in features:
        if feat not in df.columns:
            continue

        sub = df[[time_col, endpoint, feat]].copy()
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        sub = sub.dropna().copy()

        row: Dict[str, Any] = {
            "feature": feat,
            "n_used": int(len(sub)),
            "n_events": int(sub[endpoint].sum()) if not sub.empty else 0,
            "univariate_cindex": np.nan,
            "cox_coef": np.nan,
            "cox_hr": np.nan,
            "cox_p": np.nan,
            "ph_p": np.nan,
            "ph_ok": False,
            "status": "ok",
        }

        if len(sub) < 10:
            row["status"] = "too_few_rows"
            rows.append(row)
            continue

        if sub[endpoint].sum() == 0 or sub[endpoint].sum() == len(sub):
            row["status"] = "invalid_event_distribution"
            rows.append(row)
            continue

        row["univariate_cindex"] = safe_concordance_index(
            times=sub[time_col].to_numpy(),
            risk_scores=-sub[feat].to_numpy(),
            events=sub[endpoint].to_numpy(),
        )

        try:
            safe_sub = sub.copy()
            scaler = StandardScaler()
            safe_sub[[feat]] = scaler.fit_transform(safe_sub[[feat]])

            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                warnings.filterwarnings("error", category=LinAlgWarning)

                cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                cph.fit(safe_sub, duration_col=time_col, event_col=endpoint)

            row["cox_coef"] = safe_float(cph.params_[feat])
            row["cox_hr"] = safe_float(np.exp(cph.params_[feat]))
            row["cox_p"] = safe_float(cph.summary.loc[feat, "p"])

            try:
                ph = proportional_hazard_test(cph, safe_sub, time_transform="rank")
                ph_summary = ph.summary.reset_index()
                if "index" in ph_summary.columns:
                    ph_summary = ph_summary.rename(columns={"index": "feature"})
                ph_p = ph_summary.loc[ph_summary["feature"] == feat, "p"]
                row["ph_p"] = safe_float(ph_p.iloc[0]) if not ph_p.empty else np.nan
                row["ph_ok"] = bool(
                    pd.notna(row["ph_p"]) and float(row["ph_p"]) >= ph_p_threshold
                )
            except Exception:
                row["ph_p"] = np.nan
                row["ph_ok"] = False

        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}"

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        by=["univariate_cindex", "cox_p"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_candidate_pool(
    df: pd.DataFrame,
    endpoint: str,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    kept_after_low_info, low_info_df = remove_low_information_features(
        df=df,
        features=FEATURES,
        min_unique_values=config.min_unique_values,
    )

    screen_df = compute_univariate_feature_screen(
        df=df,
        endpoint=endpoint,
        features=kept_after_low_info,
        penalizer=min(config.penalizers),
        l1_ratio=min(config.l1_ratios),
        ph_p_threshold=config.ph_p_threshold,
    )

    if screen_df.empty:
        return screen_df, [], pd.DataFrame(), low_info_df

    candidate_df = screen_df.copy()

    if config.require_ph_ok:
        candidate_df = candidate_df[candidate_df["ph_ok"]].copy()

    candidate_df = candidate_df[
        candidate_df["univariate_cindex"].notna()
        & candidate_df["status"].eq("ok")
    ].copy()

    ranked = candidate_df["feature"].tolist()[: config.top_k_univariate]
    if not ranked:
        ranked = screen_df["feature"].tolist()[: config.top_k_univariate]

    if config.target_feature and config.target_feature in screen_df["feature"].tolist():
        if config.target_feature not in ranked:
            ranked = [config.target_feature] + ranked
            ranked = list(dict.fromkeys(ranked))

    corr_df = compute_corr_matrix(df, ranked)
    filtered = greedy_correlation_filter(corr_df, ranked, config.corr_threshold)

    if config.target_feature and config.target_feature in ranked and config.force_include_target:
        if config.target_feature not in filtered:
            filtered = [config.target_feature] + filtered
            filtered = list(dict.fromkeys(filtered))

    return screen_df, filtered, corr_df, low_info_df


def generate_subsets(
    features: Sequence[str],
    min_size: int,
    max_size: int,
    target_feature: Optional[str],
    force_include_target: bool,
    max_combinations_per_size: int,
) -> List[Tuple[str, ...]]:
    feats = list(dict.fromkeys(features))
    subsets: List[Tuple[str, ...]] = []

    for size in range(min_size, min(max_size, len(feats)) + 1):
        combos = list(itertools.combinations(feats, size))

        if target_feature and force_include_target:
            combos = [c for c in combos if target_feature in c]

        if len(combos) > max_combinations_per_size:
            combos = combos[:max_combinations_per_size]

        subsets.extend(combos)

    return subsets


def prepare_fold_design_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subset: Sequence[str],
    standardize_features: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    train_x = train_df[list(subset)].copy()
    test_x = test_df[list(subset)].copy()

    for feat in subset:
        train_x[feat] = pd.to_numeric(train_x[feat], errors="coerce")
        test_x[feat] = pd.to_numeric(test_x[feat], errors="coerce")

    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    test_x = test_x.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    try:
        train_x_imp = pd.DataFrame(
            imputer.fit_transform(train_x),
            columns=list(subset),
            index=train_x.index,
        )
        test_x_imp = pd.DataFrame(
            imputer.transform(test_x),
            columns=list(subset),
            index=test_x.index,
        )
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"imputation_failed:{type(exc).__name__}"

    # Drop columns with zero variance in train.
    keep_cols: List[str] = []
    for feat in subset:
        std_val = safe_float(train_x_imp[feat].std())
        if pd.notna(std_val) and std_val > 0:
            keep_cols.append(feat)

    if len(keep_cols) == 0:
        return pd.DataFrame(), pd.DataFrame(), "all_train_zero_variance"

    train_x_imp = train_x_imp[keep_cols].copy()
    test_x_imp = test_x_imp[keep_cols].copy()

    if standardize_features:
        scaler = StandardScaler()
        try:
            train_x_scaled = pd.DataFrame(
                scaler.fit_transform(train_x_imp),
                columns=keep_cols,
                index=train_x_imp.index,
            )
            test_x_scaled = pd.DataFrame(
                scaler.transform(test_x_imp),
                columns=keep_cols,
                index=test_x_imp.index,
            )
        except Exception as exc:
            return pd.DataFrame(), pd.DataFrame(), f"scaling_failed:{type(exc).__name__}"
    else:
        train_x_scaled = train_x_imp
        test_x_scaled = test_x_imp

    # Clip extreme values to reduce exp overflow inside Cox.
    train_x_scaled = train_x_scaled.clip(-5.0, 5.0)
    test_x_scaled = test_x_scaled.clip(-5.0, 5.0)

    return train_x_scaled, test_x_scaled, "ok"


def fit_cox_on_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    endpoint: str,
    subset: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    min_test_events: int,
    standardize_features: bool,
) -> Tuple[float, str, int]:
    time_col = f"{endpoint}.time"

    train_core = train_df[[time_col, endpoint]].copy()
    test_core = test_df[[time_col, endpoint]].copy()

    train_x, test_x, prep_status = prepare_fold_design_matrices(
        train_df=train_df,
        test_df=test_df,
        subset=subset,
        standardize_features=standardize_features,
    )
    if prep_status != "ok":
        return np.nan, prep_status, 0

    used_features = list(train_x.columns)

    train_sub = pd.concat([train_core, train_x], axis=1)
    test_sub = pd.concat([test_core, test_x], axis=1)

    train_sub = train_sub.replace([np.inf, -np.inf], np.nan).dropna().copy()
    test_sub = test_sub.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(train_sub) < 10:
        return np.nan, "too_few_train_rows", len(used_features)
    if len(test_sub) < 3:
        return np.nan, "too_few_test_rows", len(used_features)

    train_events = int(train_sub[endpoint].sum())
    test_events = int(test_sub[endpoint].sum())

    if train_events == 0 or train_events == len(train_sub):
        return np.nan, "invalid_train_event_distribution", len(used_features)
    if test_events < min_test_events or test_events == len(test_sub):
        return np.nan, "invalid_test_event_distribution", len(used_features)

    train_times = train_sub[time_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(train_times)) or np.min(train_times) <= 0:
        return np.nan, "invalid_train_times", len(used_features)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            warnings.filterwarnings("error", category=LinAlgWarning)

            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            cph.fit(train_sub, duration_col=time_col, event_col=endpoint)

        risk = cph.predict_partial_hazard(test_sub).to_numpy().reshape(-1)
        ci = safe_concordance_index(
            times=test_sub[time_col].to_numpy(),
            risk_scores=risk,
            events=test_sub[endpoint].to_numpy(),
        )
        return ci, "ok", len(used_features)

    except Exception as exc:
        return np.nan, f"failed:{type(exc).__name__}", len(used_features)


def cross_validated_subset_score(
    df: pd.DataFrame,
    endpoint: str,
    subset: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    n_splits: int,
    random_state: int,
    min_test_events: int,
    standardize_features: bool,
) -> Dict[str, Any]:
    time_col = f"{endpoint}.time"
    cols = [time_col, endpoint] + list(subset)
    sub = df[cols].copy()

    for feat in subset:
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[time_col, endpoint]).copy()
    sub = sub.reset_index(drop=True)

    if len(sub) < max(10, n_splits):
        return {
            "cv_mean_cindex": np.nan,
            "cv_std_cindex": np.nan,
            "n_rows_used": int(len(sub)),
            "n_events_used": int(sub[endpoint].sum()) if len(sub) > 0 else 0,
            "valid_folds": 0,
            "median_used_features": 0,
            "status": "too_few_rows",
            "fold_scores": [],
        }

    if sub[endpoint].sum() == 0 or sub[endpoint].sum() == len(sub):
        return {
            "cv_mean_cindex": np.nan,
            "cv_std_cindex": np.nan,
            "n_rows_used": int(len(sub)),
            "n_events_used": int(sub[endpoint].sum()),
            "valid_folds": 0,
            "median_used_features": 0,
            "status": "invalid_event_distribution",
            "fold_scores": [],
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []
    fold_statuses: List[str] = []
    used_feature_counts: List[int] = []

    for train_idx, test_idx in kf.split(sub):
        train_df = sub.iloc[train_idx].copy()
        test_df = sub.iloc[test_idx].copy()

        ci, status, n_used_features = fit_cox_on_subset(
            train_df=train_df,
            test_df=test_df,
            endpoint=endpoint,
            subset=subset,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            min_test_events=min_test_events,
            standardize_features=standardize_features,
        )
        fold_statuses.append(status)
        used_feature_counts.append(n_used_features)

        if pd.notna(ci):
            fold_scores.append(float(ci))

    status = "ok" if len(fold_scores) > 0 else f"all_folds_failed:{'|'.join(fold_statuses)}"
    return {
        "cv_mean_cindex": float(np.mean(fold_scores)) if fold_scores else np.nan,
        "cv_std_cindex": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else np.nan,
        "n_rows_used": int(len(sub)),
        "n_events_used": int(sub[endpoint].sum()),
        "valid_folds": int(len(fold_scores)),
        "median_used_features": int(np.median(used_feature_counts)) if used_feature_counts else 0,
        "status": status,
        "fold_scores": fold_scores,
    }


def fit_final_model_and_summary(
    df: pd.DataFrame,
    endpoint: str,
    subset: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    standardize_features: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    time_col = f"{endpoint}.time"
    core_df = df[[time_col, endpoint]].copy()

    x_df = df[list(subset)].copy()
    for feat in subset:
        x_df[feat] = pd.to_numeric(x_df[feat], errors="coerce")
    x_df = x_df.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    try:
        x_imp = pd.DataFrame(
            imputer.fit_transform(x_df),
            columns=list(subset),
            index=x_df.index,
        )
    except Exception as exc:
        return (
            pd.DataFrame({"feature": list(subset), "coef": np.nan, "exp_coef_hr": np.nan, "p": np.nan}),
            pd.DataFrame(),
            pd.DataFrame(),
            f"imputation_failed:{type(exc).__name__}",
        )

    keep_cols: List[str] = []
    for feat in subset:
        std_val = safe_float(x_imp[feat].std())
        if pd.notna(std_val) and std_val > 0:
            keep_cols.append(feat)

    if len(keep_cols) == 0:
        return (
            pd.DataFrame({"feature": list(subset), "coef": np.nan, "exp_coef_hr": np.nan, "p": np.nan}),
            pd.DataFrame(),
            pd.DataFrame(),
            "all_zero_variance",
        )

    x_imp = x_imp[keep_cols].copy()

    if standardize_features:
        scaler = StandardScaler()
        x_scaled = pd.DataFrame(
            scaler.fit_transform(x_imp),
            columns=keep_cols,
            index=x_imp.index,
        )
    else:
        x_scaled = x_imp

    x_scaled = x_scaled.clip(-5.0, 5.0)

    sub = pd.concat([core_df, x_scaled], axis=1)
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna().copy()

    coef_df = pd.DataFrame(
        {
            "feature": list(subset),
            "coef": [np.nan] * len(subset),
            "exp_coef_hr": [np.nan] * len(subset),
            "p": [np.nan] * len(subset),
            "used_in_final_fit": [feat in keep_cols for feat in subset],
        }
    )

    if len(sub) < 10:
        return coef_df, pd.DataFrame(), pd.DataFrame(), "too_few_rows"

    if sub[endpoint].sum() == 0 or sub[endpoint].sum() == len(sub):
        return coef_df, pd.DataFrame(), pd.DataFrame(), "invalid_event_distribution"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            warnings.filterwarnings("error", category=LinAlgWarning)

            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            cph.fit(sub, duration_col=time_col, event_col=endpoint)

        summary_df = cph.summary.reset_index()
        if "covariate" in summary_df.columns:
            summary_df = summary_df.rename(columns={"covariate": "feature"})
        elif "index" in summary_df.columns:
            summary_df = summary_df.rename(columns={"index": "feature"})

        if {"feature", "coef", "exp(coef)", "p"}.issubset(summary_df.columns):
            fit_coef_df = summary_df[["feature", "coef", "exp(coef)", "p"]].copy()
            fit_coef_df = fit_coef_df.rename(columns={"exp(coef)": "exp_coef_hr"})
            coef_df = coef_df.drop(columns=["coef", "exp_coef_hr", "p"]).merge(
                fit_coef_df,
                on="feature",
                how="left",
            )

        try:
            ph = proportional_hazard_test(cph, sub, time_transform="rank")
            ph_df = ph.summary.reset_index()
            if "index" in ph_df.columns:
                ph_df = ph_df.rename(columns={"index": "feature"})
        except Exception:
            ph_df = pd.DataFrame()

        risk = cph.predict_partial_hazard(sub).to_numpy().reshape(-1)
        fit_info = pd.DataFrame(
            {
                "n_rows": [int(len(sub))],
                "n_events": [int(sub[endpoint].sum())],
                "in_sample_cindex": [
                    safe_concordance_index(
                        times=sub[time_col].to_numpy(),
                        risk_scores=risk,
                        events=sub[endpoint].to_numpy(),
                    )
                ],
                "n_features_requested": [len(subset)],
                "n_features_used": [len(keep_cols)],
            }
        )

        if not ph_df.empty:
            ph_df = ph_df.merge(
                fit_info.assign(dummy_key=1),
                how="cross",
            )

        return coef_df, summary_df, ph_df, "ok"

    except Exception as exc:
        return coef_df, pd.DataFrame(), pd.DataFrame(), f"failed:{type(exc).__name__}"


def plot_top_results(
    results_df: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 15,
) -> None:
    if results_df.empty:
        return

    plot_df = results_df.sort_values(
        by="cv_mean_cindex",
        ascending=False,
        na_position="last",
    ).head(top_n)

    if plot_df.empty:
        return

    labels = [
        f"{row['subset_str']} | pen={row['penalizer']} | l1={row['l1_ratio']}"
        for _, row in plot_df.iterrows()
    ]

    plt.figure(figsize=(12, max(4, 0.45 * len(plot_df))))
    plt.barh(labels, plot_df["cv_mean_cindex"])
    plt.gca().invert_yaxis()
    plt.xlabel("CV mean C-index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def init_wandb(args: argparse.Namespace, config: ExperimentConfig) -> Optional[Any]:
    if not args.use_wandb or wandb is None:
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        config={
            "input_excel": args.input_excel,
            "output_dir": args.output_dir,
            "min_features": config.min_features,
            "max_features": config.max_features,
            "top_k_univariate": config.top_k_univariate,
            "corr_threshold": config.corr_threshold,
            "penalizers": list(config.penalizers),
            "l1_ratios": list(config.l1_ratios),
            "n_splits": config.n_splits,
            "min_rows": config.min_rows,
            "min_events": config.min_events,
            "censor_at_months": config.censor_at_months,
            "require_ph_ok": config.require_ph_ok,
            "ph_p_threshold": config.ph_p_threshold,
            "max_vif_allowed": config.max_vif_allowed,
            "target_feature": config.target_feature,
            "force_include_target": config.force_include_target,
            "min_test_events": config.min_test_events,
            "max_combinations_per_size": config.max_combinations_per_size,
            "random_state": config.random_state,
            "min_unique_values": config.min_unique_values,
            "max_abs_corr_skip": config.max_abs_corr_skip,
            "standardize_features": config.standardize_features,
        },
    )
    return run


def evaluate_group_endpoint(
    df: pd.DataFrame,
    group_name: str,
    endpoint: str,
    config: ExperimentConfig,
    group_out_dir: Path,
    wandb_run: Optional[Any],
) -> Dict[str, Any]:
    ensure_dir(group_out_dir)

    summary: Dict[str, Any] = {
        "group": group_name,
        "endpoint": endpoint,
        "status": "ok",
        "n_rows": int(len(df)),
        "n_events": int(df[endpoint].sum()) if endpoint in df.columns and not df.empty else 0,
    }

    if len(df) < config.min_rows:
        summary["status"] = "too_few_rows"
        return summary

    if int(df[endpoint].sum()) < config.min_events:
        summary["status"] = "too_few_events"
        return summary

    screen_df, candidate_pool, corr_df, low_info_df = build_candidate_pool(
        df=df,
        endpoint=endpoint,
        config=config,
    )

    screen_df.to_csv(group_out_dir / "univariate_feature_screen.csv", index=False)
    low_info_df.to_csv(group_out_dir / "low_information_feature_filter.csv", index=False)
    corr_df.to_csv(group_out_dir / "candidate_pool_correlation.csv")

    pd.DataFrame({"candidate_feature": candidate_pool}).to_csv(
        group_out_dir / "candidate_pool.csv",
        index=False,
    )

    if len(candidate_pool) == 0:
        summary["status"] = "no_candidates"
        return summary

    subsets = generate_subsets(
        features=candidate_pool,
        min_size=config.min_features,
        max_size=config.max_features,
        target_feature=config.target_feature,
        force_include_target=config.force_include_target,
        max_combinations_per_size=config.max_combinations_per_size,
    )

    if len(subsets) == 0:
        summary["status"] = "no_subsets"
        return summary

    results: List[Dict[str, Any]] = []

    for subset in subsets:
        subset_corr = compute_corr_matrix(df, subset)
        subset_max_abs_corr = max_abs_corr(subset_corr)
        subset_vif_df = compute_vif_table(df, subset)
        max_vif = safe_float(subset_vif_df["vif"].max()) if not subset_vif_df.empty else np.nan

        if pd.notna(subset_max_abs_corr) and subset_max_abs_corr >= config.max_abs_corr_skip:
            results.append(
                {
                    "group": group_name,
                    "endpoint": endpoint,
                    "subset_size": len(subset),
                    "subset_str": " + ".join(subset),
                    "subset_tuple": subset,
                    "penalizer": np.nan,
                    "l1_ratio": np.nan,
                    "cv_mean_cindex": np.nan,
                    "cv_std_cindex": np.nan,
                    "n_rows_used": np.nan,
                    "n_events_used": np.nan,
                    "valid_folds": 0,
                    "median_used_features": 0,
                    "max_vif": max_vif,
                    "max_abs_corr": subset_max_abs_corr,
                    "status": "skipped_extreme_corr",
                }
            )
            continue

        if pd.notna(max_vif) and max_vif > config.max_vif_allowed:
            results.append(
                {
                    "group": group_name,
                    "endpoint": endpoint,
                    "subset_size": len(subset),
                    "subset_str": " + ".join(subset),
                    "subset_tuple": subset,
                    "penalizer": np.nan,
                    "l1_ratio": np.nan,
                    "cv_mean_cindex": np.nan,
                    "cv_std_cindex": np.nan,
                    "n_rows_used": np.nan,
                    "n_events_used": np.nan,
                    "valid_folds": 0,
                    "median_used_features": 0,
                    "max_vif": max_vif,
                    "max_abs_corr": subset_max_abs_corr,
                    "status": "skipped_high_vif",
                }
            )
            continue

        for penalizer in config.penalizers:
            for l1_ratio in config.l1_ratios:
                score_info = cross_validated_subset_score(
                    df=df,
                    endpoint=endpoint,
                    subset=subset,
                    penalizer=penalizer,
                    l1_ratio=l1_ratio,
                    n_splits=config.n_splits,
                    random_state=config.random_state,
                    min_test_events=config.min_test_events,
                    standardize_features=config.standardize_features,
                )

                row = {
                    "group": group_name,
                    "endpoint": endpoint,
                    "subset_size": len(subset),
                    "subset_str": " + ".join(subset),
                    "subset_tuple": subset,
                    "penalizer": penalizer,
                    "l1_ratio": l1_ratio,
                    "cv_mean_cindex": score_info["cv_mean_cindex"],
                    "cv_std_cindex": score_info["cv_std_cindex"],
                    "n_rows_used": score_info["n_rows_used"],
                    "n_events_used": score_info["n_events_used"],
                    "valid_folds": score_info["valid_folds"],
                    "median_used_features": score_info["median_used_features"],
                    "max_vif": max_vif,
                    "max_abs_corr": subset_max_abs_corr,
                    "status": score_info["status"],
                }
                results.append(row)

                if wandb_run is not None:
                    wandb.log(
                        {
                            "group": group_name,
                            "endpoint": endpoint,
                            "subset_size": len(subset),
                            "subset_str": " + ".join(subset),
                            "penalizer": penalizer,
                            "l1_ratio": l1_ratio,
                            "cv_mean_cindex": score_info["cv_mean_cindex"],
                            "cv_std_cindex": score_info["cv_std_cindex"],
                            "n_rows_used": score_info["n_rows_used"],
                            "n_events_used": score_info["n_events_used"],
                            "valid_folds": score_info["valid_folds"],
                            "median_used_features": score_info["median_used_features"],
                            "max_vif": max_vif,
                            "max_abs_corr": subset_max_abs_corr,
                        }
                    )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        summary["status"] = "no_results"
        return summary

    results_df = results_df.sort_values(
        by=["cv_mean_cindex", "subset_size", "penalizer"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    results_df.to_csv(group_out_dir / "all_subset_results.csv", index=False)

    best_valid = results_df[
        results_df["cv_mean_cindex"].notna() & results_df["status"].eq("ok")
    ].copy()

    if best_valid.empty:
        summary["status"] = "no_valid_models"
        return summary

    best_row = best_valid.iloc[0].to_dict()
    best_subset = tuple(str(x) for x in best_row["subset_tuple"])
    best_penalizer = float(best_row["penalizer"])
    best_l1_ratio = float(best_row["l1_ratio"])

    coef_df, full_summary_df, ph_df, fit_status = fit_final_model_and_summary(
        df=df,
        endpoint=endpoint,
        subset=best_subset,
        penalizer=best_penalizer,
        l1_ratio=best_l1_ratio,
        standardize_features=config.standardize_features,
    )

    coef_df.to_csv(group_out_dir / "best_model_coefficients.csv", index=False)
    full_summary_df.to_csv(group_out_dir / "best_model_full_summary.csv", index=False)
    ph_df.to_csv(group_out_dir / "best_model_ph_test.csv", index=False)

    subset_vif_df = compute_vif_table(df, best_subset)
    subset_vif_df.to_csv(group_out_dir / "best_model_vif.csv", index=False)

    plot_top_results(
        results_df=best_valid,
        out_path=group_out_dir / "top_models_barplot.png",
        title=f"Top models | {group_name} | {endpoint}",
        top_n=15,
    )

    summary.update(
        {
            "best_subset": " + ".join(best_subset),
            "best_subset_size": len(best_subset),
            "best_penalizer": best_penalizer,
            "best_l1_ratio": best_l1_ratio,
            "best_cv_mean_cindex": safe_float(best_row["cv_mean_cindex"]),
            "best_cv_std_cindex": safe_float(best_row["cv_std_cindex"]),
            "best_valid_folds": int(best_row["valid_folds"]),
            "best_median_used_features": int(best_row["median_used_features"]),
            "best_max_vif": safe_float(best_row["max_vif"]),
            "best_max_abs_corr": safe_float(best_row["max_abs_corr"]),
            "candidate_pool_size": len(candidate_pool),
            "n_models_tested": int(len(results_df)),
            "fit_status": fit_status,
        }
    )

    pd.DataFrame([summary]).to_csv(group_out_dir / "best_model_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config = ExperimentConfig(
        max_features=args.max_features,
        min_features=args.min_features,
        top_k_univariate=args.top_k_univariate,
        corr_threshold=args.corr_threshold,
        penalizers=tuple(sorted(set(args.penalizers))),
        l1_ratios=tuple(sorted(set(args.l1_ratios))),
        n_splits=args.n_splits,
        min_rows=args.min_rows,
        min_events=args.min_events,
        random_state=args.random_state,
        censor_at_months=args.censor_at_months,
        require_ph_ok=args.require_ph_ok,
        ph_p_threshold=args.ph_p_threshold,
        max_vif_allowed=args.max_vif_allowed,
        target_feature=args.target_feature.strip() if args.target_feature.strip() else None,
        force_include_target=args.force_include_target,
        min_test_events=args.min_test_events,
        max_combinations_per_size=args.max_combinations_per_size,
        min_unique_values=args.min_unique_values,
        max_abs_corr_skip=args.max_abs_corr_skip,
        standardize_features=not args.no_standardize_features,
    )

    input_path = Path(args.input_excel)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logging.info("Loading data from %s", input_path)
    df = pd.read_excel(input_path)
    df = rename_feature_columns(df)
    df = normalize_string_columns(df, ["histology", "idh_status", "grade"])

    missing_features = [feat for feat in FEATURES if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    groups = create_groups(df)
    wandb_run = init_wandb(args, config)
    overall_rows: List[Dict[str, Any]] = []

    for endpoint in args.endpoints:
        time_col = f"{endpoint}.time"
        if endpoint not in df.columns or time_col not in df.columns:
            logging.warning("Skipping %s because required columns are missing.", endpoint)
            continue

        for group_name in args.groups:
            if group_name not in groups:
                logging.warning("Skipping unknown group: %s", group_name)
                continue

            logging.info("Processing | group=%s | endpoint=%s", group_name, endpoint)

            group_df = preprocess_endpoint_df(
                df=groups[group_name],
                endpoint=endpoint,
                censor_at_months=config.censor_at_months,
            )
            group_out_dir = output_dir / sanitize_name(endpoint) / sanitize_name(group_name)

            try:
                summary = evaluate_group_endpoint(
                    df=group_df,
                    group_name=group_name,
                    endpoint=endpoint,
                    config=config,
                    group_out_dir=group_out_dir,
                    wandb_run=wandb_run,
                )
            except Exception as exc:
                logging.exception(
                    "Hard failure for group=%s endpoint=%s: %s",
                    group_name,
                    endpoint,
                    exc,
                )
                summary = {
                    "group": group_name,
                    "endpoint": endpoint,
                    "status": f"hard_failure:{type(exc).__name__}",
                    "n_rows": int(len(group_df)),
                    "n_events": int(group_df[endpoint].sum()) if not group_df.empty else 0,
                }

            overall_rows.append(summary)

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(output_dir / "overall_best_models_summary.csv", index=False)

    if not overall_df.empty and "best_cv_mean_cindex" in overall_df.columns:
        pivot = overall_df.pivot(
            index="group",
            columns="endpoint",
            values="best_cv_mean_cindex",
        )

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            pivot.reindex(index=[g for g in GROUP_ORDER if g in pivot.index]),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=0.3,
            vmax=0.9,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "Best CV mean C-index"},
        )
        plt.title("Best subset-search CV C-index")
        plt.tight_layout()
        plt.savefig(output_dir / "overall_best_cv_cindex_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    if wandb_run is not None:
        summary_path = output_dir / "overall_best_models_summary.csv"
        if summary_path.exists():
            wandb.save(str(summary_path))
        wandb.finish()

    logging.info("Finished. Results saved to: %s", output_dir)
    logging.info(
        "Safer first run example:\n"
        "python 2_2.sanket_multivariate_survival_analysis_v2.py "
        "--groups Oligo IDH-mutant "
        "--endpoints OS "
        "--max_features 3 "
        "--top_k_univariate 5 "
        "--force_include_target "
        "--target_feature AFW "
        "--penalizers 0.5 1.0 2.0 5.0 10.0 "
        "--l1_ratios 0.0 0.5 1.0 "
        "--max_vif_allowed 8 "
        "--max_abs_corr_skip 0.95"
    )


if __name__ == "__main__":
    main()