import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    import seaborn as sns
except Exception as exc:  # pragma: no cover
    raise ImportError("seaborn is required for plotting.") from exc

try:
    import wandb
except Exception:
    wandb = None  # type: ignore

try:
    from pacman.config import DATA_DIR, RESULTS_DIR  # type: ignore

    DEFAULT_INPUT = str(Path(DATA_DIR) / "heatmap_cin_raw_num.xlsx")
    DEFAULT_OUTPUT = str(
        Path(RESULTS_DIR) / "survival" / "incremental_os_oligo_diagnosis"
    )
except Exception:
    DEFAULT_INPUT = "heatmap_cin_raw_num.xlsx"
    DEFAULT_OUTPUT = "incremental_os_oligo_diagnosis"


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

NUMERIC_WARNINGS = (RuntimeWarning, LinAlgWarning, UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why univariate OS C-index is high for Oligo but multivariate "
            "performance drops. The script ranks univariate features, then adds them "
            "one-by-one in fixed order, records exactly where performance drops, and "
            "writes a detailed report with plots."
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
        "--endpoint",
        type=str,
        default="OS",
        choices=["OS", "PFI", "DSS", "DFI"],
        help="Endpoint to analyze",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="Oligo",
        choices=[
            "IDH-wildtype",
            "IDH-mutant",
            "Grade 4 astro",
            "Grade 2&3 astro",
            "Oligo",
            "Grade 2&3 astro + oligo",
        ],
        help="Group to analyze",
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
        help="L2 penalizer for CoxPHFitter",
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.0,
        help="L1 ratio for CoxPHFitter",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Cross-validation folds",
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=20,
        help="Minimum rows required to run the analysis",
    )
    parser.add_argument(
        "--min_events",
        type=int,
        default=10,
        help="Minimum number of events required",
    )
    parser.add_argument(
        "--min_test_events",
        type=int,
        default=1,
        help="Minimum events required in a CV test fold",
    )
    parser.add_argument(
        "--ph_p_threshold",
        type=float,
        default=0.05,
        help="Threshold below which PH test is flagged as violation",
    )
    parser.add_argument(
        "--corr_warning_threshold",
        type=float,
        default=0.80,
        help="Warn when absolute pairwise correlation exceeds this value",
    )
    parser.add_argument(
        "--vif_warning_threshold",
        type=float,
        default=5.0,
        help="Warn when VIF exceeds this value",
    )
    parser.add_argument(
        "--standardize_features",
        action="store_true",
        help="Standardize features before Cox fitting",
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
        default="survival-incremental-diagnosis",
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


def rename_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP)


def normalize_string_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return df


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


def safe_concordance_index(
    times: np.ndarray,
    risk_scores: np.ndarray,
    events: np.ndarray,
) -> float:
    """
    Concordance index wrapper.

    Important:
    lifelines.utils.concordance_index expects larger predicted scores to mean
    *longer survival*. Our inputs are risk scores / partial hazards, where larger
    means *higher risk* and therefore *shorter survival*.

    So we must pass -risk_scores into concordance_index.
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


def compute_feature_availability(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = len(df)

    for feat in features:
        ser = pd.to_numeric(df[feat], errors="coerce")
        rows.append(
            {
                "feature": feat,
                "n_total": total,
                "n_non_missing": int(ser.notna().sum()),
                "pct_non_missing": (
                    float(ser.notna().mean()) if total > 0 else np.nan
                ),
                "n_unique": int(ser.dropna().nunique()),
                "std": safe_float(ser.std()),
            }
        )

    return pd.DataFrame(rows)


def fit_univariate_feature_screen(
    df: pd.DataFrame,
    endpoint: str,
    features: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    ph_p_threshold: float,
    standardize_features: bool,
) -> pd.DataFrame:
    time_col = f"{endpoint}.time"
    rows: List[Dict[str, Any]] = []

    for feat in features:
        sub = df[[time_col, endpoint, feat]].copy()
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        sub = sub.dropna().copy()

        row: Dict[str, Any] = {
            "feature": feat,
            "n_rows": int(len(sub)),
            "n_events": int(sub[endpoint].sum()) if len(sub) > 0 else 0,
            "univariate_cindex_feature_only": np.nan,
            "univariate_cox_in_sample_cindex": np.nan,
            "coef": np.nan,
            "hr": np.nan,
            "p": np.nan,
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

        # Raw feature is treated as risk score: larger feature => larger risk.
        row["univariate_cindex_feature_only"] = safe_concordance_index(
            times=sub[time_col].to_numpy(),
            risk_scores=sub[feat].to_numpy(),
            events=sub[endpoint].to_numpy(),
        )

        fit_df = sub.copy()
        if standardize_features:
            scaler = StandardScaler()
            fit_df[[feat]] = scaler.fit_transform(fit_df[[feat]])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                warnings.filterwarnings("error", category=LinAlgWarning)

                cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                cph.fit(fit_df, duration_col=time_col, event_col=endpoint)

            risk = cph.predict_partial_hazard(fit_df).to_numpy().reshape(-1)
            row["univariate_cox_in_sample_cindex"] = safe_concordance_index(
                times=fit_df[time_col].to_numpy(),
                risk_scores=risk,
                events=fit_df[endpoint].to_numpy(),
            )
            row["coef"] = safe_float(cph.params_[feat])
            row["hr"] = safe_float(np.exp(cph.params_[feat]))
            row["p"] = safe_float(cph.summary.loc[feat, "p"])

            try:
                ph = proportional_hazard_test(cph, fit_df, time_transform="rank")
                ph_df = ph.summary.reset_index()
                if "index" in ph_df.columns:
                    ph_df = ph_df.rename(columns={"index": "feature"})
                ph_series = ph_df.loc[ph_df["feature"] == feat, "p"]
                row["ph_p"] = (
                    safe_float(ph_series.iloc[0]) if not ph_series.empty else np.nan
                )
                row["ph_ok"] = bool(
                    pd.notna(row["ph_p"]) and float(row["ph_p"]) >= ph_p_threshold
                )
            except Exception:
                row["ph_p"] = np.nan
                row["ph_ok"] = False

        except Exception as exc:
            row["status"] = f"failed:{type(exc).__name__}"

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        by=["univariate_cindex_feature_only", "p"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def compute_corr_and_vif(
    df: pd.DataFrame,
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_df = df[list(features)].copy()
    for feat in features:
        num_df[feat] = pd.to_numeric(num_df[feat], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(num_df)
    x_df = pd.DataFrame(x, columns=list(features))

    corr_df = x_df.corr()

    vif_rows: List[Dict[str, Any]] = []
    for i, feat in enumerate(features):
        try:
            vif_val = variance_inflation_factor(x_df.values, i)
        except Exception:
            vif_val = np.nan
        vif_rows.append({"feature": feat, "vif": safe_float(vif_val)})

    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(
        drop=True
    )
    return corr_df, vif_df


def max_abs_offdiag(corr_df: pd.DataFrame) -> float:
    if corr_df.empty or corr_df.shape[1] <= 1:
        return 0.0
    arr = corr_df.abs().to_numpy().copy()
    np.fill_diagonal(arr, np.nan)
    return safe_float(np.nanmax(arr))


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

    x_imp = x_imp.clip(-5.0, 5.0)
    return x_imp


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
    x = prepare_design_matrix(df, subset, standardize_features=standardize_features)

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

    base_df = base_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[time_col, endpoint]
    )
    base_df = base_df.reset_index(drop=True)

    if len(base_df) < max(10, n_splits):
        return {
            "cv_mean_cindex": np.nan,
            "cv_std_cindex": np.nan,
            "valid_folds": 0,
            "status": "too_few_rows",
            "fold_scores": [],
        }

    if base_df[endpoint].sum() == 0 or base_df[endpoint].sum() == len(base_df):
        return {
            "cv_mean_cindex": np.nan,
            "cv_std_cindex": np.nan,
            "valid_folds": 0,
            "status": "invalid_event_distribution",
            "fold_scores": [],
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []
    fold_statuses: List[str] = []

    for train_idx, test_idx in kf.split(base_df):
        train_df = base_df.iloc[train_idx].copy()
        test_df = base_df.iloc[test_idx].copy()

        if train_df[endpoint].sum() == 0 or train_df[endpoint].sum() == len(train_df):
            fold_statuses.append("invalid_train_event_distribution")
            continue

        if (
            test_df[endpoint].sum() < min_test_events
            or test_df[endpoint].sum() == len(test_df)
        ):
            fold_statuses.append("invalid_test_event_distribution")
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
            fold_statuses.append("imputation_failed")
            continue

        keep_cols: List[str] = []
        for feat in subset:
            std_val = safe_float(train_x[feat].std())
            if pd.notna(std_val) and std_val > 0:
                keep_cols.append(feat)

        if not keep_cols:
            fold_statuses.append("all_train_zero_variance")
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
                fold_statuses.append("scaling_failed")
                continue

        train_x = train_x.clip(-5.0, 5.0)
        test_x = test_x.clip(-5.0, 5.0)

        train_fit = pd.concat([train_core, train_x], axis=1).dropna().copy()
        test_fit = pd.concat([test_core, test_x], axis=1).dropna().copy()

        if len(train_fit) < 10:
            fold_statuses.append("too_few_train_rows")
            continue

        if len(test_fit) < 3:
            fold_statuses.append("too_few_test_rows")
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
                fold_statuses.append("ok")
            else:
                fold_statuses.append("nan_cindex")
        except Exception as exc:
            fold_statuses.append(f"failed:{type(exc).__name__}")

    status = "ok" if fold_scores else f"all_folds_failed:{'|'.join(fold_statuses)}"
    return {
        "cv_mean_cindex": float(np.mean(fold_scores)) if fold_scores else np.nan,
        "cv_std_cindex": (
            float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else np.nan
        ),
        "valid_folds": int(len(fold_scores)),
        "status": status,
        "fold_scores": fold_scores,
    }


def ph_test_for_model(
    cph: CoxPHFitter,
    fit_df: pd.DataFrame,
) -> pd.DataFrame:
    try:
        ph = proportional_hazard_test(cph, fit_df, time_transform="rank")
        ph_df = ph.summary.reset_index()
        if "index" in ph_df.columns:
            ph_df = ph_df.rename(columns={"index": "feature"})
        return ph_df
    except Exception as exc:
        return pd.DataFrame([{"status": f"failed:{type(exc).__name__}"}])


def subset_case_stats(
    endpoint_df: pd.DataFrame,
    endpoint: str,
    subset: Sequence[str],
) -> Dict[str, Any]:
    time_col = f"{endpoint}.time"
    endpoint_only = endpoint_df.dropna(subset=[endpoint, time_col]).copy()
    complete_df = endpoint_df.dropna(subset=[endpoint, time_col] + list(subset)).copy()

    n_endpoint_only = len(endpoint_only)
    n_complete = len(complete_df)
    n_events_complete = int(complete_df[endpoint].sum()) if n_complete > 0 else 0

    return {
        "n_rows_after_endpoint_filter": int(n_endpoint_only),
        "n_rows_complete_case_subset": int(n_complete),
        "complete_case_loss_count": int(n_endpoint_only - n_complete),
        "complete_case_loss_pct": (
            float((n_endpoint_only - n_complete) / n_endpoint_only)
            if n_endpoint_only > 0
            else np.nan
        ),
        "n_events_complete_case_subset": int(n_events_complete),
        "epv_subset_complete_case": (
            float(n_events_complete / len(subset)) if len(subset) > 0 else np.nan
        ),
    }


def coefficient_table_from_model(
    cph: Optional[CoxPHFitter],
    requested_subset: Sequence[str],
) -> pd.DataFrame:
    out = pd.DataFrame({"feature": list(requested_subset)})
    out["coef"] = np.nan
    out["hr"] = np.nan
    out["p"] = np.nan
    out["used_in_fit"] = False

    if cph is None:
        return out

    summary_df = cph.summary.reset_index()
    if "covariate" in summary_df.columns:
        summary_df = summary_df.rename(columns={"covariate": "feature"})
    elif "index" in summary_df.columns:
        summary_df = summary_df.rename(columns={"index": "feature"})

    keep_cols = {"feature", "coef", "exp(coef)", "p"}
    if not keep_cols.issubset(summary_df.columns):
        return out

    fit_coef = summary_df[["feature", "coef", "exp(coef)", "p"]].copy()
    fit_coef = fit_coef.rename(columns={"exp(coef)": "hr"})
    fit_coef["used_in_fit"] = True

    out = out.drop(columns=["coef", "hr", "p", "used_in_fit"]).merge(
        fit_coef, on="feature", how="left"
    )
    out["used_in_fit"] = out["used_in_fit"].fillna(False)
    return out


def evaluate_progression(
    endpoint_df: pd.DataFrame,
    endpoint: str,
    ranked_features: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    n_splits: int,
    min_test_events: int,
    random_state: int,
    standardize_features: bool,
    corr_warning_threshold: float,
    vif_warning_threshold: float,
    ph_p_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    progression_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []

    for k in range(1, len(ranked_features) + 1):
        subset = list(ranked_features[:k])

        case_stats = subset_case_stats(
            endpoint_df=endpoint_df,
            endpoint=endpoint,
            subset=subset,
        )

        complete_case_df = endpoint_df.dropna(
            subset=[endpoint, f"{endpoint}.time"] + subset
        ).copy()
        corr_df = pd.DataFrame()
        vif_df = pd.DataFrame()
        max_abs_corr = np.nan
        max_vif = np.nan

        if len(complete_case_df) >= 2:
            corr_df, vif_df = compute_corr_and_vif(complete_case_df, subset)
            max_abs_corr = max_abs_offdiag(corr_df)
            max_vif = safe_float(vif_df["vif"].max()) if not vif_df.empty else np.nan

        cph, fit_df, fit_status = fit_multivariate_cox(
            df=endpoint_df,
            endpoint=endpoint,
            subset=subset,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            standardize_features=standardize_features,
        )

        coef_df = coefficient_table_from_model(cph, subset)
        for _, row in coef_df.iterrows():
            coefficient_rows.append(
                {
                    "step": k,
                    "subset_size": k,
                    "subset": " + ".join(subset),
                    "feature": row["feature"],
                    "coef": safe_float(row["coef"]),
                    "hr": safe_float(row["hr"]),
                    "p": safe_float(row["p"]),
                    "used_in_fit": bool(row["used_in_fit"]),
                }
            )

        in_sample_cindex = np.nan
        ph_df = pd.DataFrame()
        if cph is not None and not fit_df.empty:
            risk = cph.predict_partial_hazard(fit_df).to_numpy().reshape(-1)
            in_sample_cindex = safe_concordance_index(
                times=fit_df[f"{endpoint}.time"].to_numpy(),
                risk_scores=risk,
                events=fit_df[endpoint].to_numpy(),
            )
            ph_df = ph_test_for_model(cph, fit_df)

        ph_violations = 0
        if not ph_df.empty and "p" in ph_df.columns and "feature" in ph_df.columns:
            ph_violations = int(
                (pd.to_numeric(ph_df["p"], errors="coerce") < ph_p_threshold).sum()
            )

        cv_info = fit_cv_cox_for_subset(
            df=endpoint_df,
            endpoint=endpoint,
            subset=subset,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            n_splits=n_splits,
            min_test_events=min_test_events,
            random_state=random_state,
            standardize_features=standardize_features,
        )

        progression_rows.append(
            {
                "step": k,
                "subset_size": k,
                "subset": " + ".join(subset),
                "added_feature": subset[-1],
                "fit_status": fit_status,
                "in_sample_cindex": in_sample_cindex,
                "cv_mean_cindex": cv_info["cv_mean_cindex"],
                "cv_std_cindex": cv_info["cv_std_cindex"],
                "cv_valid_folds": cv_info["valid_folds"],
                "cv_status": cv_info["status"],
                "n_rows_fit_df": int(len(fit_df)),
                "n_events_fit_df": int(fit_df[endpoint].sum()) if not fit_df.empty else 0,
                "ph_violations_count": ph_violations,
                "max_abs_corr": max_abs_corr,
                "max_vif": max_vif,
                "corr_warning": bool(
                    pd.notna(max_abs_corr) and max_abs_corr >= corr_warning_threshold
                ),
                "vif_warning": bool(
                    pd.notna(max_vif) and max_vif >= vif_warning_threshold
                ),
                **case_stats,
            }
        )

    progression_df = pd.DataFrame(progression_rows)
    coef_progression_df = pd.DataFrame(coefficient_rows)

    if not progression_df.empty:
        progression_df["delta_in_sample_from_prev"] = progression_df["in_sample_cindex"].diff()
        progression_df["delta_cv_from_prev"] = progression_df["cv_mean_cindex"].diff()
        progression_df["delta_fit_rows_from_prev"] = progression_df["n_rows_fit_df"].diff()
        progression_df["delta_complete_case_loss_pct_from_prev"] = progression_df[
            "complete_case_loss_pct"
        ].diff()

    return progression_df, coef_progression_df


def infer_root_causes(
    progression_df: pd.DataFrame,
    feature_screen_df: pd.DataFrame,
    corr_warning_threshold: float,
    vif_warning_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if progression_df.empty:
        return pd.DataFrame(rows)

    for _, row in progression_df.iterrows():
        if int(row["step"]) == 1:
            continue

        reasons: List[str] = []
        evidence: List[str] = []

        delta_cv = safe_float(row.get("delta_cv_from_prev"))
        delta_in = safe_float(row.get("delta_in_sample_from_prev"))
        delta_rows = safe_float(row.get("delta_fit_rows_from_prev"))
        cc_loss = safe_float(row.get("complete_case_loss_pct"))
        delta_cc_loss = safe_float(row.get("delta_complete_case_loss_pct_from_prev"))
        max_corr = safe_float(row.get("max_abs_corr"))
        max_vif = safe_float(row.get("max_vif"))
        epv = safe_float(row.get("epv_subset_complete_case"))
        ph_viol = safe_float(row.get("ph_violations_count"))

        if pd.notna(delta_cv) and delta_cv < -0.02:
            reasons.append("cv_drop")
            evidence.append(f"CV C-index dropped by {delta_cv:.3f}")

        if pd.notna(delta_in) and delta_in < -0.02:
            reasons.append("in_sample_drop")
            evidence.append(f"In-sample C-index dropped by {delta_in:.3f}")

        if pd.notna(max_corr) and max_corr >= corr_warning_threshold:
            reasons.append("high_correlation")
            evidence.append(f"max |corr| = {max_corr:.3f}")

        if pd.notna(max_vif) and max_vif >= vif_warning_threshold:
            reasons.append("high_vif")
            evidence.append(f"max VIF = {max_vif:.2f}")

        if pd.notna(delta_rows) and delta_rows < 0:
            reasons.append("sample_loss")
            evidence.append(f"fit rows dropped by {delta_rows:.0f}")

        if pd.notna(cc_loss) and cc_loss > 0.20:
            reasons.append("high_complete_case_loss")
            evidence.append(f"complete-case loss = {cc_loss:.1%}")

        if pd.notna(delta_cc_loss) and delta_cc_loss > 0.05:
            reasons.append("stepwise_missingness_jump")
            evidence.append(f"complete-case loss increased by {delta_cc_loss:.1%}")

        if pd.notna(epv) and epv < 10:
            reasons.append("low_epv")
            evidence.append(f"EPV = {epv:.2f}")

        if pd.notna(ph_viol) and ph_viol > 0:
            reasons.append("ph_instability")
            evidence.append(f"PH violations = {int(ph_viol)}")

        added_feature = str(row["added_feature"])
        feat_row = feature_screen_df.loc[feature_screen_df["feature"] == added_feature]
        if not feat_row.empty:
            feat_p = safe_float(feat_row.iloc[0]["p"])
            feat_ci = safe_float(feat_row.iloc[0]["univariate_cindex_feature_only"])
            if pd.notna(feat_p) and feat_p > 0.10:
                reasons.append("weak_added_feature")
                evidence.append(f"added feature univariate p = {feat_p:.3f}")
            if pd.notna(feat_ci) and feat_ci < 0.55:
                reasons.append("weak_univariate_signal")
                evidence.append(f"added feature univariate C-index = {feat_ci:.3f}")

        rows.append(
            {
                "step": int(row["step"]),
                "subset": row["subset"],
                "added_feature": added_feature,
                "root_cause_flags": ", ".join(reasons) if reasons else "none_obvious",
                "evidence": " | ".join(evidence) if evidence else "none",
            }
        )

    return pd.DataFrame(rows)


def save_barplot_univariate(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if df.empty:
        return

    plot_df = df.sort_values("univariate_cindex_feature_only", ascending=True)
    plt.figure(figsize=(8, max(4, 0.45 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["univariate_cindex_feature_only"])
    plt.xlabel("Univariate C-index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_lineplot_progression(
    progression_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if progression_df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(
        progression_df["step"],
        progression_df["in_sample_cindex"],
        marker="o",
        label="In-sample Cox C-index",
    )
    plt.plot(
        progression_df["step"],
        progression_df["cv_mean_cindex"],
        marker="o",
        label="CV mean Cox C-index",
    )
    plt.xticks(
        progression_df["step"],
        progression_df["added_feature"],
        rotation=45,
        ha="right",
    )
    plt.ylabel("C-index")
    plt.xlabel("Feature added at each step")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_lineplot_sample_size(
    progression_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if progression_df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(
        progression_df["step"],
        progression_df["n_rows_complete_case_subset"],
        marker="o",
        label="Complete-case rows",
    )
    plt.plot(
        progression_df["step"],
        progression_df["n_rows_fit_df"],
        marker="o",
        label="Rows used in Cox fit",
    )
    plt.xticks(
        progression_df["step"],
        progression_df["added_feature"],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Number of cases")
    plt.xlabel("Feature added at each step")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_lineplot_collinearity(
    progression_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if progression_df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(
        progression_df["step"],
        progression_df["max_abs_corr"],
        marker="o",
        label="Max |corr|",
    )
    plt.plot(
        progression_df["step"],
        progression_df["max_vif"],
        marker="o",
        label="Max VIF",
    )
    plt.xticks(
        progression_df["step"],
        progression_df["added_feature"],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Collinearity metric")
    plt.xlabel("Feature added at each step")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_corr_heatmap(
    corr_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if corr_df.empty:
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        square=True,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_coefficient_heatmap(
    coef_progression_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if coef_progression_df.empty:
        return

    pivot = coef_progression_df.pivot(index="feature", columns="step", values="coef")
    if pivot.empty:
        return

    plt.figure(figsize=(10, max(4, 0.5 * len(pivot.index))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_risk_boxplot(
    endpoint_df: pd.DataFrame,
    endpoint: str,
    ranked_features: Sequence[str],
    penalizer: float,
    l1_ratio: float,
    standardize_features: bool,
    out_path: Path,
    title: str,
) -> None:
    if not ranked_features:
        return

    cph, fit_df, status = fit_multivariate_cox(
        df=endpoint_df,
        endpoint=endpoint,
        subset=ranked_features,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        standardize_features=standardize_features,
    )
    if status != "ok" or cph is None or fit_df.empty:
        return

    risk = cph.predict_partial_hazard(fit_df).to_numpy().reshape(-1)
    plot_df = pd.DataFrame(
        {
            "risk_score": risk,
            "event_status": fit_df[endpoint].astype(int).map(
                {0: "Censored", 1: "Event"}
            ),
        }
    )

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=plot_df, x="event_status", y="risk_score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_markdown_report(
    args: argparse.Namespace,
    global_summary: Dict[str, Any],
    feature_screen_df: pd.DataFrame,
    progression_df: pd.DataFrame,
    root_cause_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    best_feature = (
        str(feature_screen_df.iloc[0]["feature"]) if not feature_screen_df.empty else "NA"
    )
    best_uni = (
        safe_float(feature_screen_df.iloc[0]["univariate_cindex_feature_only"])
        if not feature_screen_df.empty
        else np.nan
    )

    cv_drop_rows = progression_df[
        progression_df["delta_cv_from_prev"].notna()
        & (progression_df["delta_cv_from_prev"] < 0)
    ].copy()
    first_cv_drop_step = (
        int(cv_drop_rows.iloc[0]["step"]) if not cv_drop_rows.empty else None
    )
    first_cv_drop_subset = (
        str(cv_drop_rows.iloc[0]["subset"]) if not cv_drop_rows.empty else "NA"
    )

    major_drop_rows = progression_df[
        progression_df["delta_cv_from_prev"].notna()
        & (progression_df["delta_cv_from_prev"] <= -0.02)
    ].copy()
    first_major_drop_step = (
        int(major_drop_rows.iloc[0]["step"]) if not major_drop_rows.empty else None
    )
    first_major_drop_subset = (
        str(major_drop_rows.iloc[0]["subset"]) if not major_drop_rows.empty else "NA"
    )

    best_cv_row = (
        progression_df.sort_values(
            "cv_mean_cindex",
            ascending=False,
            na_position="last",
        ).iloc[0]
        if not progression_df.empty
        else None
    )

    lines: List[str] = []
    lines.append(f"# Incremental Survival Diagnosis Report: {args.group} / {args.endpoint}")
    lines.append("")
    lines.append("## 1. Setup")
    lines.append("")
    lines.append(f"- Input file: `{args.input_excel}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Group: `{args.group}`")
    lines.append(f"- Endpoint: `{args.endpoint}`")
    lines.append(f"- Penalizer: `{args.penalizer}`")
    lines.append(f"- L1 ratio: `{args.l1_ratio}`")
    lines.append(f"- CV folds: `{args.n_splits}`")
    lines.append(f"- Standardize features: `{args.standardize_features}`")
    lines.append("")
    lines.append("## 2. Cohort summary")
    lines.append("")
    lines.append(f"- Raw group size: **{global_summary['n_raw_group']}**")
    lines.append(
        f"- Rows after endpoint cleanup: **{global_summary['n_after_endpoint_cleanup']}**"
    )
    lines.append(f"- Events after endpoint cleanup: **{global_summary['n_events']}**")
    lines.append(f"- Event rate: **{global_summary['event_rate']:.3f}**")
    lines.append(
        f"- Complete-case rows using all 8 features: **{global_summary['n_complete_case_all_8_features']}**"
    )
    lines.append(
        f"- Complete-case loss using all 8 features: **{global_summary['complete_case_loss_all_8_features_pct']:.1%}**"
    )
    lines.append(
        f"- EPV using all 8 features: **{global_summary['epv_all_8_features']:.2f}**"
    )
    lines.append("")
    lines.append("## 3. Best univariate feature")
    lines.append("")
    lines.append(f"- Best univariate feature: **{best_feature}**")
    lines.append(f"- Best univariate feature-only C-index: **{best_uni:.4f}**")
    lines.append("")
    lines.append("## 4. Where multivariate performance starts dropping")
    lines.append("")
    lines.append(
        f"- First CV drop step: **{first_cv_drop_step}** | subset: **{first_cv_drop_subset}**"
    )
    lines.append(
        f"- First major CV drop step (<= -0.02): **{first_major_drop_step}** | subset: **{first_major_drop_subset}**"
    )
    if best_cv_row is not None:
        lines.append(
            f"- Best CV subset in this fixed progression: **{best_cv_row['subset']}** "
            f"(CV mean C-index = **{safe_float(best_cv_row['cv_mean_cindex']):.4f}**)"
        )
    lines.append("")
    lines.append("## 5. Root-cause summary")
    lines.append("")
    if root_cause_df.empty:
        lines.append("- No obvious root-cause flags were detected.")
    else:
        for _, row in root_cause_df.iterrows():
            lines.append(
                f"- Step **{int(row['step'])}** | added **{row['added_feature']}** | "
                f"flags: **{row['root_cause_flags']}** | evidence: {row['evidence']}"
            )
    lines.append("")
    lines.append("## 6. Recommended interpretation")
    lines.append("")
    lines.append(
        "- If CV drops while in-sample stays flat or rises, the new feature is likely adding instability or overfitting."
    )
    lines.append(
        "- If fit rows drop sharply after adding a feature, missingness is likely driving the drop."
    )
    lines.append(
        "- If max |corr| or VIF rises at the same step that CV falls, collinearity is a likely cause."
    )
    lines.append(
        "- If coefficients flip sign or become unstable across steps, the features are competing for the same signal."
    )
    lines.append(
        "- If EPV falls too low as subset size grows, the model is likely underpowered for that subset size."
    )
    lines.append("")
    lines.append("## 7. Output files")
    lines.append("")
    lines.append("- `feature_availability.csv`")
    lines.append("- `univariate_feature_screen.csv`")
    lines.append("- `progression_results.csv`")
    lines.append("- `progression_coefficients.csv`")
    lines.append("- `root_cause_flags.csv`")
    lines.append("- `global_correlation.csv`")
    lines.append("- `global_vif.csv`")
    lines.append("- `plots/*.png`")
    lines.append("")
    lines.append("## 8. Next step")
    lines.append("")
    lines.append(
        "- Keep the best univariate feature fixed, then stop at the last step before CV drops materially."
    )
    lines.append(
        "- Prefer subsets with low sample loss, low max VIF, and stable coefficients."
    )
    lines.append(
        "- Use the fixed progression result here first, then compare it against your broader subset-search script."
    )
    lines.append("")

    return "\n".join(lines)


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

    ensure_dir(output_dir)
    ensure_dir(plots_dir)

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
        raise ValueError(
            f"Missing required feature columns after renaming: {missing_features}"
        )

    groups = create_groups(df)
    group_df_raw = groups[args.group].copy()
    endpoint_df = preprocess_endpoint_df(
        df=group_df_raw,
        endpoint=args.endpoint,
        censor_at_months=args.censor_at_months,
    )

    n_raw_group = len(group_df_raw)
    n_after_endpoint_cleanup = len(endpoint_df)
    n_events = int(endpoint_df[args.endpoint].sum()) if not endpoint_df.empty else 0
    n_complete_case_all_8 = len(
        endpoint_df.dropna(subset=[args.endpoint, f"{args.endpoint}.time"] + FEATURES)
    )

    global_summary: Dict[str, Any] = {
        "group": args.group,
        "endpoint": args.endpoint,
        "n_raw_group": int(n_raw_group),
        "n_after_endpoint_cleanup": int(n_after_endpoint_cleanup),
        "n_events": int(n_events),
        "event_rate": (
            float(n_events / n_after_endpoint_cleanup)
            if n_after_endpoint_cleanup > 0
            else np.nan
        ),
        "n_complete_case_all_8_features": int(n_complete_case_all_8),
        "complete_case_loss_all_8_features_pct": (
            float(
                (n_after_endpoint_cleanup - n_complete_case_all_8)
                / n_after_endpoint_cleanup
            )
            if n_after_endpoint_cleanup > 0
            else np.nan
        ),
        "epv_all_8_features": (
            float(n_events / len(FEATURES)) if len(FEATURES) > 0 else np.nan
        ),
    }

    pd.DataFrame([global_summary]).to_csv(output_dir / "global_summary.csv", index=False)

    if n_after_endpoint_cleanup < args.min_rows:
        raise ValueError(
            f"Too few rows after endpoint cleanup: "
            f"{n_after_endpoint_cleanup} < {args.min_rows}"
        )
    if n_events < args.min_events:
        raise ValueError(f"Too few events: {n_events} < {args.min_events}")

    availability_df = compute_feature_availability(endpoint_df, FEATURES)
    availability_df.to_csv(output_dir / "feature_availability.csv", index=False)

    feature_screen_df = fit_univariate_feature_screen(
        df=endpoint_df,
        endpoint=args.endpoint,
        features=FEATURES,
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
        ph_p_threshold=args.ph_p_threshold,
        standardize_features=args.standardize_features,
    )
    feature_screen_df.to_csv(output_dir / "univariate_feature_screen.csv", index=False)

    ranked_features = feature_screen_df.loc[
        feature_screen_df["status"].eq("ok"), "feature"
    ].tolist()
    if not ranked_features:
        raise ValueError("No valid ranked features found for incremental progression.")

    logging.info("Ranked feature order: %s", " -> ".join(ranked_features))

    complete_global_df = endpoint_df.dropna(subset=FEATURES).copy()
    if len(complete_global_df) >= 2:
        global_corr_df, global_vif_df = compute_corr_and_vif(
            complete_global_df,
            FEATURES,
        )
    else:
        global_corr_df = pd.DataFrame(index=FEATURES, columns=FEATURES, dtype=float)
        global_vif_df = pd.DataFrame(
            {"feature": FEATURES, "vif": [np.nan] * len(FEATURES)}
        )

    global_corr_df.to_csv(output_dir / "global_correlation.csv")
    global_vif_df.to_csv(output_dir / "global_vif.csv", index=False)

    progression_df, coef_progression_df = evaluate_progression(
        endpoint_df=endpoint_df,
        endpoint=args.endpoint,
        ranked_features=ranked_features,
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
        n_splits=args.n_splits,
        min_test_events=args.min_test_events,
        random_state=args.random_state,
        standardize_features=args.standardize_features,
        corr_warning_threshold=args.corr_warning_threshold,
        vif_warning_threshold=args.vif_warning_threshold,
        ph_p_threshold=args.ph_p_threshold,
    )

    progression_df.to_csv(output_dir / "progression_results.csv", index=False)
    coef_progression_df.to_csv(output_dir / "progression_coefficients.csv", index=False)

    root_cause_df = infer_root_causes(
        progression_df=progression_df,
        feature_screen_df=feature_screen_df,
        corr_warning_threshold=args.corr_warning_threshold,
        vif_warning_threshold=args.vif_warning_threshold,
    )
    root_cause_df.to_csv(output_dir / "root_cause_flags.csv", index=False)

    save_barplot_univariate(
        df=feature_screen_df,
        out_path=plots_dir / "univariate_cindex_barplot.png",
        title=f"Univariate ranking | {args.group} | {args.endpoint}",
    )
    save_lineplot_progression(
        progression_df=progression_df,
        out_path=plots_dir / "progression_cindex.png",
        title=f"Incremental C-index progression | {args.group} | {args.endpoint}",
    )
    save_lineplot_sample_size(
        progression_df=progression_df,
        out_path=plots_dir / "progression_sample_size.png",
        title=f"Sample-size progression | {args.group} | {args.endpoint}",
    )
    save_lineplot_collinearity(
        progression_df=progression_df,
        out_path=plots_dir / "progression_collinearity.png",
        title=f"Collinearity progression | {args.group} | {args.endpoint}",
    )
    save_corr_heatmap(
        corr_df=global_corr_df,
        out_path=plots_dir / "global_correlation_heatmap.png",
        title=f"Global feature correlation | {args.group} | {args.endpoint}",
    )
    save_coefficient_heatmap(
        coef_progression_df=coef_progression_df,
        out_path=plots_dir / "coefficient_drift_heatmap.png",
        title=f"Coefficient drift across steps | {args.group} | {args.endpoint}",
    )
    save_risk_boxplot(
        endpoint_df=endpoint_df,
        endpoint=args.endpoint,
        ranked_features=ranked_features,
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
        standardize_features=args.standardize_features,
        out_path=plots_dir / "final_subset_risk_boxplot.png",
        title="Risk score by event status | final fixed progression subset",
    )

    report_md = build_markdown_report(
        args=args,
        global_summary=global_summary,
        feature_screen_df=feature_screen_df,
        progression_df=progression_df,
        root_cause_df=root_cause_df,
        output_dir=output_dir,
    )
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    wandb_run = init_wandb(args)
    if wandb_run is not None:
        wandb.log(
            {
                "n_raw_group": global_summary["n_raw_group"],
                "n_after_endpoint_cleanup": global_summary["n_after_endpoint_cleanup"],
                "n_events": global_summary["n_events"],
                "event_rate": global_summary["event_rate"],
                "complete_case_loss_all_8_features_pct": global_summary[
                    "complete_case_loss_all_8_features_pct"
                ],
                "best_univariate_cindex": safe_float(
                    feature_screen_df.iloc[0]["univariate_cindex_feature_only"]
                )
                if not feature_screen_df.empty
                else np.nan,
                "best_univariate_feature": (
                    str(feature_screen_df.iloc[0]["feature"])
                    if not feature_screen_df.empty
                    else ""
                ),
                "best_progression_cv_mean_cindex": safe_float(
                    progression_df["cv_mean_cindex"].max()
                )
                if not progression_df.empty
                else np.nan,
            }
        )

        csv_files = [
            "global_summary.csv",
            "feature_availability.csv",
            "univariate_feature_screen.csv",
            "global_correlation.csv",
            "global_vif.csv",
            "progression_results.csv",
            "progression_coefficients.csv",
            "root_cause_flags.csv",
            "report.md",
        ]
        for fname in csv_files:
            path = output_dir / fname
            if path.exists():
                wandb.save(str(path))

        for plot_path in plots_dir.glob("*.png"):
            wandb.save(str(plot_path))

        wandb.finish()

    logging.info("Finished. Results saved to: %s", output_dir)
    logging.info(
        "Example:\n"
        "python sanket_incremental_os_oligo_diagnosis.py "
        "--group Oligo "
        "--endpoint OS "
        "--input_excel \"%s\" "
        "--output_dir \"%s\" "
        "--penalizer 0.1 "
        "--l1_ratio 0.0 "
        "--n_splits 5 "
        "--standardize_features",
        args.input_excel,
        args.output_dir,
    )


if __name__ == "__main__":
    main()