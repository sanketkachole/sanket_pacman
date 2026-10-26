import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Optional pacman config fallback
try:
    from pacman.config import DATA_DIR, RESULTS_DIR  # type: ignore

    DEFAULT_INPUT = str(Path(DATA_DIR) / "heatmap_cin_raw_num.xlsx")
    DEFAULT_OUTPUT = str(Path(RESULTS_DIR) / "survival" / "multivariate_diagnostics")
except Exception:
    DEFAULT_INPUT = "heatmap_cin_raw_num.xlsx"
    DEFAULT_OUTPUT = "multivariate_diagnostics"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end diagnostics for univariate vs multivariate survival analysis."
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
        help="Directory to save all diagnostics.",
    )
    parser.add_argument(
        "--censor_at_months",
        type=int,
        default=120,
        help="Administrative censoring time in months.",
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=10,
        help="Minimum rows required to run a group-endpoint analysis.",
    )
    parser.add_argument(
        "--min_events",
        type=int,
        default=5,
        help="Minimum number of events required to run a group-endpoint analysis.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="K folds for cross-validation.",
    )
    parser.add_argument(
        "--penalizer",
        type=float,
        default=0.1,
        help="Penalizer for CoxPHFitter.",
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.0,
        help="L1 ratio for CoxPHFitter.",
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
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_string_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return df


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
    out[time_col] = (out[time_col] / 30.4).astype(float)

    if censor_at_months > 0:
        mask = out[time_col] > censor_at_months
        out.loc[mask, endpoint] = 0
        out.loc[mask, time_col] = float(censor_at_months)

    return out


def compute_feature_availability(
    group_df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    rows = []
    n_total = len(group_df)

    for feat in features:
        n_non_missing = int(pd.to_numeric(group_df[feat], errors="coerce").notna().sum())
        rows.append(
            {
                "feature": feat,
                "n_total": n_total,
                "n_non_missing": n_non_missing,
                "pct_non_missing": (n_non_missing / n_total) if n_total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_complete_case_loss(
    group_df: pd.DataFrame,
    endpoint: str,
    features: List[str],
) -> Dict[str, float]:
    time_col = f"{endpoint}.time"
    n_after_endpoint = len(group_df.dropna(subset=[endpoint, time_col]))
    n_complete_case = len(group_df.dropna(subset=[endpoint, time_col] + features))

    return {
        "n_after_endpoint_filter": n_after_endpoint,
        "n_complete_case_all_features": n_complete_case,
        "complete_case_loss_count": n_after_endpoint - n_complete_case,
        "complete_case_loss_pct": (
            (n_after_endpoint - n_complete_case) / n_after_endpoint
            if n_after_endpoint > 0
            else np.nan
        ),
    }


def compute_univariate_cindex_table(
    group_df: pd.DataFrame,
    endpoint: str,
    features: List[str],
) -> pd.DataFrame:
    time_col = f"{endpoint}.time"
    rows = []

    for feat in features:
        sub = group_df[[time_col, endpoint, feat]].copy()
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        sub = sub.dropna()

        ci = np.nan
        if len(sub) >= 3 and 0 < sub[endpoint].sum() < len(sub):
            ci = safe_concordance_index(
                times=sub[time_col].to_numpy(),
                risk_scores=-sub[feat].to_numpy(),
                events=sub[endpoint].to_numpy(),
            )

        rows.append(
            {
                "feature": feat,
                "n_used": len(sub),
                "n_events": int(sub[endpoint].sum()) if len(sub) > 0 else 0,
                "univariate_cindex": ci,
            }
        )

    return pd.DataFrame(rows).sort_values("univariate_cindex", ascending=False)


def compute_corr_and_vif(
    group_df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_df = group_df[features].copy()
    for feat in features:
        num_df[feat] = pd.to_numeric(num_df[feat], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(num_df)
    X_df = pd.DataFrame(X, columns=features)

    corr_df = X_df.corr()

    vif_rows = []
    for i, feat in enumerate(features):
        try:
            vif_value = variance_inflation_factor(X_df.values, i)
        except Exception:
            vif_value = np.nan
        vif_rows.append({"feature": feat, "vif": vif_value})

    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    return corr_df, vif_df


def save_corr_heatmap(corr_df: pd.DataFrame, out_path: Path, title: str) -> None:
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


def fit_univariate_cox(
    group_df: pd.DataFrame,
    endpoint: str,
    features: List[str],
    penalizer: float,
    l1_ratio: float,
) -> pd.DataFrame:
    time_col = f"{endpoint}.time"
    rows = []

    for feat in features:
        sub = group_df[[time_col, endpoint, feat]].copy()
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        sub = sub.dropna().copy()

        result = {
            "feature": feat,
            "n_used": len(sub),
            "n_events": int(sub[endpoint].sum()) if len(sub) > 0 else 0,
            "coef": np.nan,
            "exp_coef_hr": np.nan,
            "p": np.nan,
            "cindex_in_sample": np.nan,
            "status": "ok",
        }

        if len(sub) < 10:
            result["status"] = "too_few_rows"
            rows.append(result)
            continue

        if sub[endpoint].sum() == 0 or sub[endpoint].sum() == len(sub):
            result["status"] = "invalid_event_distribution"
            rows.append(result)
            continue

        try:
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            cph.fit(sub, duration_col=time_col, event_col=endpoint)
            risk = cph.predict_partial_hazard(sub).to_numpy().reshape(-1)
            result["coef"] = float(cph.params_[feat])
            result["exp_coef_hr"] = float(np.exp(cph.params_[feat]))
            result["p"] = float(cph.summary.loc[feat, "p"])
            result["cindex_in_sample"] = safe_concordance_index(
                times=sub[time_col].to_numpy(),
                risk_scores=risk,
                events=sub[endpoint].to_numpy(),
            )
        except Exception as exc:
            result["status"] = f"failed: {type(exc).__name__}"

        rows.append(result)

    return pd.DataFrame(rows).sort_values("cindex_in_sample", ascending=False)


def empty_multivariate_coef_df(features: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": features,
            "coef": [np.nan] * len(features),
            "exp_coef_hr": [np.nan] * len(features),
            "p": [np.nan] * len(features),
        }
    )


def fit_multivariate_cox_all_features(
    group_df: pd.DataFrame,
    endpoint: str,
    features: List[str],
    penalizer: float,
    l1_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    time_col = f"{endpoint}.time"
    sub = group_df[[time_col, endpoint] + features].copy()

    for feat in features:
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
    sub = sub.dropna().copy()

    coef_df = empty_multivariate_coef_df(features)
    summary_df = pd.DataFrame()
    status = "ok"

    if len(sub) < 10:
        return coef_df, summary_df, "too_few_rows"

    n_events = int(sub[endpoint].sum())
    if n_events == 0 or n_events == len(sub):
        return coef_df, summary_df, "invalid_event_distribution"

    try:
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        cph.fit(sub, duration_col=time_col, event_col=endpoint)

        summary_df = cph.summary.reset_index()
        if "covariate" in summary_df.columns:
            summary_df = summary_df.rename(columns={"covariate": "feature"})
        elif "index" in summary_df.columns:
            summary_df = summary_df.rename(columns={"index": "feature"})

        required_cols = {"feature", "coef", "exp(coef)", "p"}
        if not required_cols.issubset(summary_df.columns):
            missing = sorted(required_cols - set(summary_df.columns))
            status = f"failed: missing_summary_columns={missing}"
            return coef_df, summary_df, status

        coef_df = summary_df[["feature", "coef", "exp(coef)", "p"]].copy()
        coef_df = coef_df.rename(columns={"exp(coef)": "exp_coef_hr"})
        coef_df = (
            coef_df.set_index("feature")
            .reindex(features)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    except (ConvergenceError, np.linalg.LinAlgError) as exc:
        status = f"failed: {type(exc).__name__}"
    except Exception as exc:
        status = f"failed: {type(exc).__name__}"

    return coef_df, summary_df, status


def cox_cv_cindex(
    group_df: pd.DataFrame,
    endpoint: str,
    features: List[str],
    n_splits: int,
    penalizer: float,
    l1_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    time_col = f"{endpoint}.time"
    sub = group_df[[time_col, endpoint] + features].copy()

    for feat in features:
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
    sub = sub.dropna().copy().reset_index(drop=True)

    rows = []

    if len(sub) < max(n_splits, 10):
        return pd.DataFrame(
            [{"fold": np.nan, "cindex": np.nan, "status": "too_few_rows"}]
        )

    if sub[endpoint].sum() == 0 or sub[endpoint].sum() == len(sub):
        return pd.DataFrame(
            [{"fold": np.nan, "cindex": np.nan, "status": "invalid_event_distribution"}]
        )

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(sub), start=1):
        train_df = sub.iloc[train_idx].copy()
        test_df = sub.iloc[test_idx].copy()

        row = {"fold": fold_idx, "cindex": np.nan, "status": "ok"}

        if train_df[endpoint].sum() == 0 or train_df[endpoint].sum() == len(train_df):
            row["status"] = "invalid_train_event_distribution"
            rows.append(row)
            continue

        if test_df[endpoint].sum() == 0 or test_df[endpoint].sum() == len(test_df):
            row["status"] = "invalid_test_event_distribution"
            rows.append(row)
            continue

        try:
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            cph.fit(train_df, duration_col=time_col, event_col=endpoint)
            risk = cph.predict_partial_hazard(test_df).to_numpy().reshape(-1)
            row["cindex"] = safe_concordance_index(
                times=test_df[time_col].to_numpy(),
                risk_scores=risk,
                events=test_df[endpoint].to_numpy(),
            )
        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}"

        rows.append(row)

    return pd.DataFrame(rows)


def ensure_expected_multivariate_columns(multi_coef_df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ["feature", "coef", "exp_coef_hr", "p"]
    out = multi_coef_df.copy()

    if "feature" not in out.columns:
        out["feature"] = np.nan

    for col in expected_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out[expected_cols]


def compare_top_univariate_vs_multivariate(
    uni_df: pd.DataFrame,
    multi_coef_df: pd.DataFrame,
) -> pd.DataFrame:
    multi_coef_df = ensure_expected_multivariate_columns(multi_coef_df)

    out = uni_df[["feature", "univariate_cindex"]].merge(
        multi_coef_df[["feature", "coef", "exp_coef_hr", "p"]],
        on="feature",
        how="left",
    )
    out["abs_coef"] = pd.to_numeric(out["coef"], errors="coerce").abs()
    return out.sort_values("univariate_cindex", ascending=False)


def check_ph_assumptions(
    group_df: pd.DataFrame,
    endpoint: str,
    features: List[str],
    penalizer: float,
    l1_ratio: float,
) -> pd.DataFrame:
    time_col = f"{endpoint}.time"
    sub = group_df[[time_col, endpoint] + features].copy()

    for feat in features:
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
    sub = sub.dropna().copy()

    if len(sub) < 10:
        return pd.DataFrame([{"status": "too_few_rows"}])

    if sub[endpoint].sum() == 0 or sub[endpoint].sum() == len(sub):
        return pd.DataFrame([{"status": "invalid_event_distribution"}])

    try:
        from lifelines.statistics import proportional_hazard_test

        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        cph.fit(sub, duration_col=time_col, event_col=endpoint)

        ph_test = proportional_hazard_test(cph, sub, time_transform="rank")
        ph_df = ph_test.summary.reset_index()
        if "index" in ph_df.columns:
            ph_df = ph_df.rename(columns={"index": "feature"})
        return ph_df
    except Exception as exc:
        return pd.DataFrame([{"status": f"failed: {type(exc).__name__}"}])


def summarize_group_endpoint(
    group_name: str,
    endpoint: str,
    group_df_raw: pd.DataFrame,
    group_df_endpoint: pd.DataFrame,
    features: List[str],
    min_rows: int,
    min_events: int,
) -> Dict[str, float]:
    n_raw = len(group_df_raw)
    n_after_endpoint = len(group_df_endpoint)
    n_complete = len(group_df_endpoint.dropna(subset=features))
    n_events = int(group_df_endpoint[endpoint].sum()) if len(group_df_endpoint) > 0 else 0

    status = "ok"
    if n_after_endpoint < min_rows:
        status = "too_few_rows"
    elif n_events < min_events:
        status = "too_few_events"
    elif n_events == n_after_endpoint:
        status = "all_events"
    elif n_events == 0:
        status = "no_events"

    return {
        "group": group_name,
        "endpoint": endpoint,
        "n_raw_group": n_raw,
        "n_after_endpoint_cleanup": n_after_endpoint,
        "n_complete_case_all_features": n_complete,
        "complete_case_loss_count": n_after_endpoint - n_complete,
        "complete_case_loss_pct": (
            (n_after_endpoint - n_complete) / n_after_endpoint
            if n_after_endpoint > 0
            else np.nan
        ),
        "n_events": n_events,
        "event_rate": (n_events / n_after_endpoint) if n_after_endpoint > 0 else np.nan,
        "status": status,
        "epv_all_8_features": (n_events / len(features)) if len(features) > 0 else np.nan,
    }


def run_group_endpoint_analysis(
    group_name: str,
    group_df: pd.DataFrame,
    endpoint: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, float]:
    logging.info("Running diagnostics | group=%s | endpoint=%s", group_name, endpoint)

    endpoint_df = preprocess_endpoint_df(
        df=group_df,
        endpoint=endpoint,
        censor_at_months=args.censor_at_months,
    )

    summary = summarize_group_endpoint(
        group_name=group_name,
        endpoint=endpoint,
        group_df_raw=group_df,
        group_df_endpoint=endpoint_df,
        features=FEATURES,
        min_rows=args.min_rows,
        min_events=args.min_events,
    )

    group_safe = group_name.replace("&", "and").replace(" ", "_")
    base_name = f"{group_safe}__{endpoint}"
    subdir = out_dir / endpoint / group_safe
    ensure_dir(subdir)

    pd.DataFrame([summary]).to_csv(subdir / f"{base_name}__summary.csv", index=False)

    availability_df = compute_feature_availability(endpoint_df, FEATURES)
    availability_df.to_csv(subdir / f"{base_name}__feature_availability.csv", index=False)

    complete_case_info = compute_complete_case_loss(endpoint_df, endpoint, FEATURES)
    pd.DataFrame([complete_case_info]).to_csv(
        subdir / f"{base_name}__complete_case_loss.csv",
        index=False,
    )

    if summary["status"] != "ok":
        logging.warning(
            "Skipping heavy analysis | group=%s | endpoint=%s | status=%s",
            group_name,
            endpoint,
            summary["status"],
        )
        return summary

    uni_ci_df = compute_univariate_cindex_table(endpoint_df, endpoint, FEATURES)
    uni_ci_df.to_csv(subdir / f"{base_name}__univariate_cindex.csv", index=False)

    uni_cox_df = fit_univariate_cox(
        group_df=endpoint_df,
        endpoint=endpoint,
        features=FEATURES,
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
    )
    uni_cox_df.to_csv(subdir / f"{base_name}__univariate_cox.csv", index=False)

    corr_input_df = endpoint_df.dropna(subset=FEATURES).copy()
    if len(corr_input_df) >= 2:
        corr_df, vif_df = compute_corr_and_vif(corr_input_df, FEATURES)
        corr_df.to_csv(subdir / f"{base_name}__feature_correlation.csv")
        vif_df.to_csv(subdir / f"{base_name}__feature_vif.csv", index=False)
        save_corr_heatmap(
            corr_df=corr_df,
            out_path=subdir / f"{base_name}__feature_correlation_heatmap.png",
            title=f"Feature correlation | {group_name} | {endpoint}",
        )
    else:
        corr_df = pd.DataFrame(index=FEATURES, columns=FEATURES, dtype=float)
        vif_df = pd.DataFrame({"feature": FEATURES, "vif": [np.nan] * len(FEATURES)})
        corr_df.to_csv(subdir / f"{base_name}__feature_correlation.csv")
        vif_df.to_csv(subdir / f"{base_name}__feature_vif.csv", index=False)

    multi_coef_df, multi_summary_df, multi_status = fit_multivariate_cox_all_features(
        group_df=endpoint_df,
        endpoint=endpoint,
        features=FEATURES,
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
    )
    multi_coef_df = ensure_expected_multivariate_columns(multi_coef_df)

    multi_coef_df.to_csv(subdir / f"{base_name}__multivariate_cox_all8.csv", index=False)
    if not multi_summary_df.empty:
        multi_summary_df.to_csv(
            subdir / f"{base_name}__multivariate_cox_all8_fullsummary.csv",
            index=False,
        )
    pd.DataFrame([{"multivariate_status": multi_status}]).to_csv(
        subdir / f"{base_name}__multivariate_status.csv",
        index=False,
    )

    cv_df = cox_cv_cindex(
        group_df=endpoint_df,
        endpoint=endpoint,
        features=FEATURES,
        n_splits=args.n_splits,
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
        random_state=args.random_state,
    )
    cv_df.to_csv(subdir / f"{base_name}__multivariate_cox_cv.csv", index=False)

    ph_df = check_ph_assumptions(
        group_df=endpoint_df,
        endpoint=endpoint,
        features=FEATURES,
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
    )
    ph_df.to_csv(subdir / f"{base_name}__ph_assumption_test.csv", index=False)

    compare_df = compare_top_univariate_vs_multivariate(
        uni_df=uni_ci_df,
        multi_coef_df=multi_coef_df,
    )
    compare_df.to_csv(
        subdir / f"{base_name}__univariate_vs_multivariate_compare.csv",
        index=False,
    )

    cv_mean = float(cv_df["cindex"].dropna().mean()) if "cindex" in cv_df.columns else np.nan
    cv_std = float(cv_df["cindex"].dropna().std()) if "cindex" in cv_df.columns else np.nan
    best_uni = float(uni_ci_df["univariate_cindex"].max()) if not uni_ci_df.empty else np.nan
    best_feat = (
        str(uni_ci_df.sort_values("univariate_cindex", ascending=False).iloc[0]["feature"])
        if not uni_ci_df.empty
        else ""
    )

    final_diag = {
        **summary,
        "best_univariate_feature": best_feat,
        "best_univariate_cindex": best_uni,
        "multivariate_cv_mean_cindex": cv_mean,
        "multivariate_cv_std_cindex": cv_std,
        "max_vif": float(vif_df["vif"].max()) if not vif_df.empty else np.nan,
        "n_features_vif_gt_5": int((vif_df["vif"] > 5).sum()) if not vif_df.empty else 0,
        "n_features_vif_gt_10": int((vif_df["vif"] > 10).sum()) if not vif_df.empty else 0,
        "multivariate_status": multi_status,
    }

    pd.DataFrame([final_diag]).to_csv(
        subdir / f"{base_name}__final_diagnostic_summary.csv",
        index=False,
    )
    return final_diag


def save_overall_heatmaps(overall_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    uni_mat = overall_df.pivot(index="group", columns="endpoint", values="best_univariate_cindex")
    uni_mat = uni_mat.reindex(index=GROUP_ORDER, columns=ENDPOINTS)

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        uni_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0.3,
        vmax=0.9,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "C-index"},
    )
    plt.title("Best univariate C-index by group and endpoint")
    plt.tight_layout()
    plt.savefig(out_dir / "overall_best_univariate_cindex_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    multi_mat = overall_df.pivot(
        index="group",
        columns="endpoint",
        values="multivariate_cv_mean_cindex",
    )
    multi_mat = multi_mat.reindex(index=GROUP_ORDER, columns=ENDPOINTS)

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        multi_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0.3,
        vmax=0.9,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "C-index"},
    )
    plt.title("Multivariate Cox CV mean C-index by group and endpoint")
    plt.tight_layout()
    plt.savefig(
        out_dir / "overall_multivariate_cv_mean_cindex_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    cc_mat = overall_df.pivot(
        index="group",
        columns="endpoint",
        values="complete_case_loss_pct",
    )
    cc_mat = cc_mat.reindex(index=GROUP_ORDER, columns=ENDPOINTS)

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        cc_mat,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Fraction lost"},
    )
    plt.title("Complete-case loss fraction by group and endpoint")
    plt.tight_layout()
    plt.savefig(
        out_dir / "overall_complete_case_loss_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    vif_mat = overall_df.pivot(index="group", columns="endpoint", values="max_vif")
    vif_mat = vif_mat.reindex(index=GROUP_ORDER, columns=ENDPOINTS)

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        vif_mat,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Max VIF"},
    )
    plt.title("Maximum VIF by group and endpoint")
    plt.tight_layout()
    plt.savefig(out_dir / "overall_max_vif_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input_excel)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

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
    overall_rows = []

    for endpoint in ENDPOINTS:
        time_col = f"{endpoint}.time"
        if endpoint not in df.columns or time_col not in df.columns:
            logging.warning("Skipping endpoint %s because required columns are missing.", endpoint)
            continue

        for group_name in GROUP_ORDER:
            group_df = groups[group_name]
            result = run_group_endpoint_analysis(
                group_name=group_name,
                group_df=group_df,
                endpoint=endpoint,
                out_dir=output_dir,
                args=args,
            )
            overall_rows.append(result)

    overall_df = pd.DataFrame(overall_rows)
    overall_df = overall_df.sort_values(["endpoint", "group"])
    overall_df.to_csv(output_dir / "overall_group_endpoint_summary.csv", index=False)

    reviewer_cols = [
        "group",
        "endpoint",
        "n_after_endpoint_cleanup",
        "n_events",
        "epv_all_8_features",
        "complete_case_loss_pct",
        "best_univariate_feature",
        "best_univariate_cindex",
        "multivariate_cv_mean_cindex",
        "multivariate_cv_std_cindex",
        "max_vif",
        "n_features_vif_gt_5",
        "n_features_vif_gt_10",
        "multivariate_status",
        "status",
    ]
    for col in reviewer_cols:
        if col not in overall_df.columns:
            overall_df[col] = np.nan

    reviewer_table = overall_df[reviewer_cols].copy()
    reviewer_table.to_csv(output_dir / "reviewer_friendly_diagnostic_table.csv", index=False)

    save_overall_heatmaps(overall_df, output_dir)

    logging.info("Finished. Outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()