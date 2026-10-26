import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pacman.config import DATA_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")

print("======= Multivariate Survival Analysis =======")

FEATURES = [
    "HSC",
    "mean(ND)",
    "cv(ND)",
    "mean(CL)",
    "mean(HC)",
    "AMH",
    "AMAH",
    "AFW",
]

ENDPOINTS = ["OS", "PFI", "DSS", "DFI"]

CENSOR_AT = 120

SAVE_DIR = os.path.join(RESULTS_DIR, "survival", "multivariate_models")
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------
# Subgroup creation
# ---------------------------------------------------
def create_subgroups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = {}

    groups["IDH-wildtype"] = df[df["idh_status"] == "IDH-wildtype"].copy()

    groups["IDH-mutant"] = df[df["idh_status"] == "IDH-mutant"].copy()

    groups["Grade4_astro"] = df[
        (df["histology"] == "Astrocytoma") & (df["grade"] == "Grade 4")
    ].copy()

    groups["Grade23_astro"] = df[
        (df["histology"] == "Astrocytoma")
        & (df["grade"].isin(["Grade 2", "Grade 3"]))
    ].copy()

    groups["Oligo"] = df[df["histology"] == "Oligodendroglioma"].copy()

    groups["Grade23_astro_oligo"] = df[
        (
            ((df["histology"] == "Astrocytoma") &
             (df["grade"].isin(["Grade 2", "Grade 3"])))
            | (df["histology"] == "Oligodendroglioma")
        )
    ].copy()

    return groups


# ---------------------------------------------------
# Safe C-index
# ---------------------------------------------------
def safe_cindex(
    times: np.ndarray,
    risk: np.ndarray,
    events: np.ndarray,
) -> float:
    times = np.asarray(times, dtype=float)
    risk = np.asarray(risk, dtype=float)
    events = np.asarray(events, dtype=float)

    mask = (~np.isnan(times)) & (~np.isnan(risk)) & (~np.isnan(events))
    times = times[mask]
    risk = risk[mask]
    events = events[mask].astype(int)

    if len(times) < 3:
        return np.nan

    if events.sum() == 0 or events.sum() == len(events):
        return np.nan

    try:
        return concordance_index(times, risk, events)
    except Exception:
        return np.nan


# ---------------------------------------------------
# RFE feature selection
# ---------------------------------------------------
def run_rfe(X: np.ndarray, y: np.ndarray, n_features_to_select: int = 4) -> np.ndarray:
    n_features_to_select = min(n_features_to_select, X.shape[1])

    try:
        model = SVR(kernel="linear")
        rfe = RFE(model, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        return rfe.support_
    except Exception:
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[:n_features_to_select] = True
        return mask


# ---------------------------------------------------
# SVM risk model
# ---------------------------------------------------
def svm_risk_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        model = SVR(kernel="rbf", C=1.0, gamma="scale")
        model.fit(X, y)
        preds = model.predict(X)
        return np.asarray(preds, dtype=float)
    except Exception:
        return np.full(X.shape[0], np.nan)


# ---------------------------------------------------
# Cox model
# ---------------------------------------------------
def cox_model(
    df: pd.DataFrame,
    features: list[str],
    time_col: str,
    event_col: str,
) -> np.ndarray:
    cox_df = df[[time_col, event_col] + features].copy()
    cox_df = cox_df.replace([np.inf, -np.inf], np.nan)

    if cox_df.empty:
        return np.full(len(df), np.nan)

    valid_mask = cox_df.notna().all(axis=1)
    cox_fit_df = cox_df.loc[valid_mask].copy()

    if len(cox_fit_df) < 10:
        return np.full(len(df), np.nan)

    if cox_fit_df[event_col].sum() == 0:
        return np.full(len(df), np.nan)

    risk = np.full(len(df), np.nan)

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_fit_df, duration_col=time_col, event_col=event_col)
        risk_valid = cph.predict_partial_hazard(cox_fit_df).values.astype(float)
        risk[np.where(valid_mask)[0]] = risk_valid
    except Exception:
        return np.full(len(df), np.nan)

    return risk


# ---------------------------------------------------
# Random Forest risk model
# ---------------------------------------------------
def rf_risk_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        return np.asarray(preds, dtype=float)
    except Exception:
        return np.full(X.shape[0], np.nan)


# ---------------------------------------------------
# Heatmap plot
# ---------------------------------------------------
def plot_heatmap(df: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    annot_df = df.copy().round(2)

    sns.heatmap(
        df,
        annot=annot_df,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0.3,
        vmax=0.8,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_title(title)

    fig.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------
def main() -> None:
    df = pd.read_excel(os.path.join(DATA_DIR, "heatmap_cin_raw_num.xlsx"))

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

    # Clean string columns
    for col in ["histology", "idh_status", "grade"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Ensure feature columns are numeric
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(df.columns.tolist())

    groups = create_subgroups(df)

    for endpoint in ENDPOINTS:
        print(f"\n==== {endpoint} ====")

        time_col = f"{endpoint}.time"

        results = {
            "SVM": {},
            "Cox": {},
            "RF": {},
        }

        diagnostics_rows = []

        for gname, gdf in groups.items():
            print("Processing", gname)

            required_cols = [endpoint, time_col] + FEATURES
            missing_cols = [c for c in required_cols if c not in gdf.columns]
            if missing_cols:
                print(f"Skipping {gname}: missing columns {missing_cols}")
                results["SVM"][gname] = np.nan
                results["Cox"][gname] = np.nan
                results["RF"][gname] = np.nan
                continue

            gdf = gdf.dropna(subset=[endpoint, time_col] + FEATURES).copy()

            if len(gdf) < 10:
                results["SVM"][gname] = np.nan
                results["Cox"][gname] = np.nan
                results["RF"][gname] = np.nan
                diagnostics_rows.append(
                    {
                        "group": gname,
                        "endpoint": endpoint,
                        "n_rows": len(gdf),
                        "n_events": np.nan,
                        "status": "too_few_rows",
                    }
                )
                continue

            gdf[endpoint] = pd.to_numeric(gdf[endpoint], errors="coerce")
            gdf[time_col] = pd.to_numeric(gdf[time_col], errors="coerce")
            gdf = gdf.dropna(subset=[endpoint, time_col]).copy()

            if len(gdf) < 10:
                results["SVM"][gname] = np.nan
                results["Cox"][gname] = np.nan
                results["RF"][gname] = np.nan
                diagnostics_rows.append(
                    {
                        "group": gname,
                        "endpoint": endpoint,
                        "n_rows": len(gdf),
                        "n_events": np.nan,
                        "status": "too_few_rows_after_numeric_cleanup",
                    }
                )
                continue

            gdf[endpoint] = gdf[endpoint].astype(int)
            gdf[time_col] = (gdf[time_col] / 30.4).astype(int)

            mask = gdf[time_col] > CENSOR_AT
            gdf.loc[mask, endpoint] = 0
            gdf.loc[mask, time_col] = CENSOR_AT

            n_events = int(gdf[endpoint].sum())

            diagnostics_rows.append(
                {
                    "group": gname,
                    "endpoint": endpoint,
                    "n_rows": len(gdf),
                    "n_events": n_events,
                    "status": "ok",
                }
            )

            if n_events == 0 or n_events == len(gdf):
                results["SVM"][gname] = np.nan
                results["Cox"][gname] = np.nan
                results["RF"][gname] = np.nan
                continue

            X = gdf[FEATURES].values.astype(float)
            y = gdf[time_col].values.astype(float)

            scaler = StandardScaler()
            try:
                X = scaler.fit_transform(X)
            except Exception:
                results["SVM"][gname] = np.nan
                results["Cox"][gname] = np.nan
                results["RF"][gname] = np.nan
                continue

            # --------------------------
            # Feature elimination
            # --------------------------
            selected = run_rfe(X, y, n_features_to_select=4)
            X_sel = X[:, selected]
            feats_sel = np.array(FEATURES)[selected].tolist()

            # --------------------------
            # SVM
            # --------------------------
            svm_risk = svm_risk_model(X_sel, y)
            svm_ci = safe_cindex(
                gdf[time_col].values,
                -svm_risk,
                gdf[endpoint].values,
            )
            results["SVM"][gname] = svm_ci

            # --------------------------
            # Cox
            # --------------------------
            cox_risk = cox_model(gdf, feats_sel, time_col, endpoint)
            cox_ci = safe_cindex(
                gdf[time_col].values,
                cox_risk,
                gdf[endpoint].values,
            )
            results["Cox"][gname] = cox_ci

            # --------------------------
            # Random Forest
            # --------------------------
            rf_risk = rf_risk_model(X_sel, y)
            rf_ci = safe_cindex(
                gdf[time_col].values,
                -rf_risk,
                gdf[endpoint].values,
            )
            results["RF"][gname] = rf_ci

        result_df = pd.DataFrame(results).T

        desired_cols = [
            "IDH-wildtype",
            "IDH-mutant",
            "Grade4_astro",
            "Grade23_astro",
            "Oligo",
            "Grade23_astro_oligo",
        ]
        result_df = result_df.reindex(columns=desired_cols)

        plot_heatmap(
            result_df,
            f"Multivariate C-index ({endpoint})",
            os.path.join(SAVE_DIR, f"multivariate_cindex_{endpoint}.png"),
        )

        result_df.to_csv(
            os.path.join(SAVE_DIR, f"multivariate_cindex_{endpoint}.csv")
        )

        diag_df = pd.DataFrame(diagnostics_rows)
        diag_df.to_csv(
            os.path.join(SAVE_DIR, f"multivariate_diagnostics_{endpoint}.csv"),
            index=False,
        )

    print("Finished multivariate analysis.")


if __name__ == "__main__":
    main()