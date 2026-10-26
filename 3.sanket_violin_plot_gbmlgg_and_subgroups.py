import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

# =========================================================
# HARD-CODED PATHS
# =========================================================
INPUT_EXCEL = r"D:\3_modelling\pacman\data\violin_plot_raw_data.xlsx"
OUTPUT_DIR = r"D:\3_modelling\pacman\results\violin_proliferation_hot_vs_cold_by_subgroup"
SHEET_NAME = None  # Keep None unless a specific sheet is needed

# =========================================================
# SETTINGS
# =========================================================
LOG_LEVEL = "INFO"
FIGURE_DPI = 400
MIN_GROUP_SIZE = 5

# =========================================================
# COLUMN SETTINGS
# =========================================================
TARGET_COL = "Proliferation"
TEMPERATURE_COL = "temperature"
TYPE_COL = "type"
BARCODE_COL = "bcr_patient_barcode"

# If the file has multiple slide rows per patient, set this to True
# so each patient contributes only once.
DEDUPLICATE_BY_PATIENT = True

# Which row to keep per patient when deduplicating.
# "first" = keep first row after sorting
# "mean"  = aggregate numeric target by patient mean
DEDUP_METHOD = "first"  # choices: "first", "mean"

# If you want to strictly limit the analysis to GBMLGG only, set True
FILTER_TO_GBMLGG = True

# Match the PACMAN-style proliferation plot range
Y_LIMITS = (-3, 3)

COHORT_ORDER = [
    "GBMLGG",
    "IDH-wildtype",
    "IDH-mutant",
    "Grade 4 astro",
    "Grade 2&3 astro",
    "Oligo",
]

PALETTE = {"Cold": "#3B5CCC", "Hot": "#C21F3A"}


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("&", "and")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        logging.warning(
            "Duplicate columns found and removed (keeping first): %s",
            duplicated,
        )
    return df.loc[:, ~df.columns.duplicated()].copy()


def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Column not found: {col}")
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def clean_string_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    s = get_series(df, col).astype(str).str.strip()
    s = s.replace(
        {
            "nan": np.nan,
            "None": np.nan,
            "": np.nan,
            "[Not Available]": np.nan,
            "[Not Applicable]": np.nan,
        }
    )
    df[col] = s


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(get_series(df, col), errors="coerce")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = drop_duplicate_columns(df)

    rename_map = {
        "WHO_CNS5_histology": "histology",
        "WHO_CNS5_hist": "idh_status",
        "WHO_CNS5_grade": "grade",
        "temperature": "temperature",
        "type": "type",
        "Proliferation": "Proliferation",
        "bcr_patient_barcode": "bcr_patient_barcode",
    }
    df = df.rename(columns=rename_map)
    df = drop_duplicate_columns(df)

    for col in [
        "histology",
        "idh_status",
        "grade",
        "temperature",
        "type",
        "bcr_patient_barcode",
    ]:
        clean_string_column(df, col)

    if TEMPERATURE_COL in df.columns:
        df[TEMPERATURE_COL] = get_series(df, TEMPERATURE_COL).replace(
            {
                "Mitotic-Hot": "Hot",
                "Mitotic-Cold": "Cold",
                "hot": "Hot",
                "cold": "Cold",
                "HOT": "Hot",
                "COLD": "Cold",
            }
        )

    # Normalize common IDH naming variants
    if "idh_status" in df.columns:
        df["idh_status"] = get_series(df, "idh_status").replace(
            {
                "WT": "IDH-wildtype",
                "Mutant": "IDH-mutant",
                "IDH WT": "IDH-wildtype",
                "IDH-Mutant": "IDH-mutant",
                "IDH-Wildtype": "IDH-wildtype",
            }
        )

    # Normalize some common histology naming variants
    if "histology" in df.columns:
        df["histology"] = get_series(df, "histology").replace(
            {
                "astrocytoma": "Astrocytoma",
                "Astro": "Astrocytoma",
                "oligodendroglioma": "Oligodendroglioma",
                "Oligo": "Oligodendroglioma",
            }
        )

    # Normalize grade naming variants
    if "grade" in df.columns:
        df["grade"] = get_series(df, "grade").replace(
            {
                "G2": "Grade 2",
                "G3": "Grade 3",
                "G4": "Grade 4",
                "grade 2": "Grade 2",
                "grade 3": "Grade 3",
                "grade 4": "Grade 4",
            }
        )

    return df


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_dataframe() -> pd.DataFrame:
    input_path = Path(INPUT_EXCEL)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logging.info("Loading dataframe from %s", input_path)

    if SHEET_NAME:
        df = pd.read_excel(input_path, sheet_name=SHEET_NAME)
    else:
        df = pd.read_excel(input_path)

    df = standardize_columns(df)

    required_cols = [
        TEMPERATURE_COL,
        TARGET_COL,
        "histology",
        "idh_status",
        "grade",
    ]
    validate_required_columns(df, required_cols)

    df[TARGET_COL] = safe_numeric(df, TARGET_COL)

    if FILTER_TO_GBMLGG and TYPE_COL in df.columns:
        before = len(df)
        df = df[get_series(df, TYPE_COL) == "GBMLGG"].copy()
        logging.info("Filtered to GBMLGG: %d -> %d rows", before, len(df))

    return df


def deduplicate_patient_rows(df: pd.DataFrame) -> pd.DataFrame:
    if not DEDUPLICATE_BY_PATIENT:
        logging.info("Patient deduplication disabled; using row-level data.")
        return df.copy()

    if BARCODE_COL not in df.columns:
        logging.warning(
            "Column '%s' not found; cannot deduplicate by patient. Using row-level data.",
            BARCODE_COL,
        )
        return df.copy()

    work_df = df.copy()
    work_df = work_df.dropna(subset=[BARCODE_COL]).copy()

    if DEDUP_METHOD == "first":
        dedup_df = work_df.drop_duplicates(subset=BARCODE_COL, keep="first").copy()
        logging.info(
            "Deduplicated by patient using first row: %d -> %d rows",
            len(work_df),
            len(dedup_df),
        )
        return dedup_df

    if DEDUP_METHOD == "mean":
        group_cols = [BARCODE_COL]
        keep_first_cols = [
            col
            for col in work_df.columns
            if col not in [TARGET_COL] and col != BARCODE_COL
        ]

        agg_dict = {TARGET_COL: "mean"}
        for col in keep_first_cols:
            agg_dict[col] = "first"

        dedup_df = work_df.groupby(BARCODE_COL, as_index=False).agg(agg_dict)
        logging.info(
            "Deduplicated by patient using mean target value: %d -> %d rows",
            len(work_df),
            len(dedup_df),
        )
        return dedup_df

    raise ValueError(f"Unsupported DEDUP_METHOD: {DEDUP_METHOD}")


def build_cohorts(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "GBMLGG": df.copy(),
        "IDH-wildtype": df[get_series(df, "idh_status") == "IDH-wildtype"].copy(),
        "IDH-mutant": df[get_series(df, "idh_status") == "IDH-mutant"].copy(),
        "Grade 4 astro": df[
            (get_series(df, "histology") == "Astrocytoma")
            & (get_series(df, "grade") == "Grade 4")
        ].copy(),
        "Grade 2&3 astro": df[
            (get_series(df, "histology") == "Astrocytoma")
            & (get_series(df, "grade").isin(["Grade 2", "Grade 3"]))
        ].copy(),
        "Oligo": df[
            get_series(df, "histology") == "Oligodendroglioma"
        ].copy(),
    }


def star_from_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "ns"
    if p_value <= 1e-4:
        return "****"
    if p_value <= 1e-3:
        return "***"
    if p_value <= 1e-2:
        return "**"
    if p_value <= 5e-2:
        return "*"
    return "ns"


def mann_whitney_p_value(df: pd.DataFrame, value_col: str) -> float:
    sub = pd.DataFrame(
        {
            TEMPERATURE_COL: get_series(df, TEMPERATURE_COL),
            value_col: safe_numeric(df, value_col),
        }
    ).dropna(subset=[TEMPERATURE_COL, value_col])

    cold = sub.loc[sub[TEMPERATURE_COL] == "Cold", value_col].to_numpy()
    hot = sub.loc[sub[TEMPERATURE_COL] == "Hot", value_col].to_numpy()

    if len(cold) < 2 or len(hot) < 2:
        return np.nan

    try:
        _, p_value = mannwhitneyu(cold, hot, alternative="two-sided")
        return float(p_value)
    except Exception as exc:
        logging.warning("Mann-Whitney failed for %s: %s", value_col, exc)
        return np.nan


def summarize_group_stats(
    sub: pd.DataFrame,
    value_col: str,
) -> Dict[str, float]:
    cold_vals = sub.loc[sub[TEMPERATURE_COL] == "Cold", value_col]
    hot_vals = sub.loc[sub[TEMPERATURE_COL] == "Hot", value_col]

    return {
        "n_cold": int((sub[TEMPERATURE_COL] == "Cold").sum()),
        "n_hot": int((sub[TEMPERATURE_COL] == "Hot").sum()),
        "median_cold": float(cold_vals.median()) if len(cold_vals) > 0 else np.nan,
        "median_hot": float(hot_vals.median()) if len(hot_vals) > 0 else np.nan,
        "mean_cold": float(cold_vals.mean()) if len(cold_vals) > 0 else np.nan,
        "mean_hot": float(hot_vals.mean()) if len(hot_vals) > 0 else np.nan,
    }


def compute_annotation_positions(y_limits: Tuple[float, float]) -> Tuple[float, float, float]:
    y_min, y_max = y_limits
    y_range = y_max - y_min
    line_y = y_max - 0.12 * y_range
    bracket_height = 0.025 * y_range
    text_y = y_max - 0.08 * y_range
    return line_y, bracket_height, text_y


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(*Y_LIMITS)


def make_proliferation_figure(
    cohorts: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    sns.set_style("white")

    plotted_any = False
    line_y, bracket_height, text_y = compute_annotation_positions(Y_LIMITS)

    for idx, cohort_name in enumerate(COHORT_ORDER):
        ax = axes[idx // 3][idx % 3]
        cohort_df = cohorts[cohort_name].copy()

        sub = pd.DataFrame(
            {
                TEMPERATURE_COL: get_series(cohort_df, TEMPERATURE_COL),
                TARGET_COL: safe_numeric(cohort_df, TARGET_COL),
            }
        )
        sub = sub[sub[TEMPERATURE_COL].isin(["Cold", "Hot"])].copy()
        sub = sub.dropna(subset=[TEMPERATURE_COL, TARGET_COL]).copy()

        p_value = mann_whitney_p_value(cohort_df, TARGET_COL)
        stars = star_from_p_value(p_value)
        stats = summarize_group_stats(sub, TARGET_COL)

        rows.append(
            {
                "cohort": cohort_name,
                "characteristic": TARGET_COL,
                **stats,
                "p_value": p_value,
                "significance": stars,
            }
        )

        if stats["n_cold"] < MIN_GROUP_SIZE or stats["n_hot"] < MIN_GROUP_SIZE:
            ax.axis("off")
            ax.set_title(f"{cohort_name}\nToo few samples", fontsize=12)
            logging.warning(
                "Skipping plot for %s due to insufficient samples: Cold=%d, Hot=%d",
                cohort_name,
                stats["n_cold"],
                stats["n_hot"],
            )
            continue

        plotted_any = True

        sns.violinplot(
            data=sub,
            x=TEMPERATURE_COL,
            y=TARGET_COL,
            order=["Cold", "Hot"],
            palette=PALETTE,
            cut=0,
            inner="box",
            linewidth=1,
            ax=ax,
        )

        ax.plot(
            [0, 0, 1, 1],
            [line_y, line_y + bracket_height, line_y + bracket_height, line_y],
            c="black",
            lw=1,
        )
        ax.text(
            0.5,
            text_y,
            stars,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

        style_axis(ax, title=cohort_name, ylabel=TARGET_COL)

    fig.suptitle(
        f"{TARGET_COL}: Mitotic-Cold vs Mitotic-Hot",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / f"{safe_name(TARGET_COL)}_stats.csv", index=False)

    if plotted_any:
        png_path = output_dir / f"{safe_name(TARGET_COL)}_hot_vs_cold_by_subgroup.png"
        pdf_path = output_dir / f"{safe_name(TARGET_COL)}_hot_vs_cold_by_subgroup.pdf"
        fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logging.info("Saved proliferation figure to %s and %s", png_path, pdf_path)
    else:
        logging.warning("No valid proliferation plots were generated.")

    plt.close(fig)
    return stats_df


def main() -> None:
    setup_logging()
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    df = load_dataframe()
    df = deduplicate_patient_rows(df)
    cohorts = build_cohorts(df)

    stats_df = make_proliferation_figure(cohorts=cohorts, output_dir=output_dir)
    stats_df.to_csv(output_dir / "all_characteristics_stats.csv", index=False)

    logging.info("Done. Outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()