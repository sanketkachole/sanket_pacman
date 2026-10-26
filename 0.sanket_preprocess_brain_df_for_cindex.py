# import pandas as pd
# from pathlib import Path
#
# # --------------------- CONFIG --------------------- #
# DATA_DIR = Path(r"D:\3_modelling\pacman\data")  # adjust if needed
# INPUT_FILE = DATA_DIR / "brain_df.xlsx"
# OUTPUT_FILE = DATA_DIR / "heatmap_cin_raw_num.xlsx"
#
# # Mitotic features already present in ST1-style data
# orig_mitotic_feats = [
#     "mit_hotspot_score",       # HSC
#     "mit_nodeDegrees_mean",    # mean(ND)
#     "mit_nodeDegrees_cv",      # cv(ND)
#     "mit_clusterCoff_mean",    # mean(CL)
#     "mit_cenHarmonic_mean",    # mean(HC)
#     "aty_hotspot_ratio",       # AMH
#     "aty_ahotspot_ratio",      # AMAH
#     "aty_wsi_ratio",           # AFW
# ]
#
# # Your NEW metrics extracted from the graph JSONs
# new_graph_metrics = [
#     "n_nodes",
#     "n_edges",
#     "density",
#     "n_components",
#     "largest_component_size",
#     "mean_degree",
#     "cv_degree",
#     "global_clustering",
#     "mean_local_clustering",
#     "mean_harmonic_cen",
#     "max_harmonic_cen",
#     "AFW",                       # already overlaps → keep once
#     "avg_shortest_path_lcc",
#     "diameter_lcc",
#     "degree_assortativity"
# ]
#
# # Survival columns (supported by the author's script)
# survival_cols = [
#     "OS", "OS.time",
#     "DSS", "DSS.time",
#     "DFI", "DFI.time",
#     "PFI", "PFI.time",
# ]
#
# # Grouping column for C-index (key: IDH-wt vs IDH-mut)
# GROUP_COL = "WHO_CNS5_hist"  # becomes "type" below
#
# # Slide identifier / patient ID columns to keep
# id_cols = [
#     "slide_id",
#     "bcr_patient_barcode",
#     "case_id",
# ]
#
# # ---------------------------------------------------- #
#
# print(f"Loading {INPUT_FILE} ...")
# df = pd.read_excel(INPUT_FILE)
#
# # Rename grouping column to "type" (required by author’s C-index script)
# df = df.rename(columns={GROUP_COL: "type"})
#
# # Build final list of required columns
# keep_cols = (
#         id_cols +
#         survival_cols +
#         orig_mitotic_feats +
#         new_graph_metrics +
#         ["type"]  # grouping column must be included
# )
#
# # Keep only columns that actually exist in the file
# keep_cols = [c for c in keep_cols if c in df.columns]
#
# df_clean = df[keep_cols].copy()
#
# # Drop rows with missing survival or features
# print("Initial rows:", len(df_clean))
# df_clean = df_clean.dropna()
# print("Rows after dropping NA:", len(df_clean))
#
# # Save processed file
# df_clean.to_excel(OUTPUT_FILE, index=False)
# print(f"Saved cleaned file to:\n{OUTPUT_FILE}")


#----------------------------------------------------------
#----------- Revised preprocessing ------------------------
#----------------------------------------------------------
import pandas as pd
from pathlib import Path

# --------------------- CONFIG --------------------- #
DATA_DIR = Path(r"D:\3_modelling\pacman\data")
INPUT_FILE = DATA_DIR / "brain_df.xlsx"
OUTPUT_FILE = DATA_DIR / "heatmap_cin_raw_num.xlsx"

id_cols = [
    "slide_id",
    "bcr_patient_barcode",
    "case_id",
]

survival_cols = [
    "OS", "OS.time",
    "DSS", "DSS.time",
    "DFI", "DFI.time",
    "PFI", "PFI.time",
]

mitotic_feats = [
    "mit_hotspot_score",       # HSC
    "mit_nodeDegrees_mean",    # mean(ND)
    "mit_nodeDegrees_cv",      # cv(ND)
    "mit_clusterCoff_mean",    # mean(CL)
    "mit_cenHarmonic_mean",    # mean(HC)
    "aty_hotspot_ratio",       # AMH
    "aty_ahotspot_ratio",      # AMAH
    "aty_wsi_ratio",           # AFW
]

group_cols = [
    "WHO_CNS5_histology",  # Astrocytoma / Oligodendroglioma / Glioblastoma
    "WHO_CNS5_hist",       # IDH-wildtype / IDH-mutant
    "WHO_CNS5_grade",      # Grade 2 / Grade 3 / Grade 4
]

# ---------------------------------------------------- #

print(f"Loading {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

# Keep only needed columns that actually exist
keep_cols = id_cols + survival_cols + mitotic_feats + group_cols
keep_cols = [c for c in keep_cols if c in df.columns]

df_clean = df[keep_cols].copy()

# Strip spaces from column names
df_clean.columns = df_clean.columns.str.strip()

# Strip spaces from string values
for col in df_clean.columns:
    if df_clean[col].dtype == "object":
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].replace({"nan": pd.NA, "None": pd.NA})

# Rename grouping columns to distinct analysis-friendly names
rename_map = {
    "WHO_CNS5_histology": "histology",
    "WHO_CNS5_hist": "idh_status",
    "WHO_CNS5_grade": "grade",
}
df_clean = df_clean.rename(columns=rename_map)

# Optional: make survival/event columns numeric if present
for col in survival_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# Drop rows missing essential grouping or survival information
required_cols = []
for col in [
    "histology",
    "idh_status",
    "grade",
    "OS", "OS.time",
    "DSS", "DSS.time",
    "DFI", "DFI.time",
    "PFI", "PFI.time",
]:
    if col in df_clean.columns:
        required_cols.append(col)

print("Initial rows:", len(df_clean))
df_clean = df_clean.dropna(subset=required_cols)
print("Rows after dropping required NA:", len(df_clean))

# Save cleaned file
df_clean.to_excel(OUTPUT_FILE, index=False)
print(f"Saved cleaned file to:\n{OUTPUT_FILE}")