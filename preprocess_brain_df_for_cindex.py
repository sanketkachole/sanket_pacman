import pandas as pd
from pathlib import Path

# --------------------- CONFIG --------------------- #

DATA_DIR = Path(r"D:\3_modelling\pacman\data")  # adjust if needed
INPUT_FILE = DATA_DIR / "brain_df.xlsx"
OUTPUT_FILE = DATA_DIR / "heatmap_cin_raw_num.xlsx"

# Mitotic features already present in ST1-style data
orig_mitotic_feats = [
    "mit_hotspot_score",       # HSC
    "mit_nodeDegrees_mean",    # mean(ND)
    "mit_nodeDegrees_cv",      # cv(ND)
    "mit_clusterCoff_mean",    # mean(CL)
    "mit_cenHarmonic_mean",    # mean(HC)
    "aty_hotspot_ratio",       # AMH
    "aty_ahotspot_ratio",      # AMAH
    "aty_wsi_ratio",           # AFW
]

# Your NEW metrics extracted from the graph JSONs
new_graph_metrics = [
    "n_nodes",
    "n_edges",
    "density",
    "n_components",
    "largest_component_size",
    "mean_degree",
    "cv_degree",
    "global_clustering",
    "mean_local_clustering",
    "mean_harmonic_cen",
    "max_harmonic_cen",
    "AFW",                       # already overlaps → keep once
    "avg_shortest_path_lcc",
    "diameter_lcc",
    "degree_assortativity"
]

# Survival columns (supported by the author's script)
survival_cols = [
    "OS", "OS.time",
    "DSS", "DSS.time",
    "DFI", "DFI.time",
    "PFI", "PFI.time",
]

# Grouping column for C-index (key: IDH-wt vs IDH-mut)
GROUP_COL = "WHO_CNS5_hist"  # becomes "type" below

# Slide identifier / patient ID columns to keep
id_cols = [
    "slide_id",
    "bcr_patient_barcode",
    "case_id",
]

# ---------------------------------------------------- #

print(f"Loading {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

# Rename grouping column to "type" (required by author’s C-index script)
df = df.rename(columns={GROUP_COL: "type"})

# Build final list of required columns
keep_cols = (
        id_cols +
        survival_cols +
        orig_mitotic_feats +
        new_graph_metrics +
        ["type"]  # grouping column must be included
)

# Keep only columns that actually exist in the file
keep_cols = [c for c in keep_cols if c in df.columns]

df_clean = df[keep_cols].copy()

# Drop rows with missing survival or features
print("Initial rows:", len(df_clean))
df_clean = df_clean.dropna()
print("Rows after dropping NA:", len(df_clean))

# Save processed file
df_clean.to_excel(OUTPUT_FILE, index=False)
print(f"Saved cleaned file to:\n{OUTPUT_FILE}")
