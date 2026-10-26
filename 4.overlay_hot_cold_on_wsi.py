#!/usr/bin/env python3
"""
Generate WSI hotspot visualizations with mitotic and atypical hotspot support.

Outputs:
1. One combined 6-panel figure (same as before)
2. Separate images for each panel:
   - panel1_wsi_with_boxes
   - panel2_mitotic_hotspot_crop
   - panel3_atypical_hotspot_crop
   - panel4_combined_overlay
   - panel5_combined_json_graph
   - panel6_combined_fully_connected_graph

Notes:
- Typical mitosis nodes are shown in GREEN.
- Atypical mitosis nodes are shown in RED.
- Mitotic hotspot box is shown in YELLOW.
- Atypical hotspot box is shown in MAGENTA.
- JSON graph edges are used in panel 5.
- Fully connected edges among all nodes in combined ROI are used in panel 6.
- If JSON contains `feat_names` with `type`, the code assumes:
    type == 2 -> atypical
    otherwise -> typical
  If unavailable, it falls back to hotspot-box membership.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# =========================================================
# HARD-CODED PATHS
# =========================================================
DEFAULT_INPUT_EXCEL = r"D:\3_modelling\pacman\data\brain_df.xlsx"
DEFAULT_WSI_ROOT = r"Y:\Scripts\TCGA-COHORT"
DEFAULT_GRAPH_ROOT = r"D:\3_modelling\pacman\data\MDFS_TCGA_mitosis_graphs\graphs_final"
DEFAULT_OUTPUT_DIR = r"D:\3_modelling\pacman\results\wsi_hotspot_visualizations"

# Optional manual DLL path if needed on Windows.
DEFAULT_OPENSLIDE_DLL_DIR = None
# Example:
# DEFAULT_OPENSLIDE_DLL_DIR = r"D:\openslide-win64\bin"

# Default single-slide test
DEFAULT_TARGET_SLIDE_ID = "TCGA-S9-A7J0-01Z-00-DX1.F7AEDF07-33C2-4238-A04F-758C8EF58851"

# =========================================================
# DEFAULT SETTINGS
# =========================================================
DEFAULT_THUMBNAIL_MAX_SIZE = 2500
DEFAULT_TOP_N_PER_GROUP = 3
DEFAULT_FIG_DPI = 300
DEFAULT_RECURSIVE_WSI_GLOB = "*.svs"

TEMPERATURE_COL = "temperature"
SLIDE_ID_COL = "slide_id"
HOTSPOT_COUNT_COL = "mit_hotspot_count"
TYPE_COL = "type"
PROLIFERATION_COL = "Proliferation"

MIT_HOTSPOT_COLS = [
    "mit_hotspot_x1",
    "mit_hotspot_y1",
    "mit_hotspot_x2",
    "mit_hotspot_y2",
]

ATY_HOTSPOT_COLS = [
    "aty_ahotspot_x1",
    "aty_ahotspot_y1",
    "aty_ahotspot_x2",
    "aty_ahotspot_y2",
]

ATY_WSI_COUNT_COL = "aty_wsi_count"
ATY_WSI_RATIO_COL = "aty_wsi_ratio"
ATY_HOTSPOT_COUNT_COL = "aty_hotspot_count"
ATY_HOTSPOT_RATIO_COL = "aty_hotspot_ratio"
ATY_AHOTSPOT_COUNT_COL = "aty_ahotspot_count"
ATY_AHOTSPOT_RATIO_COL = "aty_ahotspot_ratio"
MIT_AHOTSPOT_COUNT_COL = "mit_ahotspot_count"

TYPICAL_NODE_COLOR = "#00FF00"   # green
ATYPICAL_NODE_COLOR = "#FF2D55"  # red/pink
JSON_EDGE_COLOR = "#00E5FF"      # cyan
FULL_EDGE_COLOR = "#FFFFFF"      # white
MIT_HOTSPOT_BOX_COLOR = "#FFD84D"    # yellow
ATY_HOTSPOT_BOX_COLOR = "#FF00FF"    # magenta


def import_openslide(dll_dir: Optional[str] = None):
    if os.name == "nt":
        if dll_dir:
            dll_path = Path(dll_dir)
            if not dll_path.exists():
                raise FileNotFoundError(f"OpenSlide DLL directory not found: {dll_path}")
            os.add_dll_directory(str(dll_path))

        try:
            import openslide  # noqa: WPS433
            return openslide
        except Exception:
            pass

        try:
            import openslide_bin  # noqa: F401,WPS433

            openslide_bin_dir = Path(openslide_bin.__file__).resolve().parent / "bin"
            if openslide_bin_dir.exists():
                os.add_dll_directory(str(openslide_bin_dir))

            import openslide  # noqa: WPS433
            return openslide
        except Exception as exc:
            raise ImportError(
                "Could not import OpenSlide on Windows.\n"
                "Install:\n"
                "  pip install openslide-python openslide-bin\n"
                "Or set DEFAULT_OPENSLIDE_DLL_DIR."
            ) from exc
    else:
        try:
            import openslide  # noqa: WPS433
            return openslide
        except Exception as exc:
            raise ImportError(
                "OpenSlide import failed. Install with:\n"
                "  pip install openslide-python\n"
                "and ensure system libraries are installed."
            ) from exc


openslide = import_openslide(DEFAULT_OPENSLIDE_DLL_DIR)


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    temperature: str
    hotspot_count: float
    mit_hotspot_box: Tuple[int, int, int, int]
    aty_hotspot_box: Optional[Tuple[int, int, int, int]]
    wsi_path: Path
    graph_path: Optional[Path]
    proliferation: Optional[float] = None
    cohort_type: Optional[str] = None
    aty_wsi_count: Optional[float] = None
    aty_wsi_ratio: Optional[float] = None
    aty_hotspot_count: Optional[float] = None
    aty_hotspot_ratio: Optional[float] = None
    aty_ahotspot_count: Optional[float] = None
    aty_ahotspot_ratio: Optional[float] = None
    mit_ahotspot_count: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WSI thumbnail + mitotic/atypical hotspot visualizations.",
    )
    parser.add_argument("--input-excel", type=str, default=DEFAULT_INPUT_EXCEL)
    parser.add_argument("--wsi-root", type=str, default=DEFAULT_WSI_ROOT)
    parser.add_argument("--graph-root", type=str, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n-per-group", type=int, default=DEFAULT_TOP_N_PER_GROUP)
    parser.add_argument("--thumbnail-max-size", type=int, default=DEFAULT_THUMBNAIL_MAX_SIZE)
    parser.add_argument("--figure-dpi", type=int, default=DEFAULT_FIG_DPI)
    parser.add_argument("--filter-type", type=str, default="GBMLGG")
    parser.add_argument("--recursive-wsi-glob", type=str, default=DEFAULT_RECURSIVE_WSI_GLOB)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--slide-id",
        type=str,
        default=DEFAULT_TARGET_SLIDE_ID,
        help="Run only this slide_id. Use ALL to disable single-slide filtering.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        logging.getLogger(__name__).debug("Ignoring unknown args: %s", unknown)
    return args


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_string_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace(
        {
            "nan": np.nan,
            "None": np.nan,
            "": np.nan,
            "[Not Available]": np.nan,
            "[Not Applicable]": np.nan,
        }
    )
    return out


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if TEMPERATURE_COL in df.columns:
        df[TEMPERATURE_COL] = clean_string_series(df[TEMPERATURE_COL]).replace(
            {
                "Mitotic-Hot": "Hot",
                "Mitotic-Cold": "Cold",
                "hot": "Hot",
                "cold": "Cold",
                "HOT": "Hot",
                "COLD": "Cold",
            }
        )

    if SLIDE_ID_COL in df.columns:
        df[SLIDE_ID_COL] = clean_string_series(df[SLIDE_ID_COL])

    if TYPE_COL in df.columns:
        df[TYPE_COL] = clean_string_series(df[TYPE_COL])

    numeric_cols = (
        MIT_HOTSPOT_COLS
        + ATY_HOTSPOT_COLS
        + [
            HOTSPOT_COUNT_COL,
            PROLIFERATION_COL,
            ATY_WSI_COUNT_COL,
            ATY_WSI_RATIO_COL,
            ATY_HOTSPOT_COUNT_COL,
            ATY_HOTSPOT_RATIO_COL,
            ATY_AHOTSPOT_COUNT_COL,
            ATY_AHOTSPOT_RATIO_COL,
            MIT_AHOTSPOT_COUNT_COL,
        ]
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = [SLIDE_ID_COL, TEMPERATURE_COL, HOTSPOT_COUNT_COL] + MIT_HOTSPOT_COLS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")


def _optional_box_from_row(
    row: pd.Series,
    cols: Sequence[str],
) -> Optional[Tuple[int, int, int, int]]:
    if not all(col in row.index for col in cols):
        return None
    values = [row[col] for col in cols]
    if any(pd.isna(v) for v in values):
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in values]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def load_metadata(input_excel: Path, filter_type: str) -> pd.DataFrame:
    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_excel}")

    logging.info("Loading Excel: %s", input_excel)
    df = pd.read_excel(input_excel)
    df = standardize_dataframe(df)
    validate_required_columns(df)

    if filter_type.upper() != "ALL":
        if TYPE_COL not in df.columns:
            raise ValueError(f"'{TYPE_COL}' column is required when --filter-type is used.")
        before = len(df)
        df = df[df[TYPE_COL] == filter_type].copy()
        logging.info("Filtered rows: %s -> %s by %s=%s", before, len(df), TYPE_COL, filter_type)

    df = df[df[TEMPERATURE_COL].isin(["Hot", "Cold"])].copy()
    df = df.dropna(subset=[SLIDE_ID_COL, TEMPERATURE_COL] + MIT_HOTSPOT_COLS).copy()

    before_dedup = len(df)
    df = df.drop_duplicates(subset=[SLIDE_ID_COL], keep="first").copy()
    logging.info("Deduplicated slide rows: %s -> %s", before_dedup, len(df))

    for col in MIT_HOTSPOT_COLS:
        df[col] = df[col].round().astype(int)

    for col in ATY_HOTSPOT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_file_map(root: Path, pattern: str) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    file_map: Dict[str, Path] = {}
    paths = list(root.rglob(pattern))
    logging.info("Found %s WSI files under %s with pattern %s", len(paths), root, pattern)

    for path in paths:
        file_map[path.stem.lower()] = path

    return file_map


def build_json_map(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Graph folder not found: {root}")

    paths = list(root.rglob("*.json"))
    logging.info("Found %s graph json files under %s", len(paths), root)

    file_map: Dict[str, Path] = {}
    for path in paths:
        file_map[path.stem.lower()] = path
    return file_map


def match_slide_files(
    df: pd.DataFrame,
    wsi_map: Dict[str, Path],
    json_map: Dict[str, Path],
) -> List[SlideRecord]:
    records: List[SlideRecord] = []
    missing_wsi = 0
    missing_json = 0

    for _, row in df.iterrows():
        slide_id = str(row[SLIDE_ID_COL]).strip()
        slide_key = slide_id.lower()

        wsi_path = wsi_map.get(slide_key)
        if wsi_path is None:
            missing_wsi += 1
            continue

        graph_path = json_map.get(slide_key)
        if graph_path is None:
            missing_json += 1

        mit_hotspot_box = (
            int(row["mit_hotspot_x1"]),
            int(row["mit_hotspot_y1"]),
            int(row["mit_hotspot_x2"]),
            int(row["mit_hotspot_y2"]),
        )
        aty_hotspot_box = _optional_box_from_row(row, ATY_HOTSPOT_COLS)

        proliferation = None
        if PROLIFERATION_COL in row and not pd.isna(row[PROLIFERATION_COL]):
            proliferation = float(row[PROLIFERATION_COL])

        cohort_type = None
        if TYPE_COL in row and pd.notna(row[TYPE_COL]):
            cohort_type = str(row[TYPE_COL])

        def _num(col: str) -> Optional[float]:
            if col in row and not pd.isna(row[col]):
                return float(row[col])
            return None

        records.append(
            SlideRecord(
                slide_id=slide_id,
                temperature=str(row[TEMPERATURE_COL]),
                hotspot_count=float(row[HOTSPOT_COUNT_COL])
                if not pd.isna(row[HOTSPOT_COUNT_COL])
                else math.nan,
                mit_hotspot_box=mit_hotspot_box,
                aty_hotspot_box=aty_hotspot_box,
                wsi_path=wsi_path,
                graph_path=graph_path,
                proliferation=proliferation,
                cohort_type=cohort_type,
                aty_wsi_count=_num(ATY_WSI_COUNT_COL),
                aty_wsi_ratio=_num(ATY_WSI_RATIO_COL),
                aty_hotspot_count=_num(ATY_HOTSPOT_COUNT_COL),
                aty_hotspot_ratio=_num(ATY_HOTSPOT_RATIO_COL),
                aty_ahotspot_count=_num(ATY_AHOTSPOT_COUNT_COL),
                aty_ahotspot_ratio=_num(ATY_AHOTSPOT_RATIO_COL),
                mit_ahotspot_count=_num(MIT_AHOTSPOT_COUNT_COL),
            )
        )

    logging.info("Matched slide records: %s", len(records))
    logging.info("Missing WSI matches: %s", missing_wsi)
    logging.info("Missing graph JSON matches: %s", missing_json)
    return records


def select_representative_records(
    records: Sequence[SlideRecord],
    top_n_per_group: int,
) -> List[SlideRecord]:
    hot_records = [record for record in records if record.temperature == "Hot"]
    cold_records = [record for record in records if record.temperature == "Cold"]

    hot_records = sorted(
        hot_records,
        key=lambda record: (-np.nan_to_num(record.hotspot_count, nan=-1e9), record.slide_id),
    )[:top_n_per_group]

    cold_records = sorted(
        cold_records,
        key=lambda record: (np.nan_to_num(record.hotspot_count, nan=1e9), record.slide_id),
    )[:top_n_per_group]

    selected = hot_records + cold_records
    logging.info(
        "Selected %s Hot and %s Cold representative slides",
        len(hot_records),
        len(cold_records),
    )
    return selected


def load_graph_json(
    graph_path: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if graph_path is None or not graph_path.exists():
        return (
            np.empty((0, 2), dtype=float),
            np.empty((2, 0), dtype=int),
            None,
        )

    with open(graph_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    coordinates = np.asarray(payload.get("coordinates", []), dtype=float)
    edge_index = np.asarray(payload.get("edge_index", []), dtype=int)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        coordinates = np.empty((0, 2), dtype=float)

    if edge_index.size == 0:
        edge_index = np.empty((2, 0), dtype=int)
    elif edge_index.ndim != 2 or edge_index.shape[0] != 2:
        edge_index = np.empty((2, 0), dtype=int)

    node_types: Optional[np.ndarray] = None
    feats = payload.get("feats", None)
    feat_names = payload.get("feat_names", None)
    if feats is not None and feat_names is not None and len(feat_names) > 0:
        feat_names_list = [str(name).strip() for name in feat_names]
        if "type" in feat_names_list:
            type_idx = feat_names_list.index("type")
            feats_arr = np.asarray(feats, dtype=float)
            if feats_arr.ndim == 2 and feats_arr.shape[0] == coordinates.shape[0] and type_idx < feats_arr.shape[1]:
                node_types = feats_arr[:, type_idx].astype(int)

    return coordinates, edge_index, node_types


def open_slide(wsi_path: Path) -> openslide.OpenSlide:
    return openslide.OpenSlide(str(wsi_path))


def compute_thumbnail_size(width: int, height: int, max_size: int) -> Tuple[int, int]:
    if max(width, height) <= max_size:
        return width, height

    scale = max_size / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def get_thumbnail_array(
    slide: openslide.OpenSlide,
    max_size: int,
) -> Tuple[np.ndarray, float]:
    width, height = slide.dimensions
    thumb_w, thumb_h = compute_thumbnail_size(width, height, max_size)
    thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb_array = np.asarray(thumbnail)

    scale_x = thumb_w / float(width)
    scale_y = thumb_h / float(height)

    if not np.isclose(scale_x, scale_y, atol=1e-6):
        logging.warning("Thumbnail x/y scales differ slightly: %.6f vs %.6f", scale_x, scale_y)

    return thumb_array, scale_x


def clip_box_to_slide(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def union_boxes(
    box_a: Tuple[int, int, int, int],
    box_b: Optional[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    if box_b is None:
        return box_a
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def read_region_rgb(
    slide: openslide.OpenSlide,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    patch = slide.read_region((x1, y1), 0, (width, height)).convert("RGB")
    return np.asarray(patch)


def points_in_box_mask(
    coordinates: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    return (
        (coordinates[:, 0] >= x1)
        & (coordinates[:, 0] < x2)
        & (coordinates[:, 1] >= y1)
        & (coordinates[:, 1] < y2)
    )


def build_node_index_subset(
    coordinates: np.ndarray,
    box: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if coordinates.size == 0:
        return np.empty((0,), dtype=int), np.empty((0, 2), dtype=float)

    mask = points_in_box_mask(coordinates, box)
    node_indices = np.where(mask)[0]
    return node_indices, coordinates[mask]


def filter_edges_to_subset(
    edge_index: np.ndarray,
    node_indices: np.ndarray,
) -> np.ndarray:
    if edge_index.size == 0 or node_indices.size == 0:
        return np.empty((2, 0), dtype=int)

    node_set = set(int(value) for value in node_indices.tolist())
    keep_mask = np.array(
        [(int(src) in node_set and int(dst) in node_set) for src, dst in edge_index.T],
        dtype=bool,
    )
    return edge_index[:, keep_mask]


def build_fully_connected_edges(node_indices: np.ndarray) -> np.ndarray:
    if node_indices.size < 2:
        return np.empty((2, 0), dtype=int)

    edge_pairs = list(itertools.combinations(node_indices.tolist(), 2))
    return np.asarray(edge_pairs, dtype=int).T


def classify_subset_nodes(
    all_coordinates: np.ndarray,
    subset_node_indices: np.ndarray,
    node_types: Optional[np.ndarray],
    mit_hotspot_box: Tuple[int, int, int, int],
    aty_hotspot_box: Optional[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - typical_mask_local
    - atypical_mask_local
    """
    n = len(subset_node_indices)
    if n == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)

    if node_types is not None and len(node_types) == len(all_coordinates):
        subset_types = node_types[subset_node_indices]
        atypical_mask = subset_types == 2
        typical_mask = ~atypical_mask
        return typical_mask, atypical_mask

    subset_coords = all_coordinates[subset_node_indices]
    atypical_mask = np.zeros((n,), dtype=bool)

    if aty_hotspot_box is not None:
        atypical_mask = points_in_box_mask(subset_coords, aty_hotspot_box)

    typical_mask = ~atypical_mask

    if not typical_mask.any():
        typical_mask = ~atypical_mask

    return typical_mask, atypical_mask


def draw_box_on_axis(
    ax: plt.Axes,
    box: Tuple[int, int, int, int],
    edgecolor: str,
    linewidth: float = 2.0,
    scale: float = 1.0,
    offset_xy: Tuple[int, int] = (0, 0),
) -> None:
    x_off, y_off = offset_xy
    x1, y1, x2, y2 = box
    rect = Rectangle(
        ((x1 - x_off) * scale, (y1 - y_off) * scale),
        (x2 - x1) * scale,
        (y2 - y1) * scale,
        fill=False,
        linewidth=linewidth,
        edgecolor=edgecolor,
    )
    ax.add_patch(rect)


def draw_thumbnail_panel(
    ax: plt.Axes,
    thumbnail: np.ndarray,
    mit_hotspot_box: Tuple[int, int, int, int],
    aty_hotspot_box: Optional[Tuple[int, int, int, int]],
    scale: float,
    slide_record: SlideRecord,
) -> None:
    ax.imshow(thumbnail)

    draw_box_on_axis(
        ax=ax,
        box=mit_hotspot_box,
        edgecolor=MIT_HOTSPOT_BOX_COLOR,
        linewidth=2.0,
        scale=scale,
    )

    if aty_hotspot_box is not None:
        draw_box_on_axis(
            ax=ax,
            box=aty_hotspot_box,
            edgecolor=ATY_HOTSPOT_BOX_COLOR,
            linewidth=2.0,
            scale=scale,
        )

    subtitle = f"{slide_record.temperature} | mit_hotspot_count={slide_record.hotspot_count:g}"
    if slide_record.aty_ahotspot_count is not None:
        subtitle += f" | aty_ahotspot_count={slide_record.aty_ahotspot_count:g}"

    ax.set_title(f"WSI with both boxes\n{subtitle}", fontsize=10, fontweight="bold")
    ax.axis("off")


def draw_crop_panel(
    ax: plt.Axes,
    crop: Optional[np.ndarray],
    title: str,
) -> None:
    if crop is None:
        ax.text(0.5, 0.5, "Not available", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    ax.imshow(crop)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def _scatter_typical_atypical(
    ax: plt.Axes,
    local_coords: np.ndarray,
    typical_mask_local: np.ndarray,
    atypical_mask_local: np.ndarray,
) -> None:
    if local_coords.size == 0:
        return

    if typical_mask_local.any():
        ax.scatter(
            local_coords[typical_mask_local, 0],
            local_coords[typical_mask_local, 1],
            s=24,
            c=TYPICAL_NODE_COLOR,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.98,
            label="Typical",
            zorder=3,
        )

    if atypical_mask_local.any():
        ax.scatter(
            local_coords[atypical_mask_local, 0],
            local_coords[atypical_mask_local, 1],
            s=28,
            c=ATYPICAL_NODE_COLOR,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.98,
            label="Atypical",
            zorder=4,
        )


def draw_combined_overlay_panel(
    ax: plt.Axes,
    crop: np.ndarray,
    combined_box: Tuple[int, int, int, int],
    mit_hotspot_box: Tuple[int, int, int, int],
    aty_hotspot_box: Optional[Tuple[int, int, int, int]],
    subset_coordinates: np.ndarray,
    typical_mask_local: np.ndarray,
    atypical_mask_local: np.ndarray,
    title: str,
) -> None:
    ax.imshow(crop)

    cx1, cy1, _, _ = combined_box
    local_coords = subset_coordinates.copy()
    if local_coords.size > 0:
        local_coords[:, 0] -= cx1
        local_coords[:, 1] -= cy1

    draw_box_on_axis(
        ax=ax,
        box=mit_hotspot_box,
        edgecolor=MIT_HOTSPOT_BOX_COLOR,
        linewidth=2.0,
        offset_xy=(cx1, cy1),
    )
    if aty_hotspot_box is not None:
        draw_box_on_axis(
            ax=ax,
            box=aty_hotspot_box,
            edgecolor=ATY_HOTSPOT_BOX_COLOR,
            linewidth=2.0,
            offset_xy=(cx1, cy1),
        )

    _scatter_typical_atypical(ax, local_coords, typical_mask_local, atypical_mask_local)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def draw_graph_overlay_panel(
    ax: plt.Axes,
    crop: np.ndarray,
    combined_box: Tuple[int, int, int, int],
    mit_hotspot_box: Tuple[int, int, int, int],
    aty_hotspot_box: Optional[Tuple[int, int, int, int]],
    subset_node_indices: np.ndarray,
    subset_coordinates: np.ndarray,
    typical_mask_local: np.ndarray,
    atypical_mask_local: np.ndarray,
    subset_edges: np.ndarray,
    edge_color: str,
    title: str,
) -> None:
    ax.imshow(crop)
    cx1, cy1, _, _ = combined_box

    local_coords = subset_coordinates.copy()
    if local_coords.size > 0:
        local_coords[:, 0] -= cx1
        local_coords[:, 1] -= cy1

    draw_box_on_axis(
        ax=ax,
        box=mit_hotspot_box,
        edgecolor=MIT_HOTSPOT_BOX_COLOR,
        linewidth=2.0,
        offset_xy=(cx1, cy1),
    )
    if aty_hotspot_box is not None:
        draw_box_on_axis(
            ax=ax,
            box=aty_hotspot_box,
            edgecolor=ATY_HOTSPOT_BOX_COLOR,
            linewidth=2.0,
            offset_xy=(cx1, cy1),
        )

    global_to_local: Dict[int, int] = {
        int(global_idx): local_idx
        for local_idx, global_idx in enumerate(subset_node_indices.tolist())
    }

    if subset_edges.size > 0:
        for src_global, dst_global in subset_edges.T:
            src_local_idx = global_to_local.get(int(src_global))
            dst_local_idx = global_to_local.get(int(dst_global))
            if src_local_idx is None or dst_local_idx is None:
                continue

            src = local_coords[src_local_idx]
            dst = local_coords[dst_local_idx]
            ax.plot(
                [src[0], dst[0]],
                [src[1], dst[1]],
                color=edge_color,
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )

    _scatter_typical_atypical(ax, local_coords, typical_mask_local, atypical_mask_local)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def save_single_panel_figure(
    output_dir: Path,
    temperature: str,
    slide_id: str,
    panel_name: str,
    figure_dpi: int,
    draw_fn,
) -> Tuple[str, str]:
    temp_dir = output_dir / temperature / "separate_panels"
    ensure_dir(temp_dir)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    draw_fn(ax)

    png_path = temp_dir / f"{slide_id}_{panel_name}.png"
    pdf_path = temp_dir / f"{slide_id}_{panel_name}.pdf"

    fig.savefig(png_path, dpi=figure_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)

    return str(png_path), str(pdf_path)


def create_single_slide_figure(
    slide_record: SlideRecord,
    output_dir: Path,
    thumbnail_max_size: int,
    figure_dpi: int,
) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "slide_id": slide_record.slide_id,
        "temperature": slide_record.temperature,
        "wsi_path": str(slide_record.wsi_path),
        "graph_path": str(slide_record.graph_path) if slide_record.graph_path else "",
        "mit_hotspot_count": slide_record.hotspot_count,
        "aty_ahotspot_count": slide_record.aty_ahotspot_count,
        "mit_ahotspot_count": slide_record.mit_ahotspot_count,
        "proliferation": slide_record.proliferation,
        "combined_node_count": 0,
        "json_edge_count": 0,
        "full_edge_count": 0,
    }

    slide = open_slide(slide_record.wsi_path)

    try:
        slide_w, slide_h = slide.dimensions

        mit_hotspot_box = clip_box_to_slide(slide_record.mit_hotspot_box, slide_w, slide_h)
        aty_hotspot_box = (
            clip_box_to_slide(slide_record.aty_hotspot_box, slide_w, slide_h)
            if slide_record.aty_hotspot_box is not None
            else None
        )
        combined_box = clip_box_to_slide(
            union_boxes(mit_hotspot_box, aty_hotspot_box),
            slide_w,
            slide_h,
        )

        thumbnail, thumb_scale = get_thumbnail_array(slide, thumbnail_max_size)
        mit_crop = read_region_rgb(slide, mit_hotspot_box)
        aty_crop = read_region_rgb(slide, aty_hotspot_box) if aty_hotspot_box is not None else None
        combined_crop = read_region_rgb(slide, combined_box)

        all_coords, all_edges, node_types = load_graph_json(slide_record.graph_path)

        subset_node_indices, subset_coords = build_node_index_subset(all_coords, combined_box)
        typical_mask_local, atypical_mask_local = classify_subset_nodes(
            all_coordinates=all_coords,
            subset_node_indices=subset_node_indices,
            node_types=node_types,
            mit_hotspot_box=mit_hotspot_box,
            aty_hotspot_box=aty_hotspot_box,
        )

        subset_json_edges = filter_edges_to_subset(all_edges, subset_node_indices)
        subset_full_edges = build_fully_connected_edges(subset_node_indices)

        stats["combined_node_count"] = int(len(subset_coords))
        stats["json_edge_count"] = int(subset_json_edges.shape[1]) if subset_json_edges.size > 0 else 0
        stats["full_edge_count"] = int(subset_full_edges.shape[1]) if subset_full_edges.size > 0 else 0

        # =========================
        # Combined 6-panel figure
        # =========================
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(20, 12),
            constrained_layout=True,
        )
        axes = axes.flatten()

        suptitle = (
            f"{slide_record.slide_id}\n"
            f"{slide_record.temperature} slide"
            f" | mit_hotspot_count={slide_record.hotspot_count:g}"
        )
        if slide_record.aty_ahotspot_count is not None:
            suptitle += f" | aty_ahotspot_count={slide_record.aty_ahotspot_count:g}"
        if slide_record.mit_ahotspot_count is not None:
            suptitle += f" | mit_ahotspot_count={slide_record.mit_ahotspot_count:g}"
        if slide_record.proliferation is not None:
            suptitle += f" | proliferation={slide_record.proliferation:.3f}"

        fig.suptitle(suptitle, fontsize=13, fontweight="bold")

        draw_thumbnail_panel(
            ax=axes[0],
            thumbnail=thumbnail,
            mit_hotspot_box=mit_hotspot_box,
            aty_hotspot_box=aty_hotspot_box,
            scale=thumb_scale,
            slide_record=slide_record,
        )

        draw_crop_panel(
            ax=axes[1],
            crop=mit_crop,
            title=f"Mitotic hotspot crop\nmit_hotspot_count={slide_record.hotspot_count:g}",
        )

        aty_title = "Atypical hotspot crop"
        if slide_record.aty_ahotspot_count is not None:
            aty_title += f"\naty_ahotspot_count={slide_record.aty_ahotspot_count:g}"
        if slide_record.mit_ahotspot_count is not None:
            aty_title += f" | mit_ahotspot_count={slide_record.mit_ahotspot_count:g}"

        draw_crop_panel(
            ax=axes[2],
            crop=aty_crop,
            title=aty_title,
        )

        draw_combined_overlay_panel(
            ax=axes[3],
            crop=combined_crop,
            combined_box=combined_box,
            mit_hotspot_box=mit_hotspot_box,
            aty_hotspot_box=aty_hotspot_box,
            subset_coordinates=subset_coords,
            typical_mask_local=typical_mask_local,
            atypical_mask_local=atypical_mask_local,
            title=(
                f"Combined overlay\n"
                f"Typical={int(typical_mask_local.sum())} | "
                f"Atypical={int(atypical_mask_local.sum())}"
            ),
        )

        draw_graph_overlay_panel(
            ax=axes[4],
            crop=combined_crop,
            combined_box=combined_box,
            mit_hotspot_box=mit_hotspot_box,
            aty_hotspot_box=aty_hotspot_box,
            subset_node_indices=subset_node_indices,
            subset_coordinates=subset_coords,
            typical_mask_local=typical_mask_local,
            atypical_mask_local=atypical_mask_local,
            subset_edges=subset_json_edges,
            edge_color=JSON_EDGE_COLOR,
            title=f"Combined + JSON graph\nedges={stats['json_edge_count']}",
        )

        draw_graph_overlay_panel(
            ax=axes[5],
            crop=combined_crop,
            combined_box=combined_box,
            mit_hotspot_box=mit_hotspot_box,
            aty_hotspot_box=aty_hotspot_box,
            subset_node_indices=subset_node_indices,
            subset_coordinates=subset_coords,
            typical_mask_local=typical_mask_local,
            atypical_mask_local=atypical_mask_local,
            subset_edges=subset_full_edges,
            edge_color=FULL_EDGE_COLOR,
            title=f"Combined + fully connected graph\nedges={stats['full_edge_count']}",
        )

        temp_dir = output_dir / slide_record.temperature
        ensure_dir(temp_dir)

        png_path = temp_dir / f"{slide_record.slide_id}_mitotic_atypical_visualization.png"
        pdf_path = temp_dir / f"{slide_record.slide_id}_mitotic_atypical_visualization.pdf"

        fig.savefig(png_path, dpi=figure_dpi, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)

        stats["png_path"] = str(png_path)
        stats["pdf_path"] = str(pdf_path)

        # =========================
        # Separate panel images
        # =========================
        aty_title_single = "Atypical hotspot crop"
        if slide_record.aty_ahotspot_count is not None:
            aty_title_single += f"\naty_ahotspot_count={slide_record.aty_ahotspot_count:g}"
        if slide_record.mit_ahotspot_count is not None:
            aty_title_single += f" | mit_ahotspot_count={slide_record.mit_ahotspot_count:g}"

        panel_outputs = {}

        panel_outputs["panel1_wsi_with_boxes_png"], panel_outputs["panel1_wsi_with_boxes_pdf"] = save_single_panel_figure(
            output_dir=output_dir,
            temperature=slide_record.temperature,
            slide_id=slide_record.slide_id,
            panel_name="panel1_wsi_with_boxes",
            figure_dpi=figure_dpi,
            draw_fn=lambda ax: draw_thumbnail_panel(
                ax=ax,
                thumbnail=thumbnail,
                mit_hotspot_box=mit_hotspot_box,
                aty_hotspot_box=aty_hotspot_box,
                scale=thumb_scale,
                slide_record=slide_record,
            ),
        )

        panel_outputs["panel2_mitotic_hotspot_crop_png"], panel_outputs["panel2_mitotic_hotspot_crop_pdf"] = save_single_panel_figure(
            output_dir=output_dir,
            temperature=slide_record.temperature,
            slide_id=slide_record.slide_id,
            panel_name="panel2_mitotic_hotspot_crop",
            figure_dpi=figure_dpi,
            draw_fn=lambda ax: draw_crop_panel(
                ax=ax,
                crop=mit_crop,
                title=f"Mitotic hotspot crop\nmit_hotspot_count={slide_record.hotspot_count:g}",
            ),
        )

        panel_outputs["panel3_atypical_hotspot_crop_png"], panel_outputs["panel3_atypical_hotspot_crop_pdf"] = save_single_panel_figure(
            output_dir=output_dir,
            temperature=slide_record.temperature,
            slide_id=slide_record.slide_id,
            panel_name="panel3_atypical_hotspot_crop",
            figure_dpi=figure_dpi,
            draw_fn=lambda ax: draw_crop_panel(
                ax=ax,
                crop=aty_crop,
                title=aty_title_single,
            ),
        )

        panel_outputs["panel4_combined_overlay_png"], panel_outputs["panel4_combined_overlay_pdf"] = save_single_panel_figure(
            output_dir=output_dir,
            temperature=slide_record.temperature,
            slide_id=slide_record.slide_id,
            panel_name="panel4_combined_overlay",
            figure_dpi=figure_dpi,
            draw_fn=lambda ax: draw_combined_overlay_panel(
                ax=ax,
                crop=combined_crop,
                combined_box=combined_box,
                mit_hotspot_box=mit_hotspot_box,
                aty_hotspot_box=aty_hotspot_box,
                subset_coordinates=subset_coords,
                typical_mask_local=typical_mask_local,
                atypical_mask_local=atypical_mask_local,
                title=(
                    f"Combined overlay\n"
                    f"Typical={int(typical_mask_local.sum())} | "
                    f"Atypical={int(atypical_mask_local.sum())}"
                ),
            ),
        )

        panel_outputs["panel5_combined_json_graph_png"], panel_outputs["panel5_combined_json_graph_pdf"] = save_single_panel_figure(
            output_dir=output_dir,
            temperature=slide_record.temperature,
            slide_id=slide_record.slide_id,
            panel_name="panel5_combined_json_graph",
            figure_dpi=figure_dpi,
            draw_fn=lambda ax: draw_graph_overlay_panel(
                ax=ax,
                crop=combined_crop,
                combined_box=combined_box,
                mit_hotspot_box=mit_hotspot_box,
                aty_hotspot_box=aty_hotspot_box,
                subset_node_indices=subset_node_indices,
                subset_coordinates=subset_coords,
                typical_mask_local=typical_mask_local,
                atypical_mask_local=atypical_mask_local,
                subset_edges=subset_json_edges,
                edge_color=JSON_EDGE_COLOR,
                title=f"Combined + JSON graph\nedges={stats['json_edge_count']}",
            ),
        )

        panel_outputs["panel6_combined_fully_connected_graph_png"], panel_outputs["panel6_combined_fully_connected_graph_pdf"] = save_single_panel_figure(
            output_dir=output_dir,
            temperature=slide_record.temperature,
            slide_id=slide_record.slide_id,
            panel_name="panel6_combined_fully_connected_graph",
            figure_dpi=figure_dpi,
            draw_fn=lambda ax: draw_graph_overlay_panel(
                ax=ax,
                crop=combined_crop,
                combined_box=combined_box,
                mit_hotspot_box=mit_hotspot_box,
                aty_hotspot_box=aty_hotspot_box,
                subset_node_indices=subset_node_indices,
                subset_coordinates=subset_coords,
                typical_mask_local=typical_mask_local,
                atypical_mask_local=atypical_mask_local,
                subset_edges=subset_full_edges,
                edge_color=FULL_EDGE_COLOR,
                title=f"Combined + fully connected graph\nedges={stats['full_edge_count']}",
            ),
        )

        stats.update(panel_outputs)
        return stats

    finally:
        slide.close()


def create_summary_csv(
    slide_stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if slide_stats_df.empty:
        return

    slide_stats_df.to_csv(output_dir / "selected_slides_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_excel = Path(args.input_excel)
    wsi_root = Path(args.wsi_root)
    graph_root = Path(args.graph_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    logging.info("Python executable: %s", sys.executable)
    logging.info("Loading OpenSlide from module: %s", getattr(openslide, "__file__", "unknown"))

    df = load_metadata(input_excel=input_excel, filter_type=args.filter_type)

    wsi_map = build_file_map(wsi_root, args.recursive_wsi_glob)
    json_map = build_json_map(graph_root)

    records = match_slide_files(df=df, wsi_map=wsi_map, json_map=json_map)
    if not records:
        raise RuntimeError("No slide records could be matched to the WSI tree.")

    if args.slide_id.upper() != "ALL":
        selected_records = [record for record in records if record.slide_id == args.slide_id]
        if not selected_records:
            raise RuntimeError(f"Slide not found: {args.slide_id}")
        logging.info("Running single-slide mode for: %s", args.slide_id)
    else:
        selected_records = select_representative_records(
            records=records,
            top_n_per_group=args.top_n_per_group,
        )
        if not selected_records:
            raise RuntimeError("No representative records selected.")

    logging.info("Generating figures for %s slide(s)", len(selected_records))
    all_stats: List[Dict[str, object]] = []

    for idx, record in enumerate(selected_records, start=1):
        logging.info(
            "[%s/%s] Processing %s (%s)",
            idx,
            len(selected_records),
            record.slide_id,
            record.temperature,
        )
        try:
            stats = create_single_slide_figure(
                slide_record=record,
                output_dir=output_dir,
                thumbnail_max_size=args.thumbnail_max_size,
                figure_dpi=args.figure_dpi,
            )
            all_stats.append(stats)
        except Exception as exc:
            logging.exception("Failed for slide %s: %s", record.slide_id, exc)

    stats_df = pd.DataFrame(all_stats)
    create_summary_csv(stats_df, output_dir)

    if not stats_df.empty:
        logging.info("Done. Saved %s figure set(s) to %s", len(stats_df), output_dir)
        logging.info("Summary CSV: %s", output_dir / "selected_slides_summary.csv")
    else:
        logging.warning("No figures were generated successfully.")


if __name__ == "__main__":
    main()