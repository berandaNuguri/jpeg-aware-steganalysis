#!/usr/bin/env python3
"""Analyze how target QTables scale unit QDCT perturbations.

The analysis is intentionally QTable-level, so it does not scan the full image
dataset. For a quantized DCT coefficient X(u,v), RWNet's DQDCT construction uses
X(u,v) * Q(u,v). Therefore a unit perturbation in the quantized coefficient
domain changes the DQDCT input by Q(u,v). The resulting heatmaps are a compact
way to visualize representation-level QTable alignment.
"""

from __future__ import annotations
from jpeg_aware_steganalysis._paths import project_root, RESOURCE_ROOT

import argparse
import csv
import itertools
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


PROJECT_ROOT = project_root()


def _candidate_qtable_paths() -> List[Path]:
    workspace_root = PROJECT_ROOT.parents[1]
    return [
        workspace_root / "data/NSR_Crawling_2025/JPG_Test/qtable_info.pkl",
        workspace_root / "data/CC12M/qtable_info.pkl",
        workspace_root / "data/CC12M/JPG_10000/qtable_info.pkl",
        PROJECT_ROOT / "qtable_info.pkl",
    ]


def resolve_qtable_info_path(path: str | None) -> Path:
    if path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"QTable info file does not exist: {resolved}")
        return resolved

    for candidate in _candidate_qtable_paths():
        if candidate.exists():
            return candidate

    candidates = "\n".join(f"  - {p}" for p in _candidate_qtable_paths())
    raise FileNotFoundError(
        "Could not find qtable_info.pkl. Pass --qtable-info-path explicitly.\n"
        f"Checked:\n{candidates}"
    )


def normalize_target_name(target: str) -> List[str]:
    target = target.strip()
    candidates = [target]
    if not target.startswith("QT"):
        candidates.append(f"QT{target}")
    return candidates


def load_qtables(path: Path, targets: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    with path.open("rb") as f:
        stats = pickle.load(f)

    if not isinstance(stats, Mapping):
        raise TypeError(f"Expected a dict-like qtable_info.pkl, got {type(stats)!r}")

    target_tables: Dict[str, Dict[str, np.ndarray]] = {}
    for target in targets:
        matched_key = None
        for candidate in normalize_target_name(target):
            if candidate in stats:
                matched_key = candidate
                break
        if matched_key is None:
            raise KeyError(f"Target {target!r} was not found in {path}")

        entry = stats[matched_key]
        qtable = entry.get("qtable") if isinstance(entry, Mapping) else entry
        arr = np.asarray(qtable, dtype=np.float32).reshape(-1)
        if arr.size < 64:
            raise ValueError(f"{matched_key} has only {arr.size} QTable values")

        component_tables: Dict[str, np.ndarray] = {"Y": arr[:64].reshape(8, 8)}
        if arr.size >= 128:
            component_tables["CbCr"] = arr[64:128].reshape(8, 8)
        target_tables[matched_key] = component_tables

    return target_tables


def low_frequency_mask(size: int = 8, block: int = 4) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[:block, :block] = True
    return mask


def summarize_table(target: str, component: str, table: np.ndarray) -> Dict[str, float | str]:
    lf_mask = low_frequency_mask()
    ac_mask = np.ones_like(table, dtype=bool)
    ac_mask[0, 0] = False
    hf_mask = ~lf_mask

    return {
        "target": target,
        "component": component,
        "mean": float(np.mean(table)),
        "std": float(np.std(table)),
        "min": float(np.min(table)),
        "max": float(np.max(table)),
        "dc": float(table[0, 0]),
        "ac_mean": float(np.mean(table[ac_mask])),
        "low_freq_4x4_mean": float(np.mean(table[lf_mask])),
        "high_freq_mean": float(np.mean(table[hf_mask])),
        "high_low_ratio": float(np.mean(table[hf_mask]) / np.mean(table[lf_mask])),
    }


def write_summary_csv(rows: Sequence[Mapping[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "component",
        "mean",
        "std",
        "min",
        "max",
        "dc",
        "ac_mean",
        "low_freq_4x4_mean",
        "high_freq_mean",
        "high_low_ratio",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pairwise_csv(
    tables: Mapping[str, Mapping[str, np.ndarray]],
    component: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for left, right in itertools.combinations(tables.keys(), 2):
        a = tables[left][component].reshape(-1)
        b = tables[right][component].reshape(-1)
        rows.append(
            {
                "component": component,
                "left": left,
                "right": right,
                "l1_mean": float(np.mean(np.abs(a - b))),
                "l2": float(np.linalg.norm(a - b)),
                "cosine_distance": float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
            }
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["component", "left", "right", "l1_mean", "l2", "cosine_distance"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _qtable_colormap() -> colors.LinearSegmentedColormap:
    return colors.LinearSegmentedColormap.from_list(
        "pastel_blue_gray_red",
        [
            (0.0, "#5c7fa8"),
            (0.5, "#d3d3d3"),
            (1.0, "#cf6f69"),
        ],
    )


def _text_color_for_cell(cmap: colors.Colormap, norm: colors.Normalize, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.5 else "#202020"


def _annotate_heatmap(
    ax: plt.Axes,
    table: np.ndarray,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    fontsize: int = 9,
) -> None:
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            ax.text(
                col,
                row,
                f"{int(table[row, col])}",
                ha="center",
                va="center",
                color=_text_color_for_cell(cmap, norm, float(table[row, col])),
                fontsize=fontsize,
                fontweight="normal",
            )


def plot_component_heatmaps(
    tables: Mapping[str, Mapping[str, np.ndarray]],
    component: str,
    output_path: Path,
) -> None:
    ncols = len(tables)
    vmax = max(float(np.max(parts[component])) for parts in tables.values())
    fig, axes = plt.subplots(1, ncols, figsize=(3.9 * ncols, 4.1), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    cmap = _qtable_colormap()
    norm = colors.Normalize(vmin=0, vmax=vmax)
    image = None
    for ax, (target, component_tables) in zip(axes, tables.items()):
        table = component_tables[component]
        image = ax.imshow(table, cmap=cmap, norm=norm)
        _annotate_heatmap(ax, table, cmap, norm)
        ax.set_title(target)
        ax.set_xlabel("DCT frequency v")
        ax.set_ylabel("DCT frequency u")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))

    fig.suptitle(f"{component} channel QDCT perturbation scale", fontsize=13)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.028, pad=0.02, label="DQDCT change for +1 QDCT coefficient")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_component_heatmaps(
    tables: Mapping[str, Mapping[str, np.ndarray]],
    components: Sequence[str],
    output_path: Path,
) -> None:
    nrows = len(components)
    ncols = len(tables)
    vmax = max(float(np.max(parts[component])) for parts in tables.values() for component in components)

    fig = plt.figure(figsize=(3.35 * ncols + 0.9, 3.40 * nrows + 0.25))
    grid = fig.add_gridspec(
        nrows * 2,
        ncols,
        height_ratios=[0.15, 1.0] * nrows,
        left=0.06,
        right=0.88,
        top=0.96,
        bottom=0.07,
        wspace=0.30,
        hspace=0.26,
    )
    cmap = _qtable_colormap()
    norm = colors.Normalize(vmin=0, vmax=vmax)
    image = None
    row_titles = {
        "Y": "Y channel QDCT perturbation scale",
        "CbCr": "CbCr channel QDCT perturbation scale",
    }

    for row_idx, component in enumerate(components):
        row_title = row_titles.get(component, f"{component} channel QDCT perturbation scale")
        title_ax = fig.add_subplot(grid[2 * row_idx, :])
        title_ax.axis("off")
        title_ax.text(0.5, 0.5, row_title, ha="center", va="center", fontsize=12, fontweight="semibold")
        for col_idx, (target, component_tables) in enumerate(tables.items()):
            ax = fig.add_subplot(grid[2 * row_idx + 1, col_idx])
            table = component_tables[component]
            image = ax.imshow(table, cmap=cmap, norm=norm)
            _annotate_heatmap(ax, table, cmap, norm, fontsize=8)
            ax.set_title(target, fontsize=11, pad=6)
            ax.set_xlabel("DCT frequency v" if row_idx == nrows - 1 else "", fontsize=10)
            ax.set_ylabel("DCT frequency u", fontsize=10)
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))

    if image is not None:
        cbar_ax = fig.add_axes([0.91, 0.17, 0.018, 0.66])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("DQDCT change for +1 QDCT coefficient", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_frequency_summary(rows: Sequence[Mapping[str, float | str]], output_path: Path) -> None:
    row_components = {str(row["component"]) for row in rows}
    components = [component for component in ["Y", "CbCr"] if component in row_components]
    targets = [str(row["target"]) for row in rows if row["component"] == components[0]]
    metrics = ["low_freq_4x4_mean", "high_freq_mean"]

    fig, axes = plt.subplots(1, len(components), figsize=(4.2 * len(components), 3.3), sharey=False)
    if len(components) == 1:
        axes = [axes]

    width = 0.36
    x = np.arange(len(targets))
    for ax, component in zip(axes, components):
        component_rows = [row for row in rows if row["component"] == component]
        row_map = {str(row["target"]): row for row in component_rows}
        for offset, metric in zip([-width / 2, width / 2], metrics):
            values = [float(row_map[target][metric]) for target in targets]
            label = "Low freq. 4x4" if metric == "low_freq_4x4_mean" else "High freq."
            ax.bar(x + offset, values, width=width, label=label)
        ax.set_title(component)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=20)
        ax.set_ylabel("Mean QTable value")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Frequency-band scaling induced by target QTables", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize target-QTable scaling of unit QDCT perturbations.",
    )
    parser.add_argument(
        "--qtable-info-path",
        type=str,
        default=None,
        help="Path to qtable_info.pkl. If omitted, common project data paths are searched.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["75_1", "90_1", "95_1"],
        help="Target QTable names, with or without the QT prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/analysis/qtable_unit_perturbation"),
        help="Directory for CSV summaries and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qtable_info_path = resolve_qtable_info_path(args.qtable_info_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    tables = load_qtables(qtable_info_path, args.targets)

    rows = []
    available_components = next(iter(tables.values())).keys()
    components = [component for component in ["Y", "CbCr"] if component in available_components]
    for target, component_tables in tables.items():
        for component, table in component_tables.items():
            rows.append(summarize_table(target, component, table))

    summary_csv = output_dir / "unit_perturbation_summary.csv"
    write_summary_csv(rows, summary_csv)

    for component in components:
        plot_component_heatmaps(
            tables,
            component,
            output_dir / f"unit_perturbation_{component.lower()}_heatmaps.png",
        )
        write_pairwise_csv(
            tables,
            component,
            output_dir / f"qtable_pairwise_distance_{component.lower()}.csv",
        )

    if len(components) > 1:
        plot_combined_component_heatmaps(
            tables,
            components,
            output_dir / "unit_perturbation_y_cbcr_heatmaps.png",
        )

    plot_frequency_summary(rows, output_dir / "frequency_band_scaling_summary.png")

    print(f"Loaded QTables from: {qtable_info_path}")
    print(f"Targets: {', '.join(tables.keys())}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote figures under: {output_dir}")
    print("\nSummary:")
    for row in rows:
        print(
            f"  {row['target']} {row['component']}: "
            f"mean={row['mean']:.2f}, low4x4={row['low_freq_4x4_mean']:.2f}, "
            f"high={row['high_freq_mean']:.2f}, ratio={row['high_low_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
