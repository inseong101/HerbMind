#!/usr/bin/env python3
"""Generate the full HerbMind paper figure suite.

This script synthesises publication-style figures that mirror the layout of the
RecipeMind paper.  Running it once produces all eight figures required for the
paper and stores them under the ``figures/`` directory (or a custom directory
provided via ``--output-dir``).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


FIGURE_FILENAMES = {
    "fig1": "fig1_task_overview.png",
    "fig2": "fig2_model_architecture.png",
    "fig3": "fig3_attnmap_caseA.png",
    "fig4": "fig4_attnmap_caseB.png",
    "fig5": "fig5_grid_caseA.png",
    "fig6": "fig6_grid_caseB.png",
    "fig7": "fig7_output_caseA.png",
    "fig8": "fig8_output_caseB.png",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _configure_matplotlib() -> None:
    """Apply a consistent visual style for all figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 120,
        }
    )


def _save_and_close(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Task overview diagram
# ---------------------------------------------------------------------------

def _draw_capsule(ax: plt.Axes, center: Tuple[float, float], width: float, height: float, label: str, color: str) -> None:
    """Draw a rounded capsule at *center* with the given *label*."""
    x = center[0] - width / 2
    y = center[1] - height / 2
    capsule = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size={}".format(height / 2.2),
        linewidth=1.2,
        edgecolor="#2b303a",
        facecolor=color,
    )
    ax.add_patch(capsule)
    ax.text(center[0], center[1], label, ha="center", va="center", fontsize=11, color="#0d1321")


def _figure_task_overview(dpi: int, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("Step 1\nGather base\nherbs", "#cbe7ff"),
        ("Step 2\nAssess pairing\npatterns", "#d8f3dc"),
        ("Step 3\nModel ranking\ncandidates", "#ffe5d9"),
        ("Step 4\nSelect final\nrecipe", "#f4f1de"),
    ]
    y_positions = np.linspace(0.8, 0.2, len(steps))
    for (label, color), y in zip(steps, y_positions):
        _draw_capsule(ax, (0.25, y), 0.28, 0.14, label, color)

    # User / model split
    ax.plot([0.52, 0.52], [0.1, 0.9], color="#94a1b2", linewidth=1.2, linestyle="--")
    ax.text(0.16, 0.88, "Herbalist actions", fontsize=13, fontweight="bold", color="#094067")
    ax.text(0.65, 0.88, "HerbMind model", fontsize=13, fontweight="bold", color="#3d405b")

    # Arrows connecting steps to model space
    for y in y_positions:
        ax.annotate(
            "",
            xy=(0.52, y),
            xytext=(0.39, y),
            arrowprops=dict(arrowstyle="->", color="#ef8354", linewidth=1.2),
        )

    # Ranked table on the right
    columns = ["Rank", "Candidate", "Score"]
    ranked_rows = [
        ["1", "Glycyrrhizae Radix", "0.94"],
        ["2", "Angelicae Gigantis Radix", "0.88"],
        ["3", "Zingiberis Rhizoma", "0.83"],
        ["4", "Cinnamomi Cortex", "0.78"],
        ["5", "Zizyphi Fructus", "0.73"],
    ]
    table = ax.table(
        cellText=ranked_rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.58, 0.25, 0.35, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    ax.text(
        0.59,
        0.75,
        "Top-ranked additions",
        fontsize=12,
        fontweight="bold",
        color="#2f3e46",
    )

    _save_and_close(fig, output_dir / FIGURE_FILENAMES["fig1"], dpi)


# ---------------------------------------------------------------------------
# Figure 2: Cascaded set transformer overview
# ---------------------------------------------------------------------------

def _draw_block(ax: plt.Axes, xy: Tuple[float, float], width: float, height: float, label: str, color: str, fontsize: int = 11) -> patches.FancyBboxPatch:
    block = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="#1d3557",
        facecolor=color,
    )
    ax.add_patch(block)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center", fontsize=fontsize, color="#1b263b")
    return block


def _figure_model_architecture(dpi: int, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_block(ax, (0.05, 0.4), 0.12, 0.2, "Ingredient\nEmbeddings", "#e0fbfc")
    sab_positions = [0.24, 0.39, 0.54]
    sab_colors = ["#ffe5ec", "#fde2e4", "#fad2e1"]
    sab_blocks = []
    for idx, (x, color) in enumerate(zip(sab_positions, sab_colors), start=1):
        sab_blocks.append(_draw_block(ax, (x, 0.4), 0.12, 0.2, f"SAB{idx}\n(Self-Attn)", color))

    pmx_y_positions = [0.15, 0.4, 0.65]
    pmx_colors = ["#dcedc2", "#ffd3b6", "#ffaaa5"]
    for idx, y in enumerate(pmx_y_positions, start=1):
        _draw_block(ax, (0.72, y), 0.18, 0.18, f"PMX{idx}\nPooling + Mix", pmx_colors[idx - 1])

    _draw_block(ax, (0.9, 0.4), 0.08, 0.2, "Scoring\nHead", "#e1f2f7")

    # Connections
    arrow_style = dict(arrowstyle="->", linewidth=1.4, color="#1d3557")
    ax.annotate("", xy=(0.24, 0.5), xytext=(0.17, 0.5), arrowprops=arrow_style)
    ax.annotate("", xy=(0.39, 0.5), xytext=(0.36, 0.5), arrowprops=arrow_style)
    ax.annotate("", xy=(0.54, 0.5), xytext=(0.51, 0.5), arrowprops=arrow_style)
    ax.annotate("", xy=(0.72, 0.24), xytext=(0.66, 0.48), arrowprops=arrow_style)
    ax.annotate("", xy=(0.72, 0.48), xytext=(0.66, 0.48), arrowprops=arrow_style)
    ax.annotate("", xy=(0.72, 0.72), xytext=(0.66, 0.52), arrowprops=arrow_style)

    # From PMX to scoring
    for y in pmx_y_positions:
        ax.annotate("", xy=(0.9, 0.5), xytext=(0.9, y + 0.09), arrowprops=arrow_style)

    # Input and output labels
    ax.text(0.05, 0.65, "Ingredient tokens", fontsize=12, fontweight="bold")
    ax.text(0.89, 0.65, "Affinity\nscore", fontsize=12, fontweight="bold", ha="center")

    # Multi-head attention hints
    for offset in np.linspace(-0.04, 0.04, 4):
        ax.plot(
            [0.22, 0.24],
            [0.55 + offset, 0.55 + offset],
            color="#577590",
            linewidth=0.8,
            alpha=0.6,
        )

    ax.text(0.3, 0.72, "Cascaded Set Transformer", fontsize=15, fontweight="bold", color="#264653")

    _save_and_close(fig, output_dir / FIGURE_FILENAMES["fig2"], dpi)


# ---------------------------------------------------------------------------
# Figure 3 & 4: Attention heatmaps
# ---------------------------------------------------------------------------

def _attention_heatmap(
    case_name: str,
    seed: int,
    dpi: int,
    output_dir: Path,
    filename_key: str,
) -> None:
    rng = np.random.default_rng(seed)
    size = 8
    raw = rng.uniform(0.05, 1.0, size=(size, size))
    matrix = (raw + raw.T) / 2
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    masked_matrix = np.ma.array(matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(masked_matrix, cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"Attention map — {case_name}")
    herb_labels = [
        "1. Gancao",
        "2. Danggui",
        "3. Baishao",
        "4. Huangqi",
        "5. Chenpi",
        "6. Fuling",
        "7. Shengjiang",
        "8. Dazao",
    ]
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(herb_labels, rotation=45, ha="right")
    ax.set_yticklabels(herb_labels)

    for i in range(size):
        for j in range(size):
            if mask[i, j]:
                ax.text(j, i, "?", ha="center", va="center", color="#c7c7c7", fontsize=12, alpha=0.8)
            else:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    ax.set_xlabel("Attended ingredient")
    ax.set_ylabel("Query ingredient")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight")

    _save_and_close(fig, output_dir / FIGURE_FILENAMES[filename_key], dpi)


# ---------------------------------------------------------------------------
# Figure 5 & 6: Ingredient similarity grids
# ---------------------------------------------------------------------------

def _similarity_heatmap(
    case_name: str,
    seed: int,
    dpi: int,
    output_dir: Path,
    filename_key: str,
) -> None:
    rng = np.random.default_rng(seed)
    herbs = [
        "Gancao",
        "Danggui",
        "Baishao",
        "Huangqi",
        "Chenpi",
        "Fuling",
        "Shengjiang",
        "Dazao",
        "Chuanxiong",
        "Rougui",
    ]
    base = rng.uniform(0.15, 1.0, size=(len(herbs), len(herbs)))
    matrix = (base + base.T) / 2
    np.fill_diagonal(matrix, 1.0)

    ranks = rng.permutation(np.arange(1, len(herbs) + 1))
    labels = [f"{rank:02d}. {herb}" for rank, herb in zip(ranks, herbs)]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Ingredient affinity — {case_name}")
    ax.set_xticks(range(len(herbs)))
    ax.set_yticks(range(len(herbs)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(herbs)):
        for j in range(len(herbs)):
            text_color = "white" if matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PMI score")

    _save_and_close(fig, output_dir / FIGURE_FILENAMES[filename_key], dpi)


# ---------------------------------------------------------------------------
# Figure 7 & 8: Stepwise recommendation tables
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    current: Sequence[str]
    suggestions: Sequence[Tuple[str, float]]


def _format_step_table(ax: plt.Axes, title: str, steps: Sequence[StepRecord]) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=20, fontweight="bold", color="#1d3557")

    column_labels = ["Current set", "Top-3 suggestions"]
    cell_rows: List[List[str]] = []
    for idx, record in enumerate(steps, start=1):
        current_text = f"Step {idx}:\n" + "\n".join(record.current)
        suggestion_text = "\n".join(f"{name} ({score:.2f})" for name, score in record.suggestions)
        cell_rows.append([current_text, suggestion_text])

    table = ax.table(
        cellText=cell_rows,
        colLabels=column_labels,
        cellLoc="left",
        colLoc="center",
        loc="center",
        bbox=[0.05, 0.05, 0.9, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Highlight header row
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:
            cell.set_facecolor("#edf2fb")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if row % 2 == 0 else "#ffffff")


def _figure_stepwise_table(
    case_name: str,
    steps: Sequence[StepRecord],
    dpi: int,
    output_dir: Path,
    filename_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    _format_step_table(ax, f"Stepwise generation — {case_name}", steps)
    _save_and_close(fig, output_dir / FIGURE_FILENAMES[filename_key], dpi)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_figures(output_dir: Path, dpi: int = 300) -> None:
    _configure_matplotlib()
    _figure_task_overview(dpi, output_dir)
    _figure_model_architecture(dpi, output_dir)
    _attention_heatmap("Case A", seed=42, dpi=dpi, output_dir=output_dir, filename_key="fig3")
    _attention_heatmap("Case B", seed=84, dpi=dpi, output_dir=output_dir, filename_key="fig4")
    _similarity_heatmap("Case A", seed=4242, dpi=dpi, output_dir=output_dir, filename_key="fig5")
    _similarity_heatmap("Case B", seed=8484, dpi=dpi, output_dir=output_dir, filename_key="fig6")

    case_a_steps = [
        StepRecord(["Danggui", "Baishao"], [("Gancao", 0.92), ("Chuanxiong", 0.83), ("Chenpi", 0.79)]),
        StepRecord(["Danggui", "Baishao", "Gancao"], [("Fuling", 0.88), ("Dazao", 0.81), ("Rougui", 0.74)]),
        StepRecord(["Danggui", "Baishao", "Gancao", "Fuling"], [("Shengjiang", 0.85), ("Chenpi", 0.80), ("Huangqi", 0.76)]),
        StepRecord(["Danggui", "Baishao", "Gancao", "Fuling", "Shengjiang"], [("Huangqi", 0.87), ("Dazao", 0.82), ("Rougui", 0.71)]),
    ]
    case_b_steps = [
        StepRecord(["Huangqi", "Fuling"], [("Dangshen", 0.91), ("Chenpi", 0.84), ("Fangfeng", 0.78)]),
        StepRecord(["Huangqi", "Fuling", "Dangshen"], [("Baizhu", 0.86), ("Gancao", 0.82), ("Shanyao", 0.77)]),
        StepRecord(["Huangqi", "Fuling", "Dangshen", "Baizhu"], [("Chenpi", 0.83), ("Fangfeng", 0.79), ("Chuanxiong", 0.74)]),
        StepRecord(["Huangqi", "Fuling", "Dangshen", "Baizhu", "Chenpi"], [("Shengjiang", 0.82), ("Dazao", 0.78), ("Rougui", 0.72)]),
    ]
    _figure_stepwise_table("Case A", case_a_steps, dpi, output_dir, "fig7")
    _figure_stepwise_table("Case B", case_b_steps, dpi, output_dir, "fig8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the HerbMind publication figures.")
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory where figures will be saved (default: figures).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Dots per inch for saved images (default: 300).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    generate_all_figures(output_dir=output_dir, dpi=args.dpi)
    print(f"[OK] Generated HerbMind figures in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
