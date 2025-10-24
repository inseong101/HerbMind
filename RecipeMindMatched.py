#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RecipeMindMatched.py

Generate publication-quality figures that closely match the layouts of
RecipeMind (CIKM'22) figures. The script relies on optional metrics in
``outputs/metrics.json`` and the stepwise configuration in ``cases.yaml``.
When the required data are absent, visually plausible placeholders are used
so that the figure layouts remain inspectable.

Produced figures:
- fig1_overview_matched.png         – Overview split panel (User vs. HerbMind)
- fig2_spmir_kde_matched.png        – sPMIr KDE line curves (|S| buckets)
- fig3_cascaded_arch_matched.png    – Cascaded SAB/PMX architecture diagram
- fig5_baselines_heat_matched.png   – Baseline performance heat tables
- fig6_ablation_heat_matched.png    – Ablation study heat tables
- fig7_case*_stepwise_matched.png   – Stepwise two-table cases (A/B)
- fig9_case*_attn_matched.png       – Triangular attention heatmaps (A/B)
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from StepwiseCases import (
    build_counts,
    generate_case_summary,
    load_cases,
    load_metrics as load_metrics_json,
    load_model_scorer,
    load_prescriptions,
    render_triangular_attention,
    spmir_score,
)

FIG_DIR = "figures"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_fig_dir() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_metrics(path: str = "outputs/metrics.json") -> Dict:
    return load_metrics_json(path)


# ---------------------------------------------------------------------------
# Figure 1 – Overview panel
# ---------------------------------------------------------------------------

def fig1_overview_matched(metrics: Dict) -> None:
    ensure_fig_dir()
    configure_style()

    overview = metrics.get("overview", {})
    steps = overview.get(
        "steps",
        [
            {"label": "Step 1", "text": "Seed Herbs"},
            {"label": "Step 2", "text": "Retrieve Set Context"},
            {"label": "Step 3", "text": "Rank Candidates"},
            {"label": "Step 4", "text": "Select Addition"},
        ],
    )
    ranked = overview.get(
        "ranking",
        [("숙지황", 0.92), ("산수유", 0.90), ("산약", 0.88), ("당귀", 0.85), ("백출", 0.83)],
    )
    affinity = float(overview.get("affinity", 0.91))

    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.05, 0.92, "Recipe Ideation", fontsize=18, fontweight="bold")
    ax.text(0.05, 0.87, "Step-wise ingredient expansion with HerbMind", fontsize=11)

    # Step capsules (left column)
    capsule_colors = ["#FFE7C2", "#FFD199", "#FFB347", "#FF8A00"]
    for idx, step in enumerate(steps):
        y = 0.8 - idx * 0.14
        color = capsule_colors[idx % len(capsule_colors)]
        capsule = patches.FancyBboxPatch(
            (0.05, y),
            0.22,
            0.1,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5,
            edgecolor="#E6893C",
            facecolor=color,
        )
        ax.add_patch(capsule)
        ax.text(0.16, y + 0.065, step.get("label", f"Step {idx+1}"), ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(0.16, y + 0.035, step.get("text", ""), ha="center", va="center", fontsize=9)

    # Dotted overview panel on the right
    outer = patches.FancyBboxPatch(
        (0.35, 0.08),
        0.6,
        0.8,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2.0,
        edgecolor="#1F497D",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(outer)

    user_box = patches.Rectangle((0.37, 0.1), 0.28, 0.76, facecolor="#FFE089", edgecolor="none", alpha=0.9)
    model_box = patches.Rectangle((0.65, 0.1), 0.28, 0.76, facecolor="#1976D2", edgecolor="none", alpha=0.9)
    ax.add_patch(user_box)
    ax.add_patch(model_box)

    ax.text(0.51, 0.82, "User", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.79, 0.82, "HerbMind", ha="center", fontsize=13, fontweight="bold", color="white")

    ax.text(0.51, 0.77, "Provide initial seeds", ha="center", fontsize=9)
    ax.text(0.79, 0.77, "Model ranks candidate herbs", ha="center", fontsize=9, color="white")

    # Ranked list table inside HerbMind area
    rank_ax = fig.add_axes([0.63, 0.36, 0.29, 0.32])
    rank_ax.axis("off")
    table_data = [[idx + 1, herb, f"{score:.3f}"] for idx, (herb, score) in enumerate(ranked[:5])]
    rank_table = rank_ax.table(
        cellText=table_data,
        colLabels=["Rank", "Ingredient", "Affinity"],
        cellLoc="center",
        loc="center",
    )
    rank_table.auto_set_font_size(False)
    rank_table.set_fontsize(9)
    rank_table.scale(1.0, 1.1)

    # Affinity bar
    bar_ax = fig.add_axes([0.63, 0.25, 0.29, 0.08])
    bar_ax.barh([0], [affinity], color="#FF8A00")
    bar_ax.set_xlim(0, 1)
    bar_ax.set_yticks([])
    bar_ax.set_xticks(np.linspace(0, 1, 6))
    bar_ax.set_xlabel("Affinity Score")
    bar_ax.set_title("Selected Candidate")
    bar_ax.spines["left"].set_visible(False)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)

    ax.text(0.39, 0.32, "Ranked List", fontsize=11, fontweight="bold")
    ax.text(0.39, 0.28, "Affinity-driven suggestions", fontsize=9)

    plt.savefig(os.path.join(FIG_DIR, "fig1_overview_matched.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved fig1_overview_matched.png")


# ---------------------------------------------------------------------------
# Figure 2 – KDE of sPMIr
# ---------------------------------------------------------------------------

def _gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(grid)
    std = float(np.std(values))
    if std <= 1e-6:
        std = 1.0
    bandwidth = 1.06 * std * values.size ** (-1 / 5)
    bandwidth = max(bandwidth, 0.1)
    diff = grid[:, None] - values[None, :]
    kernel = np.exp(-0.5 * (diff / bandwidth) ** 2)
    density = kernel.sum(axis=1) / (values.size * bandwidth * np.sqrt(2 * np.pi))
    return density


def fig2_spmir_kde_matched(metrics: Dict) -> None:
    ensure_fig_dir()
    configure_style()

    distribution = metrics.get("distribution", {})
    size_keys = sorted(distribution.keys()) or ["size2", "size3", "size4"]
    samples = []
    for key in size_keys:
        vals = np.array(distribution.get(key, []), dtype=float)
        if vals.size == 0:
            rng = np.random.default_rng(abs(hash(key)) % 2**32)
            offset = 0.2 * size_keys.index(key)
            vals = rng.normal(0.1 + offset, 0.6, 800)
        samples.append((key, vals))

    all_values = np.concatenate([vals for _, vals in samples])
    grid = np.linspace(all_values.min() - 0.5, all_values.max() + 0.5, 400)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = ["#AFD3FF", "#6AA9FF", "#2A6FDB", "#0B3C8C", "#001B44"]

    for idx, (key, vals) in enumerate(samples):
        density = _gaussian_kde(vals, grid)
        label = key.replace("size", "|S|=")
        color = palette[idx % len(palette)]
        ax.plot(grid, density, color=color, linewidth=2.0, label=label)
        mean_val = float(np.mean(vals))
        ax.scatter([mean_val], [np.interp(mean_val, grid, density)], marker="v", color=color, s=60)

    ax.set_xlabel("sPMIr Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Affinity Scores by Set Size")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_spmir_kde_matched.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved fig2_spmir_kde_matched.png")


# ---------------------------------------------------------------------------
# Figure 3 – Cascaded SAB/PMX architecture
# ---------------------------------------------------------------------------

def fig3_cascaded_arch_matched() -> None:
    ensure_fig_dir()
    configure_style()

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.axis("off")

    def draw_block(x: float, y: float, text: str, width: float = 0.18, height: float = 0.12, color: str = "#E3F2FD"):
        block = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.4,
            edgecolor="#1565C0",
            facecolor=color,
        )
        ax.add_patch(block)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)

    # Ingredient stream (SAB)
    sab_x = 0.08
    sab_y_positions = [0.62, 0.42, 0.22]
    for idx, y in enumerate(sab_y_positions, start=1):
        draw_block(sab_x, y, f"SAB{idx}\nSet Attention", color="#E1F5FE")
        ax.text(sab_x - 0.04, y + 0.06, f"S{idx-1}", fontsize=9, fontweight="bold")

    # Candidate stream (PMX)
    pmx_x = 0.58
    pmx_y_positions = [0.62, 0.42, 0.22]
    for idx, y in enumerate(pmx_y_positions, start=1):
        draw_block(pmx_x, y, f"PMX{idx}\nCross Interaction", color="#E8EAF6")
        ax.text(pmx_x + 0.22, y + 0.06, f"a{idx-1}", fontsize=9, fontweight="bold")

    # Sum pooling and representation vectors
    draw_block(0.28, 0.02, "Sum Pooling\n$S_c$", width=0.18, height=0.12, color="#D0F0C0")
    draw_block(0.78, 0.02, "Pooling\n$a_c$", width=0.18, height=0.12, color="#FFE0B2")

    # Concatenate + MLP block
    concat = patches.FancyBboxPatch(
        (0.42, 0.02),
        0.28,
        0.12,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        edgecolor="#2E7D32",
        facecolor="#C8E6C9",
    )
    ax.add_patch(concat)
    ax.text(0.56, 0.08, "Concat($S_c$, $a_c$)\nMLP", ha="center", va="center", fontsize=10, fontweight="bold")

    # Output node
    output = patches.FancyBboxPatch(
        (0.72, 0.52),
        0.2,
        0.14,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6,
        edgecolor="#EF6C00",
        facecolor="#FFE0B2",
    )
    ax.add_patch(output)
    ax.text(0.82, 0.59, "Affinity Score\n$y$", ha="center", va="center", fontsize=10, fontweight="bold")

    # Connections
    def connect(p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", lw=1.2, color="#1565C0"))

    for sab_y, pmx_y in zip(sab_y_positions, pmx_y_positions):
        connect((sab_x + 0.18, sab_y + 0.06), (pmx_x, pmx_y + 0.06))

    connect((sab_x + 0.09, sab_y_positions[-1]), (0.37, 0.14))
    connect((pmx_x + 0.09, pmx_y_positions[-1]), (0.63, 0.14))

    connect((0.56, 0.14), (0.82, 0.52))
    connect((0.56, 0.08), (0.56, 0.14))

    ax.text(0.16, 0.88, "Ingredient Set Stream", fontsize=11, fontweight="bold")
    ax.text(0.66, 0.88, "Candidate Herb Stream", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_cascaded_arch_matched.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved fig3_cascaded_arch_matched.png")


# ---------------------------------------------------------------------------
# Figures 5 & 6 – Heat tables
# ---------------------------------------------------------------------------

def _extract_heat_section(metrics: Dict, key: str, fallback: Dict) -> Dict:
    section = metrics.get(key, {})
    resolved = {}
    for metric_name, spec in fallback.items():
        metric_section = section.get(metric_name, {}) if isinstance(section, dict) else {}
        values = np.array(metric_section.get("values", spec["values"]), dtype=float)
        rows = metric_section.get("rows", spec["rows"])
        cols = metric_section.get("cols", spec["cols"])
        resolved[metric_name] = {"values": values, "rows": rows, "cols": cols}
    return resolved


def _render_heat_panels(data: Dict[str, Dict], title: str, out_path: str) -> None:
    ensure_fig_dir()
    configure_style()

    metric_names = list(data.keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(10, 4.5))
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metric_names):
        spec = data[metric_name]
        values = np.array(spec["values"], dtype=float)
        rows = spec["rows"]
        cols = spec["cols"]

        im = ax.imshow(values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_title(metric_name)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center", fontsize=8, color="#0D47A1")

        ax.set_xlabel("Set Size |S|")
        ax.set_ylabel("Model")
        ax.grid(which="major", color="white", linewidth=1.0)
        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {os.path.basename(out_path)}")


def fig5_heat_baselines(metrics: Dict) -> None:
    fallback = {
        "RMSE ↓": {
            "values": np.array(
                [
                    [0.8923, 0.8675, 0.8541, 0.8420],
                    [0.9341, 0.9124, 0.9015, 0.8892],
                    [0.9510, 0.9407, 0.9294, 0.9210],
                ]
            ),
            "rows": ["HerbMind", "NoHerb2Vec", "Baseline"],
            "cols": ["|S|=2", "|S|=3", "|S|=4", "|S|=5"],
        },
        "PCORR ↑": {
            "values": np.array(
                [
                    [0.9123, 0.9244, 0.9311, 0.9420],
                    [0.8841, 0.8952, 0.9042, 0.9138],
                    [0.8610, 0.8705, 0.8790, 0.8884],
                ]
            ),
            "rows": ["HerbMind", "NoHerb2Vec", "Baseline"],
            "cols": ["|S|=2", "|S|=3", "|S|=4", "|S|=5"],
        },
    }
    data = _extract_heat_section(metrics, "baseline_heat", fallback)
    _render_heat_panels(data, "Baseline Performance", os.path.join(FIG_DIR, "fig5_baselines_heat_matched.png"))


def fig6_heat_ablation(metrics: Dict) -> None:
    fallback = {
        "RMSE ↓": {
            "values": np.array(
                [
                    [0.8923, 0.8675, 0.8541, 0.8420],
                    [0.9012, 0.8795, 0.8667, 0.8544],
                    [0.9188, 0.8933, 0.8821, 0.8710],
                ]
            ),
            "rows": ["Full", "-Set Transformer", "-Attention"],
            "cols": ["|S|=2", "|S|=3", "|S|=4", "|S|=5"],
        },
        "PCORR ↑": {
            "values": np.array(
                [
                    [0.9123, 0.9244, 0.9311, 0.9420],
                    [0.9015, 0.9130, 0.9201, 0.9287],
                    [0.8940, 0.9051, 0.9132, 0.9224],
                ]
            ),
            "rows": ["Full", "-Set Transformer", "-Attention"],
            "cols": ["|S|=2", "|S|=3", "|S|=4", "|S|=5"],
        },
    }
    data = _extract_heat_section(metrics, "ablation_heat", fallback)
    _render_heat_panels(data, "Ablation Study", os.path.join(FIG_DIR, "fig6_ablation_heat_matched.png"))


# ---------------------------------------------------------------------------
# Figures 7 & 8 – Stepwise cases (matched layout)
# ---------------------------------------------------------------------------

def _prepare_stepwise_data(metrics: Dict) -> Tuple[Dict, Dict, List[str]]:
    cases = load_cases()
    all_presc = load_prescriptions()
    if not all_presc:
        herb_freq, presc_sets, N = {}, [], 0
        universe = sorted({herb for cfg in cases.values() for herb in cfg.get("seeds", [])})
    else:
        herb_freq, presc_sets, N = build_counts(all_presc)
        universe = sorted({herb for herbs in all_presc.values() for herb in herbs})

    model_scorer = load_model_scorer(metrics)
    if model_scorer:
        scorer = model_scorer
    elif presc_sets:
        scorer = lambda S, x: spmir_score(S, x, herb_freq, presc_sets, N)
    else:
        scorer = lambda S, x: 0.0

    summaries = {}
    for case_key, cfg in cases.items():
        seeds = cfg.get("seeds", [])
        steps = int(cfg.get("steps", 8))
        summary = generate_case_summary(seeds, steps, scorer, universe)
        summaries[case_key] = {
            "title": cfg.get("title_ko", case_key),
            "seeds": seeds,
            "summary": summary,
        }
    return summaries, cases, universe


def _draw_stepwise_matched(case_key: str, entry: Dict) -> None:
    summary = entry["summary"]
    seeds = entry["seeds"]
    title = entry["title"]

    if not summary:
        print(f"[WARN] No recommendations for {case_key}; skipping matched figure.")
        return

    ensure_fig_dir()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].axis("off")
    rows_left = [["Seed", ", ".join(seeds)]]
    for step in summary:
        rows_left.append([f"Step {step['step']}", ", ".join(step["set"])])

    table_left = axes[0].table(
        cellText=rows_left,
        colLabels=["Step", "Ingredient Set"],
        cellLoc="center",
        loc="center",
    )
    table_left.auto_set_font_size(False)
    table_left.set_fontsize(9)
    table_left.scale(1.1, 1.2)
    axes[0].set_title(title, fontsize=12, fontweight="bold", pad=10)

    for col_idx in range(len(rows_left[0])):
        cell = table_left[(0, col_idx)]
        cell.set_facecolor("#FFDAD7")
        cell.set_edgecolor("#B65C5C")
        cell.get_text().set_color("#8B1A1A")
        cell.get_text().set_fontweight("bold")

    axes[1].axis("off")
    right_rows = []
    for step in summary:
        top3 = "\n".join(f"{h} ({s:.3f})" for h, s in step["top3"])
        right_rows.append([f"Step {step['step']}", top3])

    table_right = axes[1].table(
        cellText=right_rows,
        colLabels=["Step", "Top-3 Recommendations"],
        cellLoc="center",
        loc="center",
    )
    table_right.auto_set_font_size(False)
    table_right.set_fontsize(9)
    table_right.scale(1.1, 1.2)
    axes[1].set_title("Top-3 Recommendations by HerbMind", fontsize=12, fontweight="bold", pad=10)

    if right_rows:
        for col_idx in range(len(right_rows[0])):
            cell = table_right[(0, col_idx)]
            cell.set_facecolor("#FFF3E0")
            cell.set_edgecolor("#C27D3A")
            cell.get_text().set_color("#9C5700")
            cell.get_text().set_fontweight("semibold")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"fig_{case_key}_stepwise_matched.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {os.path.basename(out_path)}")


def fig7_8_stepwise(metrics: Dict) -> None:
    summaries, _cases, _ = _prepare_stepwise_data(metrics)
    for case_key, entry in summaries.items():
        _draw_stepwise_matched(case_key, entry)


# ---------------------------------------------------------------------------
# Figures 9 & 10 – Triangular attention (matched style)
# ---------------------------------------------------------------------------

def fig9_10_attention(metrics: Dict) -> None:
    attentions = metrics.get("case_attentions")
    if not isinstance(attentions, dict):
        print("[INFO] No case_attentions found; skipping matched attention figures.")
        return

    for case_key, entry in attentions.items():
        if isinstance(entry, dict):
            matrix = entry.get("matrix") or entry.get("values") or entry.get("data")
            xlabels = entry.get("xlabels") or entry.get("x")
            ylabels = entry.get("ylabels") or entry.get("y")
        else:
            matrix = entry
            xlabels = ylabels = None
        if matrix is None:
            continue
        render_triangular_attention(case_key + "_matched", matrix, xlabels, ylabels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    metrics = load_metrics()

    fig1_overview_matched(metrics)
    fig2_spmir_kde_matched(metrics)
    fig3_cascaded_arch_matched()
    fig5_heat_baselines(metrics)
    fig6_heat_ablation(metrics)
    fig7_8_stepwise(metrics)
    fig9_10_attention(metrics)

    print("[DONE] Layout-matched figures generated in ./figures/")


if __name__ == "__main__":
    main()
