#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mypaperfiguretable_Pro.py
Publication-quality figure generator for HerbMind (600 dpi, Helvetica-like fonts).

Figures are saved under ./figures/ with the same filenames used in the manuscript:
fig1_model_architecture.png
fig2_spmir_distribution.png
fig3_accuracy_comparison.png
fig4_attention_heatmap.png
fig5_qualitative_examples.png

Optional: reads ./outputs/metrics.json with the structure:
{
    "distribution": {"size1":[...], "size2":[...], "size3":[...]},
    "performance": {"Baseline":{"Hits@3":..},"NoHerb2Vec":{..},"HerbMind":{..}},
    "attention": [[...],[...],...],  # 2D array
    "labels": {"x":["Ing1","Ing2",...], "y":["Step1","Step2",...]},
    "qualitative": {"base":[...], "top":[["Herb",0.9],...], "low":[["Herb",0.1],...]}
}
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "figures"
METRICS_PATH = "outputs/metrics.json"

# Style (Helvetica/Arial, 600 dpi, consistent palette)
mpl.rcParams.update(
    {
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,
    }
)

PALETTE = {
    "blue1": "#2a6fdb",
    "blue2": "#5b8ee6",
    "gray1": "#666666",
    "gray2": "#a0a0a0",
    "green1": "#00a07a",
    "red1": "#d64f45",
    "amber": "#ffb000",
}


def _ensure_dir() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def _load_metrics() -> Optional[Dict[str, Any]]:
    if os.path.isfile(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                print("[WARN] metrics.json is not valid JSON. Using placeholders.")
    return None


# Figure 1: Model architecture (clean schematic)
def fig_model_architecture() -> None:
    _ensure_dir()
    fig = plt.figure(figsize=(5.2, 3.3))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("HerbMind Model Architecture", pad=8)

    def box(x: float, y: float, w: float, h: float, label: str, fc: str = "#eef2ff", ec: str = "#1f1f1f") -> None:
        rect = mpl.patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.8,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    # Layout positions (0..1 axes fraction)
    box(0.05, 0.60, 0.25, 0.25, "Input Herbs (S)\n(Set of herb embeddings)")
    box(0.05, 0.20, 0.25, 0.25, "Candidate Herb (h)\n(Herb2Vec embedding)")
    box(0.38, 0.55, 0.25, 0.25, "Set Transformer\n(Self-Attention)", fc="#e6f4ea")
    box(0.38, 0.20, 0.25, 0.25, "Cross Interaction\n(via scoring MLP)", fc="#fff3e0")
    box(0.70, 0.38, 0.25, 0.25, "Affinity Scorer (MLP)\nŷ = score(S,h)", fc="#fce8e6")

    # Arrows
    def arr(xy1: tuple[float, float], xy2: tuple[float, float]) -> None:
        ax.annotate(
            "",
            xy=xy2,
            xytext=xy1,
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#333333"),
        )

    arr((0.30, 0.72), (0.38, 0.68))
    arr((0.30, 0.32), (0.38, 0.32))
    arr((0.63, 0.65), (0.70, 0.50))
    arr((0.63, 0.32), (0.70, 0.50))

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_model_architecture.png"), bbox_inches="tight")
    plt.close(fig)


# Figure 2: sPMIr distribution (read real data if available)
def fig_score_distribution(metrics: Optional[Dict[str, Any]]) -> None:
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    if metrics and "distribution" in metrics:
        distribution = metrics.get("distribution", {})
        s1 = np.asarray(distribution.get("size1", []), dtype=float)
        s2 = np.asarray(distribution.get("size2", []), dtype=float)
        s3 = np.asarray(distribution.get("size3", []), dtype=float)
    else:
        # placeholders (will be replaced if metrics.json exists)
        rng = np.random.default_rng(0)
        s1 = rng.normal(0.0, 1.0, 1200)
        s2 = rng.normal(0.2, 1.0, 1200)
        s3 = rng.normal(0.4, 1.0, 1200)

    ax.hist(s1, bins=40, alpha=0.55, label="|S|=1", color=PALETTE["gray2"])
    ax.hist(s2, bins=40, alpha=0.55, label="|S|=2", color=PALETTE["blue2"])
    ax.hist(s3, bins=40, alpha=0.55, label="|S|=3", color=PALETTE["green1"])
    ax.set_xlabel("sPMIr score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Affinity Scores by Set Size")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_spmir_distribution.png"), bbox_inches="tight")
    plt.close(fig)


# Figure 3: Performance comparison (Hits@3)
def fig_accuracy_comparison(metrics: Optional[Dict[str, Any]]) -> None:
    _ensure_dir()
    models = ["Baseline", "NoHerb2Vec", "HerbMind"]
    if metrics and "performance" in metrics:
        perf = metrics.get("performance", {})
        vals = [float(perf.get(model, {}).get("Hits@3", float("nan"))) for model in models]
    else:
        vals = [0.45, 0.50, 0.55]

    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    colors = [PALETTE["gray2"], "#9cd3ff", PALETTE["blue1"]]
    bars = ax.bar(models, vals, color=colors)
    for bar, value in zip(bars, vals):
        if not np.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylabel("Hits@3")
    ax.set_ylim(0, max(0.65, np.nanmax(vals) + 0.05) if not np.all(np.isnan(vals)) else 1)
    ax.set_title("Performance Comparison (Hits@3)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_accuracy_comparison.png"), bbox_inches="tight")
    plt.close(fig)


# Figure 4: Attention heatmap
def fig_attention_heatmap(metrics: Optional[Dict[str, Any]]) -> None:
    _ensure_dir()
    if metrics and "attention" in metrics:
        A = np.asarray(metrics.get("attention", []), dtype=float)
        labels = metrics.get("labels", {})
        xl = labels.get("x", [f"Ing{i + 1}" for i in range(A.shape[1] if A.ndim == 2 else 0)])
        yl = labels.get("y", [f"Step{i + 1}" for i in range(A.shape[0] if A.ndim == 2 else 0)])
    else:
        A = np.array(
            [
                [0.56, 0.43, 0.00, 0.00],
                [0.36, 0.32, 0.31, 0.00],
                [0.20, 0.25, 0.30, 0.25],
            ]
        )
        xl = ["Ing1", "Ing2", "Ing3", "Ing4"]
        yl = ["Step1", "Step2", "Step3"]

    if A.ndim != 2 or A.size == 0:
        print("[WARN] Attention matrix is empty or invalid. Skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    vmax = float(np.nanmax(A)) if not np.isnan(A).all() else 1.0
    vmax = max(0.30, vmax)
    im = ax.imshow(A, cmap="Reds", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(xl)))
    ax.set_xticklabels(xl, rotation=45, ha="right")
    ax.set_yticks(range(len(yl)))
    ax.set_yticklabels(yl)
    ax.set_title("Attention Weights by Expansion Step")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Weight", rotation=270, va="bottom")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_attention_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


# Figure 5: Qualitative examples (table-like)
def fig_qualitative_examples(metrics: Optional[Dict[str, Any]]) -> None:
    _ensure_dir()
    if metrics and "qualitative" in metrics:
        qual = metrics.get("qualitative", {})
        base = qual.get("base", [])
        top = qual.get("top", [])
        low = qual.get("low", [])
    else:
        base = ["HerbA", "HerbB"]
        top = [["HerbX", 0.95], ["HerbY", 0.88], ["HerbZ", 0.85]]
        low = [["HerbM", 0.20], ["HerbN", 0.15], ["HerbO", 0.10]]

    def _format_row(entries: Any) -> str:
        formatted: list[str] = []
        for item in entries:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                herb, score = item
                try:
                    formatted.append(f"{herb} ({float(score):.3f})")
                except (TypeError, ValueError):
                    formatted.append(f"{item[0]} ({item[1]})")
            else:
                formatted.append(str(item))
        return ", ".join(formatted) if formatted else "-"

    fig, ax = plt.subplots(figsize=(6.2, 1.8))
    ax.axis("off")
    cols = ["Base Herbs", "Top Additions (score)", "Lowest Additions (score)"]
    row = [", ".join(base) if base else "-", _format_row(top), _format_row(low)]
    table = ax.table(cellText=[row], colLabels=cols, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_qualitative_examples.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = _load_metrics()
    fig_model_architecture()
    fig_score_distribution(metrics)
    fig_accuracy_comparison(metrics)
    fig_attention_heatmap(metrics)
    fig_qualitative_examples(metrics)
    print("[OK] Pro figures saved under ./figures (600 dpi).")


if __name__ == "__main__":
    main()
