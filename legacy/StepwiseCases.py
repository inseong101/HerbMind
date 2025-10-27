#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StepwiseCases.py

Generate step-wise recommendation figures that mirror RecipeMind Fig.7/8.
Two default cases (caseA, caseB) are described in ``cases.yaml``.

- Left table: cumulative ingredient set after each recommendation step.
- Right table: per-step Top-3 recommended herbs with scores.
- Optional triangular attention heatmaps (if provided via ``metrics.json``).

The script prefers a model-based scorer (``outputs/model_scores.pkl``) when
available and falls back to a statistical sPMIr scorer computed from the
co-occurrence counts in ``data/*.csv``.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib import patches

DATA_DIR = "data"
OUT_DIR = "figures"
METRICS_JSON = "outputs/metrics.json"
MODEL_SCORES = "outputs/model_scores.pkl"


def load_metrics(path: str = METRICS_JSON) -> Dict:
    """Load optional metrics JSON if present."""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse {path}; ignoring metrics.")
    return {}


def load_cases(path: str = "cases.yaml") -> Dict[str, Dict]:
    """Load case configuration from YAML (falls back to defaults)."""
    default_cases = {
        "caseA": {
            "title_ko": "사례 A: 숙지황·산수유 → 단계적 추천",
            "seeds": ["숙지황", "山茱萸"],
            "steps": 8,
        },
        "caseB": {
            "title_ko": "사례 B: 마황·계지 → 단계적 추천",
            "seeds": ["麻黃", "계지"],
            "steps": 8,
        },
    }
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key, cfg in default_cases.items():
                data.setdefault(key, cfg)
            return data
        except yaml.YAMLError:
            print(f"[WARN] Failed to parse {path}; using defaults.")
    return default_cases


def _guess_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    id_col, herb_col = None, None
    for col in df.columns:
        col_upper = str(col).upper()
        if id_col is None and ("처방" in str(col) or "ID" in col_upper):
            id_col = col
        if herb_col is None and ("약재" in str(col) or "HERB" in col_upper):
            herb_col = col
    return id_col, herb_col


def load_prescriptions(data_dir: str = DATA_DIR) -> Dict[str, set]:
    """Load prescription data into a mapping of prescription -> herb set."""
    all_presc: Dict[str, set] = {}
    if not os.path.isdir(data_dir):
        return all_presc

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to read {path}: {exc}")
            continue
        id_col, herb_col = _guess_cols(df)
        if not id_col or not herb_col:
            continue
        for pid, group in df.groupby(id_col):
            herbs = [str(x).strip() for x in group[herb_col].dropna() if str(x).strip()]
            if herbs:
                all_presc.setdefault(str(pid), set()).update(herbs)
    return all_presc


def build_counts(all_presc: Dict[str, set]) -> Tuple[Dict[str, int], List[set], int]:
    """Return herb frequency counts, prescription sets list, and corpus size."""
    presc_sets = list(all_presc.values())
    herb_freq: Dict[str, int] = {}
    for herbs in presc_sets:
        for herb in herbs:
            herb_freq[herb] = herb_freq.get(herb, 0) + 1
    return herb_freq, presc_sets, len(presc_sets)


def spmir_score(
    S: Sequence[str],
    x: str,
    herb_freq: Dict[str, int],
    presc_sets: Sequence[set],
    N: int,
    alpha: float = 20.0,
) -> float:
    """Compute the smoothed PMI-r score for candidate ``x`` given set ``S``."""
    S_set = set(S)
    if x in S_set:
        return -1e9
    r_union = sum(1 for herbs in presc_sets if S_set.union({x}).issubset(herbs))
    r_S = sum(1 for herbs in presc_sets if S_set.issubset(herbs))
    r_x = herb_freq.get(x, 0)
    num = N * r_union + 1e-9
    den = (r_S * r_x) + alpha + 1e-9
    score = float(np.log(num / den))
    return score


def load_model_scorer(metrics: Optional[Dict] = None) -> Optional[Callable[[Sequence[str], str], float]]:
    """Load a model-based scorer from pickle or metrics (if provided)."""
    if metrics:
        case_scores = metrics.get("case_scores")
        if isinstance(case_scores, dict):
            def score_from_dict(S: Sequence[str], x: str) -> float:
                key = "+".join(sorted(list(S) + [x]))
                return float(case_scores.get(key, case_scores.get(x, 0.0)))

            return score_from_dict

    if os.path.isfile(MODEL_SCORES):
        with open(MODEL_SCORES, "rb") as f:
            obj = pickle.load(f)
        if callable(obj):
            return lambda S, x: float(obj(S, x))
        if hasattr(obj, "score") and callable(obj.score):
            return lambda S, x: float(obj.score(S, x))
        if isinstance(obj, dict):
            def score_from_dict(S: Sequence[str], x: str) -> float:
                key = tuple(sorted(list(S) + [x]))
                if key in obj:
                    return float(obj[key])
                return float(obj.get(x, 0.0))

            return score_from_dict
    return None


def rank_candidates(
    S: Sequence[str],
    universe: Iterable[str],
    scorer: Callable[[Sequence[str], str], float],
) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    S_set = set(S)
    for herb in universe:
        if herb in S_set:
            continue
        try:
            score = float(scorer(S, herb))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] scorer failed for ({S}, {herb}): {exc}")
            continue
        scores.append((herb, score))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def generate_case_summary(
    seeds: Sequence[str],
    steps: int,
    scorer: Callable[[Sequence[str], str], float],
    universe: Iterable[str],
) -> List[Dict]:
    """Return per-step cumulative sets and top-3 recommendations."""
    current: List[str] = list(seeds)
    used = set(current)
    summary: List[Dict] = []
    for step_idx in range(steps):
        rankings = rank_candidates(current, universe, scorer)
        if not rankings:
            break
        top3 = rankings[:3]
        # Add the best candidate to the current set
        best = top3[0][0]
        if best in used:
            break
        current.append(best)
        used.add(best)
        summary.append(
            {
                "step": step_idx + 1,
                "set": list(current),
                "top3": top3,
            }
        )
    return summary


def _format_top3(top3: List[Tuple[str, float]]) -> str:
    return "\n".join(f"{h} ({s:.3f})" for h, s in top3)


def plot_stepwise_case(
    case_key: str,
    title: str,
    seeds: Sequence[str],
    summary: List[Dict],
) -> None:
    if not summary:
        print(f"[WARN] No recommendations generated for {case_key}; skipping figure.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Left table: cumulative ingredient set
    left_rows = [["Seed", ", ".join(seeds)]]
    for entry in summary:
        left_rows.append([f"Step {entry['step']}", ", ".join(entry["set"])])

    axes[0].axis("off")
    table_left = axes[0].table(
        cellText=left_rows,
        colLabels=["Step", "Ingredient Set"],
        cellLoc="center",
        loc="center",
    )
    table_left.auto_set_font_size(False)
    table_left.set_fontsize(9)
    table_left.scale(1.1, 1.2)
    axes[0].set_title(title, pad=10, fontsize=12, fontweight="bold")

    # Highlight the seed row in soft red with bold text
    seed_row_idx = 0
    for col_idx in range(len(left_rows[0])):
        cell = table_left[(seed_row_idx, col_idx)]
        cell.set_facecolor("#FFDAD7")
        cell.set_edgecolor("#B65C5C")
        cell.get_text().set_color("#8B1A1A")
        cell.get_text().set_fontweight("bold")

    # Right table: per-step Top-3 recommendations
    right_rows = []
    for entry in summary:
        right_rows.append([f"Step {entry['step']}", _format_top3(entry["top3"])])

    axes[1].axis("off")
    table_right = axes[1].table(
        cellText=right_rows,
        colLabels=["Step", "Top-3 Recommendations"],
        cellLoc="center",
        loc="center",
    )
    table_right.auto_set_font_size(False)
    table_right.set_fontsize(9)
    table_right.scale(1.1, 1.2)
    axes[1].set_title("Top-3 Recommendations by HerbMind", pad=10, fontsize=12, fontweight="bold")

    # Slight highlight for the first recommendation row
    if right_rows:
        for col_idx in range(len(right_rows[0])):
            cell = table_right[(0, col_idx)]
            cell.set_facecolor("#FFF3E0")
            cell.set_edgecolor("#C27D3A")
            cell.get_text().set_color("#9C5700")
            cell.get_text().set_fontweight("semibold")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"fig_{case_key}_stepwise.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def render_triangular_attention(
    case_key: str,
    matrix: Sequence[Sequence[float]],
    xlabels: Optional[Sequence[str]] = None,
    ylabels: Optional[Sequence[str]] = None,
) -> None:
    arr = np.array(matrix, dtype=float)
    if arr.ndim != 2:
        print(f"[WARN] Attention for {case_key} is not 2D; skipping.")
        return
    steps, cols = arr.shape
    if xlabels is None:
        xlabels = [f"Ing {i+1}" for i in range(cols)]
    if ylabels is None:
        ylabels = [f"Step {i+1}" for i in range(steps)]

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 5.0))

    mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
    display = arr.copy()
    display[mask] = np.nan

    vmax = max(0.6, float(np.nanmax(display))) if np.isfinite(np.nanmax(display)) else 0.6
    cmap = plt.get_cmap("Reds")
    im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=vmax)

    for i in range(steps):
        for j in range(cols):
            if mask[i, j]:
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor="#FFF2CC", edgecolor="#C4B27C", linewidth=0.8)
                ax.add_patch(rect)
                ax.text(j, i, "?", ha="center", va="center", fontsize=9, color="#B28300", fontweight="bold")
            else:
                ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#4A0D0D")

    ax.set_xticks(range(cols))
    ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(steps))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_title(f"Attention Heatmap · {case_key}", fontsize=12, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Weight", rotation=270, labelpad=12)

    out_path = os.path.join(OUT_DIR, f"fig_{case_key}_attn.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def _extract_attention_for_case(metrics: Dict, case_key: str) -> Optional[Dict]:
    attentions = metrics.get("case_attentions")
    if isinstance(attentions, dict):
        entry = attentions.get(case_key)
        if isinstance(entry, dict):
            matrix = entry.get("matrix") or entry.get("values") or entry.get("data")
            if matrix is None:
                return None
            return {
                "matrix": matrix,
                "xlabels": entry.get("xlabels") or entry.get("x") or entry.get("columns"),
                "ylabels": entry.get("ylabels") or entry.get("y") or entry.get("rows"),
            }
        if isinstance(entry, list):
            return {"matrix": entry}
    return None


def main() -> None:
    metrics = load_metrics()
    cases = load_cases()

    all_presc = load_prescriptions()
    if not all_presc:
        print("[WARN] No prescription data loaded from ./data/*.csv. Using seeds only.")
        herb_freq, presc_sets, N = {}, [], 0
        universe = sorted({herb for cfg in cases.values() for herb in cfg.get("seeds", [])})
    else:
        herb_freq, presc_sets, N = build_counts(all_presc)
        universe = sorted({herb for herbs in all_presc.values() for herb in herbs})

    model_scorer = load_model_scorer(metrics)
    if model_scorer:
        scorer = model_scorer
        print("[INFO] Using model-based scorer for recommendations.")
    elif presc_sets:
        scorer = lambda S, x: spmir_score(S, x, herb_freq, presc_sets, N)
        print("[INFO] Using sPMIr fallback scorer.")
    else:
        scorer = lambda S, x: 0.0
        print("[INFO] Using neutral scores (no data available).")

    for case_key, cfg in cases.items():
        seeds = cfg.get("seeds", [])
        steps = int(cfg.get("steps", 8))
        title = cfg.get("title_ko", case_key)

        summary = generate_case_summary(seeds, steps, scorer, universe)
        plot_stepwise_case(case_key, title, seeds, summary)

        attn_entry = _extract_attention_for_case(metrics, case_key)
        if attn_entry and attn_entry.get("matrix") is not None:
            render_triangular_attention(
                case_key,
                attn_entry["matrix"],
                attn_entry.get("xlabels"),
                attn_entry.get("ylabels"),
            )


if __name__ == "__main__":
    main()
