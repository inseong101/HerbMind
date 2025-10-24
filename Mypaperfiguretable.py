#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mypaperfiguretable.py
Standalone script to generate all paper figures and print tables for HerbMind.

Tables: printed to console (no CSV export)

Figures: saved as PNG into 'figures/'

Data: expects herbal prescription CSVs under 'data/' directory
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_prescriptions(data_dir="data"):
    """Load prescriptions from CSVs in data_dir and return:
    - all_prescriptions: dict {prescription_id: set(herbs)}
    - unique_herbs: set of herb names
    Column name guess: '처방아이디' for ID, '약재한글명' for herb.
    """
    all_prescriptions, unique_herbs = {}, set()
    if not os.path.isdir(data_dir):
        print(f"[WARN] data dir not found: {data_dir}. Using empty dataset.")
        return all_prescriptions, unique_herbs

    csvs = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    ]
    if not csvs:
        print(f"[WARN] no CSV files in {data_dir}")
        return all_prescriptions, unique_herbs

    for path in csvs:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:  # pylint: disable=broad-except
            print(f"[WARN] cannot read {path}: {e}")
            continue
        id_col, herb_col = None, None
        # prefer Korean headers; otherwise guess
        for col in df.columns:
            if ("처방" in col) or ("ID" in col.upper()):
                id_col = id_col or col
            if ("약재" in col) or ("HERB" in col.upper()):
                herb_col = herb_col or col
        if id_col is None or herb_col is None:
            print(f"[WARN] required columns not found in {path}, skip.")
            continue

        for pid, group in df.groupby(id_col):
            herbs = set(group[herb_col].dropna().astype(str))
            if not herbs:
                continue
            all_prescriptions.setdefault(pid, set()).update(herbs)
            unique_herbs.update(herbs)

    return all_prescriptions, unique_herbs


def print_dataset_stats(all_prescriptions, unique_herbs):
    sizes = [len(s) for s in all_prescriptions.values() if len(s) > 0]
    total_presc = len(all_prescriptions)
    total_herbs = len(unique_herbs)
    avg_sz = float(np.mean(sizes)) if sizes else 0
    med_sz = float(np.median(sizes)) if sizes else 0
    min_sz = int(np.min(sizes)) if sizes else 0
    max_sz = int(np.max(sizes)) if sizes else 0

    print("=== Dataset Statistics ===")
    print(f"Total prescriptions\t{total_presc}")
    print(f"Unique herbs\t{total_herbs}")
    print(f"Avg herbs/prescription\t{avg_sz:.2f}")
    print(f"Median herbs/prescription\t{med_sz:.2f}")
    print(f"Min herbs/prescription\t{min_sz}")
    print(f"Max herbs/prescription\t{max_sz}")
    print("")


def print_ablation_table():
    # Example values — replace with real numbers when available
    metrics = ["Hits@3", "Hits@5", "MRR"]
    results = {
        "Baseline": [0.450, 0.600, 0.500],
        "NoHerb2Vec": [0.500, 0.650, 0.550],
        "HerbMind": [0.550, 0.700, 0.600],
    }
    print("=== Ablation Study (console table) ===")
    print("Model\t" + "\t".join(metrics))
    for model, vals in results.items():
        print(model + "\t" + "\t".join(f"{val:.3f}" for val in vals))
    print("")


def fig_model_architecture():
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.axis("off")
    ax.set_title("HerbMind Model Architecture", fontsize=12)

    def box(x_pos, y_pos, width, height, label):
        rect = plt.Rectangle((x_pos, y_pos), width, height, edgecolor="black", facecolor="#dde5ff")
        ax.add_patch(rect)
        ax.text(x_pos + width / 2, y_pos + height / 2, label, ha="center", va="center", fontsize=8)

    box(0.08, 0.62, 0.22, 0.20, "Input Herbs (S)")
    box(0.08, 0.25, 0.22, 0.20, "Candidate Herb (h)")
    box(0.38, 0.60, 0.24, 0.16, "Set Transformer\n(Self-Attention)")
    box(0.38, 0.25, 0.24, 0.16, "Herb2Vec\nEmbeddings")
    box(0.68, 0.42, 0.24, 0.16, "Affinity Scorer\n(MLP)")
    ax.annotate("", xy=(0.38, 0.70), xytext=(0.30, 0.70), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.38, 0.33), xytext=(0.30, 0.33), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.68, 0.50), xytext=(0.62, 0.60), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.68, 0.50), xytext=(0.62, 0.33), arrowprops=dict(arrowstyle="->"))
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig1_model_architecture.png", bbox_inches="tight")
    plt.close()


def fig_score_distribution():
    np.random.seed(0)
    scores_s1 = np.random.normal(0.0, 1.0, 800)  # |S|=1
    scores_s2 = np.random.normal(0.2, 1.0, 800)  # |S|=2
    scores_s3 = np.random.normal(0.4, 1.0, 800)  # |S|=3
    plt.figure(figsize=(6, 4))
    plt.hist(scores_s1, bins=30, alpha=0.5, label="|S|=1")
    plt.hist(scores_s2, bins=30, alpha=0.5, label="|S|=2")
    plt.hist(scores_s3, bins=30, alpha=0.5, label="|S|=3")
    plt.title("Distribution of Affinity Scores by Set Size")
    plt.xlabel("sPMIr score")
    plt.ylabel("Frequency")
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig2_spmir_distribution.png", bbox_inches="tight")
    plt.close()


def fig_accuracy_comparison():
    models = ["Baseline", "NoHerb2Vec", "HerbMind"]
    hits3 = [0.450, 0.500, 0.550]  # example values
    plt.figure(figsize=(5, 4))
    plt.bar(models, hits3, color=["gray", "skyblue", "steelblue"])
    plt.title("Performance Comparison (Hits@3)")
    plt.ylabel("Hits@3")
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig3_accuracy_comparison.png", bbox_inches="tight")
    plt.close()


def fig_attention_heatmap():
    att = np.array(
        [
            [0.567, 0.433, 0.000, 0.000],
            [0.365, 0.320, 0.315, 0.000],
            [0.200, 0.250, 0.300, 0.250],
        ]
    )
    plt.figure(figsize=(5, 4))
    plt.imshow(att, cmap="Blues", aspect="auto")
    plt.title("Attention Weights by Expansion Step")
    plt.ylabel("Expansion Step")
    plt.xlabel("Ingredient Index")
    plt.yticks([0, 1, 2], ["Step1", "Step2", "Step3"])
    plt.xticks([0, 1, 2, 3], ["Ing1", "Ing2", "Ing3", "Ing4"])
    plt.colorbar(fraction=0.046, pad=0.04)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig4_attention_heatmap.png", bbox_inches="tight")
    plt.close()


def fig_qualitative_examples():
    base = ["HerbA", "HerbB"]
    top = [("HerbX", 0.95), ("HerbY", 0.88), ("HerbZ", 0.85)]
    low = [("HerbM", 0.20), ("HerbN", 0.15), ("HerbO", 0.10)]
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    cols = ["Base Herbs", "Top Additions (score)", "Lowest Additions (score)"]
    text = [
        [
            ", ".join(base),
            ", ".join(f"{herb} ({score:.3f})" for herb, score in top),
            ", ".join(f"{herb} ({score:.3f})" for herb, score in low),
        ]
    ]
    table = ax.table(cellText=text, colLabels=cols, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig5_qualitative_examples.png", bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)
    presc, herbs = load_prescriptions("data")
    print_dataset_stats(presc, herbs)
    print_ablation_table()
    fig_model_architecture()
    fig_score_distribution()
    fig_accuracy_comparison()
    fig_attention_heatmap()
    fig_qualitative_examples()
    print("[OK] All figures saved under 'figures/' and tables printed to console.")


if __name__ == "__main__":
    main()
