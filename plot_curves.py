#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates continuous ROC and PR curves for TD and supervised model categories across all mortality horizons
(1, 3, 7, 14, and 28 day) on both internal and external datasets.
"""

import os
import glob
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.interpolate import interp1d

# Define helper to map filename to model group
def get_model_group(filename):
    if "TD" in filename:
        return "TD"
    for day in ["14d", "28d", "7d", "3d", "1d"]:
        if day in filename:
            if "balanced" in filename:
                return f"{day}+"
            else:
                return day
    return None

def main():
    results_dir = "./evaluation_results"
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist. Please run evaluate.py first.")
        return

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No prediction CSV files found in '{results_dir}'.")
        return

    # Group files by dataset and model group
    groups = {"internal": {}, "external": {}}
    for filepath in csv_files:
        basename = os.path.basename(filepath)
        dataset = "internal" if basename.startswith("internal_") else "external" if basename.startswith("external_") else None
        if not dataset:
            continue

        group = get_model_group(basename)
        if not group:
            continue

        if group not in groups[dataset]:
            groups[dataset][group] = []
        groups[dataset][group].append(filepath)

    datasets = ["internal", "external"]
    model_groups_ordered = ["TD", "1d", "1d+", "3d", "3d+", "7d", "7d+", "14d", "14d+", "28d", "28d+"]
    horizons_ordered = ["1d", "3d", "7d", "14d", "28d"]

    # Custom color palette
    colors = {
        'TD': '#E31A1C',
        '1d': '#A6CEE3',
        '1d+': '#1F78B4',
        '3d': '#B2DF8A',
        '3d+': '#33A02C',
        '7d': '#FDBF6F',
        '7d+': '#FF7F00',
        '14d': '#CAB2D6',
        '14d+': '#6A3D9A',
        '28d': '#FFFF99',
        '28d+': '#B15928'
    }

    # Standardized grid for interpolating curves
    grid_x = np.linspace(0.0, 1.0, 100)

    roc_metrics = {ds: {hz: {} for hz in horizons_ordered} for ds in datasets}
    prc_metrics = {ds: {hz: {} for hz in horizons_ordered} for ds in datasets}

    print("Loading prediction files and computing continuous curves...")
    for dataset in datasets:
        print(f"Processing {dataset} dataset...")
        for group in model_groups_ordered:
            if group not in groups[dataset]:
                continue

            files = groups[dataset][group]

            seeds_predictions = []
            seeds_labels = {hz: [] for hz in horizons_ordered}

            for file in files:
                columns_to_load = ["prediction"] + [f"label_{hz}" for hz in horizons_ordered]
                df = pl.read_csv(file, columns=columns_to_load)

                # Invert: prediction is survival probability -> y_prob is mortality probability
                y_prob = 1.0 - df["prediction"].to_numpy()
                seeds_predictions.append(y_prob)

                for hz in horizons_ordered:
                    # Invert: label is survival (1=alive, 0=died) -> y_true is mortality (1=died, 0=alive)
                    y_true = 1.0 - df[f"label_{hz}"].to_numpy()
                    seeds_labels[hz].append(y_true)

            for hz in horizons_ordered:
                seed_roc_y = []
                seed_prc_y = []
                base_rates = []

                for s in range(len(files)):
                    y_prob_s = seeds_predictions[s]
                    y_true_s = seeds_labels[hz][s]

                    base_rates.append(np.mean(y_true_s))

                    # ROC
                    fpr, tpr, _ = roc_curve(y_true_s, y_prob_s)
                    f_roc = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
                    seed_roc_y.append(f_roc(grid_x))

                    # PRC
                    precision, recall, _ = precision_recall_curve(y_true_s, y_prob_s)
                    
                    # Interp1d requires strictly increasing x. Recall from precision_recall_curve is decreasing.
                    rev_recall = recall[::-1]
                    rev_precision = precision[::-1]

                    unique_recall, indices_uniq = np.unique(rev_recall, return_index=True)
                    unique_precision = rev_precision[indices_uniq]

                    f_prc = interp1d(unique_recall, unique_precision, bounds_error=False, fill_value=(unique_precision[0], unique_precision[-1]))
                    seed_prc_y.append(f_prc(grid_x))

                # Average across seeds
                y_roc_mean = np.mean(seed_roc_y, axis=0)
                y_roc_std = np.std(seed_roc_y, axis=0)
                mean_auroc = auc(grid_x, y_roc_mean)

                y_prc_mean = np.mean(seed_prc_y, axis=0)
                y_prc_std = np.std(seed_prc_y, axis=0)
                mean_auprc = auc(grid_x, y_prc_mean)

                mean_base_rate = np.mean(base_rates)

                roc_metrics[dataset][hz][group] = {
                    "y_mean": y_roc_mean,
                    "y_std": y_roc_std,
                    "mean_auc": mean_auroc
                }
                prc_metrics[dataset][hz][group] = {
                    "y_mean": y_prc_mean,
                    "y_std": y_prc_std,
                    "mean_auc": mean_auprc,
                    "base_rate": mean_base_rate
                }

    sns.set_theme(style="whitegrid")
    os.makedirs("./figures", exist_ok=True)

    def plot_figure(metrics_dict, filename, xlabel, ylabel, plot_baseline=False):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 14))

        for i, ds in enumerate(datasets):
            for j, hz in enumerate(horizons_ordered):
                ax = axes[i, j]

                if plot_baseline:
                    # Draw horizontal baseline using the base rate from TD
                    if "TD" in metrics_dict[ds][hz]:
                        base_rate = metrics_dict[ds][hz]["TD"]["base_rate"]
                        ax.plot([0, 1], [base_rate, base_rate], linestyle="--", color="grey", alpha=0.8, linewidth=1.5, zorder=1)
                else:
                    # Diagonal line for ROC
                    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.8, linewidth=1.5, zorder=1)

                for group in model_groups_ordered:
                    if group not in metrics_dict[ds][hz]:
                        continue

                    m_data = metrics_dict[ds][hz][group]
                    y_plot = m_data["y_mean"]
                    y_std_plot = m_data["y_std"]
                    mean_auc = m_data["mean_auc"]

                    color = colors[group]
                    lw = 2.5 if group == "TD" else 1.2
                    zorder = 10 if group == "TD" else 2

                    label_text = f"{group} (AUC: {mean_auc:.3f})"
                    ax.plot(grid_x, y_plot, label=label_text, color=color, linewidth=lw, zorder=zorder)

                    ax.fill_between(grid_x, np.clip(y_plot - y_std_plot, 0, 1), np.clip(y_plot + y_std_plot, 0, 1),
                                    color=color, alpha=0.15, zorder=zorder - 1)

                if plot_baseline and "TD" in metrics_dict[ds][hz]:
                    base_rate = metrics_dict[ds][hz]["TD"]["base_rate"]
                    ax.plot([], [], linestyle="--", color="grey", alpha=0.8, linewidth=1.5, label=f"Prevalence ({base_rate:.2f})")

                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.0])
                ax.tick_params(axis='both', which='major', labelsize=18)

                if i == 1:
                    ax.set_xlabel(xlabel, fontsize=22, labelpad=15)

                if i == 0:
                    ax.set_title(f"{hz} Mortality", fontsize=32, fontweight='bold', color='#1A3B5C', pad=120)

                if j == 0:
                    row_label = "Internal" if i == 0 else "External"
                    ax.set_ylabel(rf"$\mathbf{{{row_label}}}$" + f"\n{ylabel}", fontsize=20, labelpad=15)
                else:
                    ax.set_ylabel("")

                ax.legend(loc="upper left", bbox_to_anchor=(0.01, 1.36), fontsize=12, ncol=2, framealpha=0.85,
                          edgecolor='none')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        fig.subplots_adjust(top=0.84, bottom=0.08, hspace=0.50)
        
        pdf_path = f"./figures/{filename}"
        plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Saved {pdf_path}")
        plt.close()

    print("Generating ROC curves...")
    plot_figure(roc_metrics, "roc_curves.pdf", "False Positive Rate", "True Positive Rate", plot_baseline=False)

    print("Generating PR curves...")
    plot_figure(prc_metrics, "pr_curves.pdf", "Recall", "Precision", plot_baseline=True)

    print("\n" + "=" * 50)
    print("All curves generated successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
