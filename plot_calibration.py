#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 4, 2026
@author: Antigravity

Generates continuous LOESS calibration curves, ICI (Integrated Calibration Index),
and ECE metrics for TD and supervised model categories across all mortality horizons
(1, 3, 7, 14, and 28 day) on both internal and external datasets.
"""

import os
import glob
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
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


# Expected Calibration Error (ECE) computation function
def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return ece


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
        dataset = "internal" if basename.startswith("internal_") else "external" if basename.startswith(
            "external_") else None
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

    # Create a standardized grid for seamless seed averaging
    grid_x = np.linspace(0.0, 1.0, 100)

    # Data structure to hold computed calibration metrics
    metrics = {ds: {hz: {} for hz in horizons_ordered} for ds in datasets}

    print("Loading prediction files and computing continuous calibration statistics...")
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
                seed_grids_y = []
                seed_icis = []

                for s in range(len(files)):
                    y_prob_s = seeds_predictions[s]
                    y_true_s = seeds_labels[hz][s]

                    # Subsample to speed up LOESS on very large datasets
                    if len(y_prob_s) > 20000:
                        np.random.seed(42 + s)
                        indices = np.random.choice(len(y_prob_s), size=20000, replace=False)
                        y_prob_sub = y_prob_s[indices]
                        y_true_sub = y_true_s[indices]
                    else:
                        y_prob_sub = y_prob_s
                        y_true_sub = y_true_s

                    # 1. Compute Continuous LOESS
                    lowess_res = sm.nonparametric.lowess(y_true_sub, y_prob_sub, frac=0.75, it=0, delta=0.001)
                    loess_x = lowess_res[:, 0]
                    loess_y = lowess_res[:, 1]

                    # Deduplicate x-values to prevent division by zero in interpolation
                    unique_x, indices_uniq = np.unique(loess_x, return_index=True)
                    unique_y = loess_y[indices_uniq]

                    # Interpolate onto our standardized grid
                    f_interp = interp1d(unique_x, unique_y, bounds_error=False, fill_value="extrapolate")
                    seed_grids_y.append(f_interp(grid_x))

                    # 2. Metrics (Inline ICI calculation utilizing the interpolation)
                    smoothed_full = f_interp(y_prob_s)
                    ici_s = np.mean(np.abs(smoothed_full - y_prob_s))
                    seed_icis.append(ici_s)

                # Average across seeds
                y_mean = np.mean(seed_grids_y, axis=0)
                y_std = np.std(seed_grids_y, axis=0)
                mean_ici = np.mean(seed_icis)

                metrics[dataset][hz][group] = {
                    "x_mean": grid_x,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "mean_ici": mean_ici,
                    "all_probs": np.concatenate(seeds_predictions)  # Save for marginal density plot
                }

    print("Generating calibration plots...")
    sns.set_theme(style="whitegrid")

    # Create 2x5 figure
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 14))

    for i, ds in enumerate(datasets):
        for j, hz in enumerate(horizons_ordered):
            ax = axes[i, j]

            # 1. Plot the underlying sample distribution (histogram) using TD as baseline
            if "TD" in metrics[ds][hz]:
                sample_probs = metrics[ds][hz]["TD"]["all_probs"]
                ax_hist = ax.twinx()
                ax_hist.hist(sample_probs, bins=50, alpha=0.15, color="grey", range=(0, 1), density=True)
                ax_hist.set_yticks([])
                ax_hist.grid(False)
                # Hide histogram spines so it doesn't add border clutter
                for spine in ax_hist.spines.values():
                    spine.set_visible(False)

            # 2. Plot the diagonal perfect calibration line
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.8, linewidth=1.5, zorder=1)

            # 3. Plot each model category
            for group in model_groups_ordered:
                if group not in metrics[ds][hz]:
                    continue

                m_data = metrics[ds][hz][group]
                x_plot = m_data["x_mean"]
                y_plot = m_data["y_mean"]
                y_std_plot = m_data["y_std"]
                mean_ici = m_data["mean_ici"]

                color = colors[group]
                lw = 2.5 if group == "TD" else 1.2
                zorder = 10 if group == "TD" else 2

                # Plot mean line with ICI in the legend
                label_text = f"{group} ({mean_ici:.3f})"
                ax.plot(x_plot, y_plot, label=label_text, color=color, linewidth=lw, zorder=zorder)

                # Plot shaded standard deviation area
                ax.fill_between(x_plot, np.clip(y_plot - y_std_plot, 0, 1), np.clip(y_plot + y_std_plot, 0, 1),
                                color=color,
                                alpha=0.15, zorder=zorder - 1)

            # Formatting and labeling
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.tick_params(axis='both', which='major', labelsize=18)

            # Only put X label on the bottom row
            if i == 1:
                ax.set_xlabel("Predicted Probability of Death", fontsize=22, labelpad=15)

            # Row and column titles
            if i == 0:
                ax.set_title(f"{hz} Mortality", fontsize=32, fontweight='bold', color='#1A3B5C', pad=120)

            # Y labels and row indicator on the first column
            if j == 0:
                row_label = "Internal" if i == 0 else "External"
                ax.set_ylabel(rf"$\mathbf{{{row_label}}}$" + "\nObserved Fraction of Deaths", fontsize=20, labelpad=15)
            else:
                ax.set_ylabel("")

            # Place legend
            ax.legend(loc="upper left", bbox_to_anchor=(0.01, 1.36), fontsize=12, ncol=2, framealpha=0.85,
                      edgecolor='none')

            # Hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(top=0.84, bottom=0.08, hspace=0.50)
    os.makedirs("./figures", exist_ok=True)

    pdf_path = "./figures/calibration_curves.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")

    print("\n" + "=" * 50)
    print("Calibration curves generated successfully!")
    print(f"PDF saved to: {pdf_path}")
    print("=" * 50)
    plt.close()


if __name__ == "__main__":
    main()