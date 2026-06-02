#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 28, 2026
@author: Antigravity

Statistical testing comparing the ensembled TD model group against each
supervised learning model group (standard and balanced "+") for each mortality target.
Averages fractional ranks across five seeds, runs DeLong's test, and applies
the Benjamini-Hochberg procedure for multiple testing correction.
"""

import os
import glob
import numpy as np
import polars as pl
import scipy.stats
from scipy import stats
import datetime
import numba
from tqdm import tqdm
from joblib import Parallel, delayed

@numba.njit(cache=True)
def compute_midrank(x):
    """Computes midranks to handle tied prediction values.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=numba.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=numba.float64)
    T2[J] = T + 1
    return T2

@numba.njit(cache=True)
def compute_auc_score(gt, preds):
    n = len(gt)
    n_pos = np.sum(gt)
    n_neg = n - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Re-use our fast Numba midrank function!
    ranks = compute_midrank(preds)
    
    # Sum the ranks of the positive class
    pos_ranks_sum = 0.0
    for i in range(n):
        if gt[i] == 1:
            pos_ranks_sum += ranks[i]
            
    # Calculate AUROC directly via the U-statistic formula
    u_stat = pos_ranks_sum - (n_pos * (n_pos + 1.0)) / 2.0
    return u_stat / (n_pos * n_neg)


def _compute_single_group_horizon(dataset, group, h, gt, ens_rank, seeds_preds):
    """Worker function for parallel AUROC calculation."""
    # 1. Calculate ensemble AUC
    ensemble_auc = compute_auc_score(gt, ens_rank)
    
    # 2. Calculate individual seed AUCs
    seed_aucs = [compute_auc_score(gt, sp) for sp in seeds_preds]
    
    # Return everything needed to reconstruct the dictionary later
    return dataset, group, h, ensemble_auc, np.mean(seed_aucs), np.std(seed_aucs), seed_aucs



def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    Vectorized DeLong's method for computing the covariance of unadjusted AUC.
    Reference: Sun & Xu, 2014 (IEEE Signal Processing Letters)
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1]) or np.array_equal(np.unique(ground_truth), [0]) or np.array_equal(np.unique(ground_truth), [1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes DeLong's test for comparing two correlated AUCs.
    Returns: (auc1, auc2, p_value)
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    if label_1_count == 0 or label_1_count == len(ground_truth):
        # Degenerate case: all 0 or all 1 labels
        return 0.5, 0.5, 1.0
    predictions_sorted_transposed = np.vstack([predictions_one[order], predictions_two[order]])
    
    aucs, delongcov = fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    
    l = np.array([[1, -1]])
    variance = np.dot(np.dot(l, delongcov), l.T)[0, 0]
    if variance <= 0:
        return aucs[0], aucs[1], 1.0
        
    z = np.abs(aucs[0] - aucs[1]) / np.sqrt(variance)
    p_value = 2 * scipy.stats.norm.sf(z)
    
    return aucs[0], aucs[1], p_value


def compute_fractional_ranks(predictions):
    """
    Converts probabilistic predictions to fractional ranks (0-1) across the dataset.
    Handles tied predictions by assigning their average rank.
    """
    n = len(predictions)
    if n <= 1:
        return np.zeros_like(predictions)
    ranks = scipy.stats.rankdata(predictions, method='average')
    return (ranks - 1.0) / (n - 1.0)


def get_model_group(filename):
    """
    Maps filename to its model group: TD, 1d, 1d+, 3d, 3d+, etc.
    "+" indicates balanced cross-entropy.
    Works for both new model naming conventions and pre-trained/past conventions.
    """
    if "TD" in filename:
        return "TD"
    
    # Check in descending order of length to avoid "1d" matching "14d"
    for day in ["14d", "28d", "7d", "3d", "1d"]:
        if day in filename:
            if "balanced" in filename:
                return f"{day}+"
            else:
                return day
    return None


def benjamini_hochberg_correction(p_values):
    """
    Performs Benjamini-Hochberg false discovery rate correction in pure Python.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    if n == 0:
        return np.array([])
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]
    
    adjusted_p = np.zeros(n)
    prev_adj = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = sorted_p[i] * n / rank
        adj = min(adj, prev_adj)
        adjusted_p[i] = adj
        prev_adj = adj
        
    original_idx = np.argsort(sort_idx)
    return adjusted_p[original_idx]


def run_statistical_analysis():
    results_dir = "./evaluation_results"
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist. Please run evaluate.py first.")
        return

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No prediction CSV files found in '{results_dir}'. Please run evaluate.py first.")
        return

    # Group files by dataset and model group
    # Structure: groups[dataset][model_group] = [list of filepaths]
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

    print(f"Found datasets and groups in '{results_dir}':")
    for dataset in ["internal", "external"]:
        print(f"  {dataset.upper()} test set:")
        for group, files in sorted(groups[dataset].items()):
            print(f"    - Group '{group}': {len(files)} seed models (files)")

    # Load all predictions and labels
    loaded_data = {"internal": {}, "external": {}}
    labels = {"internal": None, "external": None}
    
    for dataset in ["internal", "external"]:
        for group, files in sorted(groups[dataset].items()):
            seeds_predictions = []
            seeds_ranks = []
            for file in files:
                df = pl.read_csv(file).sort("sample_idx")
                n = df.height
                if n > 1:
                    df = df.with_columns(
                        ((pl.col("prediction").rank(method="average") - 1.0) / (n - 1.0)).alias("frac_rank")
                    )
                else:
                    df = df.with_columns(pl.lit(0.0).alias("frac_rank"))

                pred = df["prediction"].to_numpy()
                frac_rank = df["frac_rank"].to_numpy()
                seeds_predictions.append(pred)
                seeds_ranks.append(frac_rank)
                
                if labels[dataset] is None:
                    labels[dataset] = {
                        "label_1d": df["label_1d"].to_numpy(),
                        "label_3d": df["label_3d"].to_numpy(),
                        "label_7d": df["label_7d"].to_numpy(),
                        "label_14d": df["label_14d"].to_numpy(),
                        "label_28d": df["label_28d"].to_numpy()
                    }
            
            ensemble_rank = np.mean(seeds_ranks, axis=0)
            
            if dataset not in loaded_data:
                loaded_data[dataset] = {}
            loaded_data[dataset][group] = {
                "seeds_predictions": seeds_predictions,
                "seeds_ranks": seeds_ranks,
                "ensemble_rank": ensemble_rank
            }

    # Define strict ordering for groups and horizons
    model_groups_ordered = ["TD", "1d", "1d+", "3d", "3d+", "7d", "7d+", "14d", "14d+", "28d", "28d+"]
    horizons_ordered = ["1d", "3d", "7d", "14d", "28d"]

    # 1. Compute raw AUROCs for each model on each horizon
    print("Preparing parallel AUROC tasks...")
    tasks = []
    
    # Step A: Flatten the nested loops into a list of standalone tasks
    for dataset in ["internal", "external"]:
        for group in model_groups_ordered:
            if group not in loaded_data[dataset]:
                continue
            for h in horizons_ordered:
                gt = labels[dataset][f"label_{h}"]
                ens_rank = loaded_data[dataset][group]["ensemble_rank"]
                seeds_preds = loaded_data[dataset][group]["seeds_predictions"]
                
                # Append the raw data needed for this specific combination
                tasks.append((dataset, group, h, gt, ens_rank, seeds_preds))

    # Step B: Execute all tasks in parallel across all CPU cores
    print(f"Executing {len(tasks)} tasks across multiple cores...")
    # n_jobs=-1 tells Joblib to use all available CPU cores
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_compute_single_group_horizon)(*task) 
        for task in tqdm(tasks, desc="Calculating AUROCs")
    )

    # Step C: Reassemble the results back into your nested dictionary format
    raw_auroc_results = {"internal": {}, "external": {}}
    for dataset in ["internal", "external"]:
        raw_auroc_results[dataset] = {}
        for group in model_groups_ordered:
            raw_auroc_results[dataset][group] = {}

    for res in results:
        dataset, group, h, ens_auc, seed_mean, seed_std, seed_aucs = res
        raw_auroc_results[dataset][group][h] = {
            "ensemble": ens_auc,
            "seed_mean": seed_mean,
            "seed_std": seed_std,
            "seed_aucs": seed_aucs
        }

    # 2. Perform TD vs Supervised comparisons across all horizons
    all_tests = []
    for dataset in ["internal", "external"]:
        if "TD" not in loaded_data[dataset]:
            print(f"Warning: No TD models found for {dataset} dataset. Skipping comparisons.")
            continue
        
        td_ensemble = loaded_data[dataset]["TD"]["ensemble_rank"]
        
        for h in tqdm(horizons_ordered, desc=f"Running DeLong Tests ({dataset})"):
            gt = labels[dataset][f"label_{h}"]
            
            for group in model_groups_ordered:
                if group == "TD" or group not in loaded_data[dataset]:
                    continue
                
                group_ensemble = loaded_data[dataset][group]["ensemble_rank"]
                
                # Perform DeLong's test
                try:
                    auc_td, auc_group, p_val = delong_roc_test(gt, td_ensemble, group_ensemble)
                    all_tests.append({
                        "dataset": dataset,
                        "target": h,
                        "comparison": f"TD vs {group}",
                        "auc_td": auc_td,
                        "auc_supervised": auc_group,
                        "p_value": p_val
                    })
                except Exception as e:
                    print(f"Error performing DeLong's test for {dataset} {group} on {h}: {e}")

    # Apply BH correction globally
    if all_tests:
        p_values = [t["p_value"] for t in all_tests]
        adj_p_values = benjamini_hochberg_correction(p_values)
        for idx, adj_p in enumerate(adj_p_values):
            all_tests[idx]["adj_p_value"] = adj_p

    # Format the report
    report_lines = []
    report_lines.append("# Statistical Testing and Model Comparison Report")
    report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("## Methodology Summary")
    report_lines.append("1. **Fractional Ranks**: For each seed model and dataset, probabilistic predictions were converted to fractional ranks (0-1) across the dataset, handling ties using the average rank method.")
    report_lines.append("2. **Rank-Averaged Ensemble**: Fractional ranks were averaged across all available seeds for each model group to construct a stable ensembled prediction score.")
    report_lines.append("3. **Cross-Horizon Evaluation**: All model groups (both TD and supervised model categories) were evaluated across every target mortality horizon label (1, 3, 7, 14, and 28 day mortality).")
    report_lines.append("4. **DeLong's Test**: The ensembled TD model group was compared directly to each supervised model group (both standard and balanced `+` models) for every target mortality horizon using DeLong's test for correlated ROC curves.")
    report_lines.append("5. **FDR Control**: P-values across all tests were corrected globally using the Benjamini-Hochberg procedure.\n")

    # Add Raw AUROC tables first
    report_lines.append("## Raw AUROC Performance by Model and Horizon")
    report_lines.append("Values are displayed as **Ensemble AUROC (Seed Mean ± SD)** across the 5 training seeds.\n")

    for dataset in ["internal", "external"]:
        report_lines.append(f"### {dataset.upper()} Dataset - Raw AUROC Scores")
        report_lines.append("| Model Group | 1d Mortality | 3d Mortality | 7d Mortality | 14d Mortality | 28d Mortality |")
        report_lines.append("| --- | --- | --- | --- | --- | --- |")
        for group in model_groups_ordered:
            if group not in raw_auroc_results[dataset]:
                continue
            cells = []
            for h in horizons_ordered:
                metrics = raw_auroc_results[dataset][group][h]
                cells.append(f"{metrics['ensemble']:.5f} ({metrics['seed_mean']:.5f} ± {metrics['seed_std']:.5f})")
            report_lines.append(f"| **{group}** | " + " | ".join(cells) + " |")
        report_lines.append("\n")

    # Add Statistical Comparisons tables
    report_lines.append("## Statistical Comparisons (TD vs. Supervised Models)")
    report_lines.append("DeLong's test was conducted for every target horizon comparing the TD ensemble against each supervised ensemble model. P-values were adjusted globally for multiple comparisons using the Benjamini-Hochberg procedure.\n")

    for dataset in ["internal", "external"]:
        dataset_tests = [t for t in all_tests if t["dataset"] == dataset]
        if not dataset_tests:
            continue
        
        # Sort tests to group by target horizon first
        dataset_tests_sorted = sorted(dataset_tests, key=lambda x: (horizons_ordered.index(x["target"]), x["comparison"]))

        report_lines.append(f"### {dataset.upper()} Dataset - DeLong Comparison Tests")
        report_lines.append("| Target Horizon | Comparison | TD Ensemble AUC | Supervised Ensemble AUC | Difference | Raw p-value | Adjusted p-value (BH) | Significance (α=0.05) |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        
        for t in dataset_tests_sorted:
            sig = "★ Significant" if t["adj_p_value"] < 0.05 else "Not Significant"
            diff = t["auc_td"] - t["auc_supervised"]
            sign_char = "+" if diff >= 0 else ""
            report_lines.append(
                f"| {t['target']} | {t['comparison']} | {t['auc_td']:.5f} | {t['auc_supervised']:.5f} | {sign_char}{diff:.5f} | "
                f"{t['p_value']:.2g} | {t['adj_p_value']:.2g} | {sig} |"
            )
        report_lines.append("\n")

    # Add Seed-Level Raw AUROC tables at the end of the report
    report_lines.append("## Detailed Seed-Level Raw AUROC Scores")
    report_lines.append("Below are the individual seed AUROC scores (one for each of the 5 training seeds) for each model category, target horizon, and dataset.\n")

    for dataset in ["internal", "external"]:
        report_lines.append(f"### {dataset.upper()} Dataset - Seed-Level Raw AUROC Scores")
        report_lines.append("| Model Group | Target Horizon | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for group in model_groups_ordered:
            if group not in raw_auroc_results[dataset]:
                continue
            for h in horizons_ordered:
                metrics = raw_auroc_results[dataset][group][h]
                seed_vals = metrics["seed_aucs"]
                # Format to 5 decimal places
                formatted_seeds = [f"{val:.5f}" for val in seed_vals]
                # If there are fewer than 5 seeds, pad with N/A
                while len(formatted_seeds) < 5:
                    formatted_seeds.append("N/A")
                report_lines.append(f"| **{group}** | {h} | " + " | ".join(formatted_seeds) + " |")
        report_lines.append("\n")

    report_content = "\n".join(report_lines)
    
    # Save the report
    report_path = os.path.join(results_dir, "statistical_test_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    
    # Save a separate structured JSON of raw seed AUROCs for programmatic usage
    import json
    json_results = []
    for dataset in ["internal", "external"]:
        for group in model_groups_ordered:
            if group not in raw_auroc_results[dataset]:
                continue
            for h in horizons_ordered:
                metrics = raw_auroc_results[dataset][group][h]
                json_results.append({
                    "dataset": dataset,
                    "model_group": group,
                    "target_horizon": h,
                    "ensemble_auroc": metrics["ensemble"],
                    "seed_mean_auroc": metrics["seed_mean"],
                    "seed_std_auroc": metrics["seed_std"],
                    "seed_aurocs": [float(val) for val in metrics["seed_aucs"]]
                })

    json_path = os.path.join(results_dir, "seed_level_aurocs.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=4)
    
    print("\n" + "="*50)
    print("Statistical Testing Completed successfully!")
    print(f"Report written to: {report_path}")
    print(f"Structured JSON written to: {json_path}")
    print("="*50)
    print(report_content)

if __name__ == "__main__":
    run_statistical_analysis()
