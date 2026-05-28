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
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

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

    report_lines = []
    report_lines.append("# Statistical Testing and Model Comparison Report")
    report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("## Methodology Summary")
    report_lines.append("1. **Fractional Ranks**: For each seed model and dataset, probabilistic predictions were converted to fractional ranks (0-1) across the dataset, handling ties using the average rank method.")
    report_lines.append("2. **Rank-Averaged Ensemble**: Fractional ranks were averaged across all available seeds for each model group to construct a stable ensembled prediction score.")
    report_lines.append("3. **DeLong's Test**: The ensembled TD model group was compared directly to each available supervised model group (both standard and balanced `+` models) on the corresponding mortality target label using DeLong's test for correlated ROC curves.")
    report_lines.append("4. **FDR Control**: P-values across all tests were corrected globally using the Benjamini-Hochberg procedure.\n")

    # Define target label mapping for DeLong comparison
    target_mapping = {
        "1d": "label_1d",
        "1d+": "label_1d",
        "3d": "label_3d",
        "3d+": "label_3d",
        "7d": "label_7d",
        "7d+": "label_7d",
        "14d": "label_14d",
        "14d+": "label_14d",
        "28d": "label_28d",
        "28d+": "label_28d"
    }

    all_tests = []
    
    for dataset in ["internal", "external"]:
        td_files = groups[dataset].get("TD", [])
        if not td_files:
            print(f"Warning: No TD models found for {dataset} dataset. Skipping TD comparisons.")
            continue
        
        # Build TD ensembled predictions
        td_ranks = []
        td_labels = None
        for file in td_files:
            df = pl.read_csv(file).sort("sample_idx")
            td_ranks.append(compute_fractional_ranks(df["prediction"].to_numpy()))
            if td_labels is None:
                # Keep all label columns
                td_labels = {
                    "label_1d": df["label_1d"].to_numpy(),
                    "label_3d": df["label_3d"].to_numpy(),
                    "label_7d": df["label_7d"].to_numpy(),
                    "label_14d": df["label_14d"].to_numpy(),
                    "label_28d": df["label_28d"].to_numpy()
                }
        td_ensemble = np.mean(td_ranks, axis=0)

        # Loop through other model groups and compare
        for group, files in sorted(groups[dataset].items()):
            if group == "TD":
                continue
            
            # Identify corresponding label
            label_col = target_mapping.get(group)
            if not label_col or label_col not in td_labels:
                print(f"Warning: Could not resolve target label for group '{group}'. Skipping.")
                continue
            
            # Build group ensembled predictions
            group_ranks = []
            for file in files:
                df = pl.read_csv(file).sort("sample_idx")
                group_ranks.append(compute_fractional_ranks(df["prediction"].to_numpy()))
            group_ensemble = np.mean(group_ranks, axis=0)

            # Get target ground truth labels
            ground_truth = td_labels[label_col]

            # Perform DeLong's test
            try:
                auc_td, auc_group, p_val = delong_roc_test(ground_truth, td_ensemble, group_ensemble)
                all_tests.append({
                    "dataset": dataset,
                    "target": label_col.replace("label_", ""),
                    "comparison": f"TD vs {group}",
                    "auc_td": auc_td,
                    "auc_supervised": auc_group,
                    "p_value": p_val
                })
            except Exception as e:
                print(f"Error performing DeLong's test for {dataset} '{group}': {e}")

    if not all_tests:
        print("No valid statistical comparisons were performed.")
        return

    # Apply BH correction globally
    p_values = [t["p_value"] for t in all_tests]
    adj_p_values = benjamini_hochberg_correction(p_values)
    for idx, adj_p in enumerate(adj_p_values):
        all_tests[idx]["adj_p_value"] = adj_p

    # Format and save report
    for dataset in ["internal", "external"]:
        dataset_tests = [t for t in all_tests if t["dataset"] == dataset]
        if not dataset_tests:
            continue
        
        report_lines.append(f"## {dataset.upper()} Dataset Comparisons")
        report_lines.append("| Target Horizon | Comparison | TD Ensemble AUC | Supervised Ensemble AUC | Raw p-value | Adjusted p-value (BH) | Significance (α=0.05) |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        
        for t in dataset_tests:
            sig = "★ Significant" if t["adj_p_value"] < 0.05 else "Not Significant"
            report_lines.append(
                f"| {t['target']} | {t['comparison']} | {t['auc_td']:.5f} | {t['auc_supervised']:.5f} | "
                f"{t['p_value']:.2g} | {t['adj_p_value']:.2g} | {sig} |"
            )
        report_lines.append("\n")

    report_content = "\n".join(report_lines)
    
    # Save the report
    report_path = os.path.join(results_dir, "statistical_test_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print("\n" + "="*50)
    print("Statistical Testing Completed successfully!")
    print(f"Report written to: {report_path}")
    print("="*50)
    print(report_content)

if __name__ == "__main__":
    run_statistical_analysis()
