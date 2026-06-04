import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Parse significance dynamically from statistical_test_report.md
def parse_statistical_report(report_path):
    results = {}
    if not os.path.exists(report_path):
        print(f"Warning: Report file '{report_path}' not found. Using fallback significance.")
        return results
        
    current_dataset = None
    with open(report_path, 'r') as f:
        for line in f:
            line = line.strip()
            if "### INTERNAL Dataset - DeLong & Bootstrap Comparison Tests" in line:
                current_dataset = "internal"
            elif "### EXTERNAL Dataset - DeLong & Bootstrap Comparison Tests" in line:
                current_dataset = "external"
            elif line.startswith("|") and current_dataset is not None:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 16:
                    horizon_part = parts[1].split()[0]  # e.g., "1d"
                    comparison_part = parts[2]          # e.g., "TD vs 14d"
                    if not comparison_part.startswith("TD vs "):
                        continue
                    model = comparison_part.replace("TD vs ", "").strip()
                    
                    # Parse adj p-value
                    try:
                        adj_p = float(parts[7])
                    except ValueError:
                        adj_p = 1.0
                        
                    # Parse CIs
                    def parse_ci(ci_str):
                        ci_str = ci_str.replace("[", "").replace("]", "").strip()
                        ci_parts = [float(x) for x in ci_str.split(",")]
                        return ci_parts[0], ci_parts[1]
                        
                    try:
                        ci_95 = parse_ci(parts[11])
                        ci_99 = parse_ci(parts[12])
                        ci_999 = parse_ci(parts[13])
                    except Exception:
                        ci_95 = (0.0, 0.0)
                        ci_99 = (0.0, 0.0)
                        ci_999 = (0.0, 0.0)
                        
                    results[(current_dataset, horizon_part, model)] = {
                        "auroc_p": adj_p,
                        "auprc_95_ci": ci_95,
                        "auprc_99_ci": ci_99,
                        "auprc_999_ci": ci_999
                    }
    return results

# Load significance mapping globally
parsed_sig = parse_statistical_report('./evaluation_results/statistical_test_report.md')

# 2. Helper function to determine significance based on parsed results
def get_significance(dataset, horizon, model, metric):
    if model == 'TD':
        return ""  # Reference group doesn't get a star
        
    key = (dataset, horizon, model)
    if key not in parsed_sig:
        return "***"  # Fallback
        
    if metric == 'AUROC':
        adj_p = parsed_sig[key]["auroc_p"]
        if adj_p >= 0.05:
            return "ns"
        elif adj_p >= 0.01:
            return "*"
        elif adj_p >= 0.001:
            return "**"
        else:
            return "***"
    elif metric == 'AUPRC':
        ci_95 = parsed_sig[key]["auprc_95_ci"]
        ci_99 = parsed_sig[key]["auprc_99_ci"]
        ci_999 = parsed_sig[key]["auprc_999_ci"]
        
        # Check if intervals contain 0 (i.e. low <= 0 <= high)
        if ci_95[0] <= 0.0 <= ci_95[1]:
            return "ns"
        elif ci_99[0] <= 0.0 <= ci_99[1]:
            return "*"
        elif ci_999[0] <= 0.0 <= ci_999[1]:
            return "**"
        else:
            return "***"
            
    return ""

# 3. Core plotting function
def generate_plot(metric):
    # Load the dataset
    with open('./evaluation_results/seed_level_aurocs.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Configure variables for the target metric
    seed_col = 'seed_aurocs' if metric == 'AUROC' else 'seed_auprcs'
    ens_col = 'ensemble_auroc' if metric == 'AUROC' else 'ensemble_auprc'
    file_name = 'final_aurocs_with_significance.pdf' if metric == 'AUROC' else 'final_auprcs_with_significance.pdf'
    
    df_exploded = df.explode(seed_col)
    df_exploded[seed_col] = df_exploded[seed_col].astype(float)
    
    datasets = ['internal', 'external']
    horizons = ['1d', '3d', '7d', '14d', '28d']
    model_order = ['TD', '1d', '1d+', '3d', '3d+', '7d', '7d+', '14d', '14d+', '28d', '28d+']
    
    # Custom color palette (TD = Orange, everything else = Blue)
    colors = {'TD': '#F46C43'}
    for m in model_order[1:]:
        colors[m] = '#4581B5'
        
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(22, 10))
    
    for i, ds in enumerate(datasets):
        for j, hz in enumerate(horizons):
            ax = axes[i, j]
            
            sub_exp = df_exploded[(df_exploded['dataset'] == ds) & (df_exploded['target_horizon'] == hz)]
            sub_ens = df[(df['dataset'] == ds) & (df['target_horizon'] == hz)]
            
            # Draw the Bar Plot
            sns.barplot(
                data=sub_ens, x='model_group', y=ens_col,
                order=model_order, palette=colors, hue='model_group', legend=False, alpha=0.7, ax=ax, zorder=1
            )
            
            # Draw the Swarm Plot
            sns.swarmplot(
                data=sub_exp, x='model_group', y=seed_col,
                order=model_order, palette=colors, ax=ax, hue='model_group', legend=False,
                size=5, edgecolor='black', linewidth=0.8, zorder=2, warn_thresh=0.9
            )
            
            # Add Significance Stars
            for idx, model in enumerate(model_order):
                if model == 'TD':
                    continue
                    
                model_seeds = sub_exp[sub_exp['model_group'] == model][seed_col]
                model_ens = sub_ens[sub_ens['model_group'] == model][ens_col]
                
                if len(model_seeds) > 0 and len(model_ens) > 0:
                    model_max_y = max(model_seeds.max(), model_ens.max())
                else:
                    continue
                    
                sig_text = get_significance(ds, hz, model, metric)
                
                # Place the text slightly above the highest point of the swarm
                ax.text(
                    x=idx,
                    y=model_max_y + 0.005,  # Offset above the highest seed
                    s=sig_text,
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    fontweight='bold',
                    color='#333333'
                )
                
            # Formatting
            if metric == 'AUROC':
                ax.set_ylim(0.5, 0.99)
            else:
                ax.set_ylim(0.0, 0.85)
                
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=60, labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            
            if i == 0:
                ax.set_title(f"{hz} mortality", fontsize=30, pad=0, fontweight='bold', color='#1A3B5C')
                
            if j == 0:
                row_label = "A (Internal)" if i == 0 else "B (External)"
                ax.set_ylabel(f"{row_label}\n\n{metric}", fontsize=16, fontweight='bold', rotation=0, labelpad=50, va='center')
            else:
                ax.set_ylabel("")
                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
                
    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    
    save_path = os.path.join('./figures', file_name)
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Plot successfully generated and saved as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    print("Generating AUROC Plot...")
    generate_plot('AUROC')
    print("\nGenerating AUPRC Plot...")
    generate_plot('AUPRC')
