import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
with open('./evaluation_results/seed_level_aurocs.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

df_exploded = df.explode('seed_aurocs')
df_exploded['seed_aurocs'] = df_exploded['seed_aurocs'].astype(float)

# 2. Setup the grid layout and aesthetics
datasets = ['internal', 'external']
horizons = ['1d', '3d', '7d', '14d', '28d']

# Original alternating order
model_order = ['TD', '1d', '1d+', '3d', '3d+', '7d', '7d+', '14d', '14d+', '28d', '28d+']

# Custom color palette (TD = Orange, everything else = Blue)
colors = {'TD': '#F46C43'}
for m in model_order[1:]:
    colors[m] = '#4581B5'

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(22, 10))
sns.set_theme(style="whitegrid")


# 3. Helper function to determine significance based on your specific report
def get_significance(dataset, horizon, model):
    if model == 'TD':
        return ""  # Reference group doesn't get a star

    # Internal dataset specific p-values
    if dataset == 'internal':
        if horizon == '1d' and model == '7d':
            return "ns"  # p = 0.91
        elif horizon == '3d' and model == '7d+':
            return "*"  # p = 0.014
        elif horizon == '7d' and model == '3d':
            return "*"  # p = 0.031

    # If not explicitly caught above, your report shows p < 0.001 for all others
    return "***"


# 4. Iterate through rows and columns to build subplots
for i, ds in enumerate(datasets):
    for j, hz in enumerate(horizons):
        ax = axes[i, j]

        sub_exp = df_exploded[(df_exploded['dataset'] == ds) & (df_exploded['target_horizon'] == hz)]
        sub_ens = df[(df['dataset'] == ds) & (df['target_horizon'] == hz)]

        # Draw the Bar Plot
        sns.barplot(
            data=sub_ens, x='model_group', y='ensemble_auroc',
            order=model_order, palette=colors, hue='model_group', legend=False, alpha=0.7, ax=ax, zorder=1
        )

        # Draw the Swarm Plot
        sns.swarmplot(
            data=sub_exp, x='model_group', y='seed_aurocs',
            order=model_order, palette=colors, ax=ax, hue='model_group', legend=False,
            size=5, edgecolor='black', linewidth=0.8, zorder=2, warn_thresh=0.9
        )

        # Add Significance Stars
        for idx, model in enumerate(model_order):
            if model == 'TD':
                continue

            # Find the max y-value for this specific model to place the star above the highest dot
            model_max_y = max(sub_exp[sub_exp['model_group'] == model]['seed_aurocs'].max(),
                              sub_exp[sub_exp['model_group'] == model]['ensemble_auroc'].max())

            sig_text = get_significance(ds, hz, model)

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

        # 5. Formatting
        ax.set_ylim(0.5, 0.99)  # Increased top limit slightly to fit stars
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=60, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        if i == 0:
            ax.set_title(f"{hz} mortality", fontsize=30, pad=0, fontweight='bold', color='#1A3B5C')

        if j == 0:
            row_label = "A (Internal)" if i == 0 else "B (External)"
            ax.set_ylabel(f"{row_label}\n\nAUROC", fontsize=16, fontweight='bold', rotation=0, labelpad=50, va='center')
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if j > 0:
            ax.spines['left'].set_visible(False)

# 6. Final layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
os.makedirs('./figures', exist_ok=True)
plt.savefig('./figures/final_aurocs_with_significance.pdf', format='pdf', dpi=600, bbox_inches='tight')
print("Plot successfully generated and saved as 'final_aurocs_with_significance.pdf'")
plt.show()
