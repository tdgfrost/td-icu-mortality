import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load data and clean the X-axis categories
df = pd.read_csv('./evaluation_results/state_interval_results.csv')

# Extract numeric hour values for sorting and clean labels (e.g., "TD-4hr" -> "4h")
df['Hours'] = df['Models'].str.extract(r'(\d+)').astype(int)
df = df.sort_values('Hours')
df['X_Labels'] = df['Hours'].astype(str) + 'h'

# 2. Set up high-quality publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'  # Softened axis edges
plt.rcParams['axes.linewidth'] = 0.8

# Initialize figure with balanced dimensions
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.set_axisbelow(True)

# Elegant horizontal-only gridlines for a cleaner background
ax.grid(True, axis='y', linestyle=':', alpha=0.6, color='#999999')

# 3. Defined polished, cohesive color palette and line-styles
# (Grayscale robust: pairing distinct markers with unique line dash patterns)
horizons = [
    ('1-day-mortality', '1-Day Mortality', '#0F2C59', 'o', '-'),     # Deep Navy / Solid
    ('3-day-mortality', '3-Day Mortality', '#1D5D9B', 's', '--'),    # Steel Blue / Dashed
    ('7-day-mortality', '7-Day Mortality', '#6499E9', 'D', '-.'),    # Soft Blue / Dash-Dot
    ('14-day-mortality', '14-Day Mortality', '#F3A738', '^', ':'),   # Muted Amber / Dotted
    ('28-day-mortality', '28-Day Mortality', '#CC5A37', 'v', (0, (3, 1, 1, 1))) # Terracotta / Alternative Dash
]

# 4. Plot each horizon line
for col_name, label, color, marker, linestyle in horizons:
    ax.plot(
        df['X_Labels'],
        df[col_name],
        label=label,
        color=color,
        marker=marker,
        markersize=6.0,
        linewidth=1.6,
        linestyle=linestyle,
        markeredgecolor='white',
        markeredgewidth=0.7,
        alpha=0.95
    )

# 5. Fine-tune axes, labels, and boundaries
ax.set_ylim(0.80, 0.91)  # Raised slightly to clear whitespace for the legend
ax.set_xlabel('TD State Interval Delay (Hours)', fontsize=12, fontweight='bold', labelpad=10, color='#333333')
ax.set_ylabel('Ensemble AUROC Score', fontsize=12, fontweight='bold', labelpad=20, color='#333333')

# Style ticks with professional grey tones
ax.tick_params(axis='both', which='major', labelsize=10, colors='#444444')

# Clean layout borders (Despine top and right)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# 6. Add polished layout for the legend in the open upper-left quadrant
ax.legend(
    loc=(0.05, 0.75),
    frameon=True,
    facecolor='white',
    edgecolor='#EAEAEA',
    framealpha=0.92,
    fontsize=8.5,
    shadow=False
).get_frame().set_linewidth(0.8)

# Optimize spaces and save vector-ready file for LaTeX integration
ax.xaxis.grid(False)
plt.tight_layout()
os.makedirs('./figures', exist_ok=True)
plt.savefig('./figures/td_interval_performance.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()
