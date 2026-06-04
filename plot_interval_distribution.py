import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

df = pl.concat([pl.scan_parquet(f'./data/mimic/{key}/dataframe_{key}/*.parquet') for key in ['train', 'val', 'test']])
df = df.filter(pl.col('next_labeltime').is_not_null()).select((pl.col('next_labeltime') - pl.col('labeltime')).dt.total_minutes() / 60).collect().to_numpy()

# --- 1. Data Preparation ---
# Flattens your Polars/NumPy array to a 1D vector
data = df.flatten()

# --- 2. Advanced Boundary Correction (Reflection Method) ---
# Standard KDE leaks mass below the hard 24-hour boundary.
# We reflect the data across x = 24 to correct this boundary bias.
boundary = 24.0
reflected_data = boundary - (data - boundary)
combined_data = np.concatenate([data, reflected_data])

# Fit the Kernel Density Estimate on the augmented dataset
kde = gaussian_kde(combined_data)

# Generate smooth evaluation points from 24 to 36 hours
x_eval = np.linspace(24, 36, 1000)

# Evaluate and multiply by 2 to re-normalize the reflected distribution density
pdf_vals = kde(x_eval) * 2

# Calculate the Empirical CDF directly from your original data for 100% precision
cdf_vals = np.array([np.mean(data <= x) for x in x_eval])

# --- 3. Publication-Quality Styling ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.edgecolor': '#000000',   # Pure black borders
    'axes.labelcolor': '#000000',  # Pure black axis labels
    'xtick.color': '#000000',      # Pure black x-ticks
    'ytick.color': '#000000',      # Pure black y-ticks
    'text.color': '#000000'        # Pure black text (legend, etc.)
})

fig, ax1 = plt.subplots(figsize=(6, 4), dpi=600)

# Colors: Professional lighter steel blue for the data curves
pdf_color = '#756bb1'
cdf_color = '#bcbddc'

# --- Plot PDF (Left Axis) ---
line_pdf, = ax1.plot(x_eval, pdf_vals, color=pdf_color, linewidth=2.2, label='PDF')
ax1.set_xlabel('Hours Until Next State', fontsize=12, fontweight='bold', labelpad=8)
ax1.set_ylabel('Density Estimate (PDF)', fontsize=12, fontweight='bold', labelpad=11)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_xlim(24, 36)
ax1.set_ylim(0, max(pdf_vals) * 1.05)

# Faint grid lines only on the primary axis to prevent dual-axis grid collisions
ax1.grid(True, linestyle='--', alpha=0.4, color='#cbd5e1')

# --- Plot CDF (Right Axis) ---
ax2 = ax1.twinx()
line_cdf, = ax2.plot(x_eval, cdf_vals, color=cdf_color, linewidth=2.0, linestyle='--', label='CDF')
ax2.set_ylabel('Cumulative Proportion (CDF)', fontsize=12, fontweight='bold', labelpad=11)
ax2.tick_params(axis='y', labelsize=10)
ax2.set_ylim(0, 1.02)
ax2.grid(False)  # Keeps the background clean

# --- Integrated Legend & Layout ---
lines = [line_pdf, line_cdf]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.92, 0.85),
           frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.95, fontsize=11)

plt.tight_layout()

# --- 4. Exports ---
os.makedirs('./figures', exist_ok=True)
plt.savefig('./figures/state_interval_distribution.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()