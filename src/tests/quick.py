import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data ---
lat_path = 'logs/HNSWPostfilter_YFCC_20260609-225348/latencies.npy'
recall_path = 'logs/HNSWPostfilter_YFCC_20260609-225348/recalls.npy'
selectivity_path = 'data/YFCC/subsets/1/analysis/selectivity.npy'
output_path = 'benchmark_plots'

os.makedirs(output_path, exist_ok=True)

latencies = np.load(lat_path)
recalls = np.load(recall_path)
selectivities = np.load(selectivity_path)

df = pd.DataFrame({
    'recall': recalls,
    'latency': latencies,
    'selectivity': selectivities
})

# Convert latencies to milliseconds for better readability if they are in seconds
# Remove this line if your raw npy data is already in milliseconds!
df['latency_ms'] = df['latency'] * 1000 

# Sort by selectivity for the trend lines
df = df.sort_values('selectivity').reset_index(drop=True)

# Calculate rolling averages to see the correlation trends
# window=200 means it averages the nearest 200 queries around any given selectivity
df['avg_recall'] = df['recall'].rolling(window=200, center=True, min_periods=1).mean()
df['avg_latency'] = df['latency_ms'].rolling(window=200, center=True, min_periods=1).mean()

# Set style
sns.set_theme(style="whitegrid")

# --- Plot 1: Hexbin Plots (Density Counts) ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# A. Recall vs Latency Hexbin
hb1 = axes[0].hexbin(df['recall'], df['latency_ms'], gridsize=15, cmap='YlGnBu', mincnt=1)
fig.colorbar(hb1, ax=axes[0], label='Number of Queries')
axes[0].set_title('Recall vs Latency Density')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Latency (ms)')

# B. Selectivity vs Recall Hexbin
hb2 = axes[1].hexbin(df['selectivity'], df['recall'], gridsize=20, cmap='YlGnBu', mincnt=1)
fig.colorbar(hb2, ax=axes[1], label='Number of Queries')
axes[1].set_title('Selectivity vs Recall Density')
axes[1].set_xlabel('Selectivity (Fraction of data passing filter)')
axes[1].set_ylabel('Recall')

# C. Selectivity vs Latency Hexbin
hb3 = axes[2].hexbin(df['selectivity'], df['latency_ms'], gridsize=20, cmap='YlGnBu', mincnt=1)
fig.colorbar(hb3, ax=axes[2], label='Number of Queries')
axes[2].set_title('Selectivity vs Latency Density')
axes[2].set_xlabel('Selectivity (Fraction of data passing filter)')
axes[2].set_ylabel('Latency (ms)')

plt.tight_layout()
plt.savefig(f'{output_path}/density_hexbins.png', dpi=300)
plt.close()


# --- Plot 2: Selectivity Correlation Trend Lines ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# A. Selectivity vs Average Recall Trend
# Background scatter with low opacity so we can see the barcode distribution safely
ax1.scatter(df['selectivity'], df['recall'], alpha=0.05, color='gray', s=10, label='Individual Queries')
# The continuous correlation line
ax1.plot(df['selectivity'], df['avg_recall'], color='crimson', linewidth=3, label='Rolling Average Trend')
ax1.set_title('How Selectivity Affects Accuracy (Recall)')
ax1.set_xlabel('Selectivity (Higher = Looser Filter)')
ax1.set_ylabel('Recall')
ax1.set_ylim(-0.05, 1.05)
ax1.legend()

# B. Selectivity vs Average Latency Trend
ax2.scatter(df['selectivity'], df['latency_ms'], alpha=0.05, color='gray', s=10, label='Individual Queries')
ax2.plot(df['selectivity'], df['avg_latency'], color='darkorange', linewidth=3, label='Rolling Average Trend')
ax2.set_title('How Selectivity Affects Search Speed (Latency)')
ax2.set_xlabel('Selectivity (Higher = Looser Filter)')
ax2.set_ylabel('Latency (ms)')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{output_path}/selectivity_trends.png', dpi=300)
plt.close()

print(f"Plots successfully generated and saved to the '{output_path}' directory!")