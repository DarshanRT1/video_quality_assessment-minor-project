import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid", context="talk")

variants = ['xs', 's', 'm']
labels = ['X3D-XS\n(4 frames, 160p)', 'X3D-S\n(13 frames, 160p)', 'X3D-M\n(16 frames, 256p)']
artifacts = ['spatial', 'lighting', 'rendering', 'hallucination', 'physics', 'flicker', 'motion']
artifact_labels = [a.capitalize() for a in artifacts]

mos_data = []
auc_data = []

for var, label in zip(variants, labels):
    test_csv = f"/home/drive3/low_encoder/results_10k_{var}/test_metrics_detailed.csv"
    if not os.path.exists(test_csv):
        print(f"Missing {test_csv}")
        continue
    
    df = pd.read_csv(test_csv)
    row = df.iloc[0]
    
    # Collect MOS Data
    mos_data.append({
        'Model': label,
        'Metric': 'SRCC',
        'Score': row['mos_srcc']
    })
    mos_data.append({
        'Model': label,
        'Metric': 'PLCC',
        'Score': row['mos_plcc']
    })
    
    # Collect AUC Data
    for art, art_label in zip(artifacts, artifact_labels):
        auc = row.get(f'{art}_auc', 0.5)
        auc_data.append({
            'Model': label,
            'Artifact': art_label,
            'AUC Score': auc
        })

# 1. Plot MOS Comparison
df_mos = pd.DataFrame(mos_data)
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_mos, x='Model', y='Score', hue='Metric', palette='viridis')

# Annotate bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=12)

plt.title('MOS Regression Performance by Backbone Capacity', pad=20, fontweight='bold')
plt.ylim(0.55, 0.72)
plt.ylabel('Correlation Score')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('mos_comparison.png', dpi=300)
print("Saved mos_comparison.png")

# 2. Plot Artifact AUC Comparison
df_auc = pd.DataFrame(auc_data)
plt.figure(figsize=(14, 7))
ax = sns.barplot(data=df_auc, x='Artifact', y='AUC Score', hue='Model', palette='rocket')

# Annotate bars (rotate to fit)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9, rotation=90)

plt.title('Artifact Detection AUC by Backbone Capacity', pad=20, fontweight='bold')
plt.ylim(0.35, 0.8)
plt.ylabel('AUC Score')
plt.legend(title='Backbone', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('artifact_auc_comparison.png', dpi=300)
print("Saved artifact_auc_comparison.png")
