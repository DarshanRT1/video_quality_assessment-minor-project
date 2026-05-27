import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/drive3/low_encoder/results_10k/val_metrics_detailed.csv')

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.set_theme(style="whitegrid")

# 1. MOS Correlation Metrics
axes[0, 0].plot(df['epoch'], df['mos_srcc'], label='SRCC', linewidth=2)
axes[0, 0].plot(df['epoch'], df['mos_plcc'], label='PLCC', linewidth=2)
axes[0, 0].set_title('MOS Correlation (SRCC/PLCC)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()

# 2. Loss
axes[0, 1].plot(df['epoch'], df['loss'], color='red', linewidth=2)
axes[0, 1].set_title('Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')

# 3. Artifact F1 Scores
f1_cols = [c for c in df.columns if 'f1' in c]
for col in f1_cols:
    name = col.replace('_flag_f1', '').capitalize()
    axes[1, 0].plot(df['epoch'], df[col], label=name, alpha=0.8)
axes[1, 0].set_title('Artifact Detection F1 Scores')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].legend(loc='lower right', ncol=2)

# 4. RMSE vs Quality Acc
ax1 = axes[1, 1]
ax2 = ax1.twinx()
l1 = ax1.plot(df['epoch'], df['mos_rmse'], color='purple', label='MOS RMSE', linewidth=2)
l2 = ax2.plot(df['epoch'], df['quality_accuracy'], color='green', label='Quality Acc', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('RMSE', color='purple')
ax2.set_ylabel('Accuracy', color='green')
ax1.set_title('RMSE & Quality Classification Accuracy')
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.tight_layout()
plt.savefig('/home/drive3/low_encoder/results_10k/learning_curves.png', dpi=300)
print("Plot saved.")
