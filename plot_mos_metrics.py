import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load validation metrics
df = pd.read_csv('/home/drive3/low_encoder/results_10k/val_metrics_detailed.csv')

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.set_theme(style="whitegrid")

# 1. Validation SRCC vs Epochs
axes[0].plot(df['epoch'], df['mos_srcc'], color='blue', linewidth=2)
axes[0].set_title('Validation SRCC vs Epochs', fontsize=14)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('SRCC', fontsize=12)

# 2. Validation PLCC vs Epochs
axes[1].plot(df['epoch'], df['mos_plcc'], color='green', linewidth=2)
axes[1].set_title('Validation PLCC vs Epochs', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('PLCC', fontsize=12)

# 3. Validation RMSE vs Epochs
axes[2].plot(df['epoch'], df['mos_rmse'], color='red', linewidth=2)
axes[2].set_title('Validation RMSE vs Epochs', fontsize=14)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('RMSE', fontsize=12)

plt.tight_layout()

# Save the figure to the artifacts directory
save_path = '/home/user3/.gemini/antigravity/brain/ba4c84ef-a194-42bc-8bb6-cabbc1920133/mos_metrics_10k.png'
plt.savefig(save_path, dpi=300)
print(f"Plot saved successfully to {save_path}")
