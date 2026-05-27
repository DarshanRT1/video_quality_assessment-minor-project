import pandas as pd
import os

variants = ['xs', 's', 'm']
headers = ['Encoder', 'Split', 'SRCC', 'PLCC', 'Spatial(AUC/F1)', 'Lighting(AUC/F1)', 'Rendering(AUC/F1)', 'Hallucination(AUC/F1)', 'Physics(AUC/F1)', 'Flicker(AUC/F1)', 'Motion(AUC/F1)']

# Output table header
print("| " + " | ".join(headers) + " |")
print("| " + " | ".join(["---"] * len(headers)) + " |")

artifacts = ['spatial', 'lighting', 'rendering', 'hallucination', 'physics', 'flicker', 'motion']

for var in variants:
    results_dir = f"/home/drive3/low_encoder/results_10k_{var}"
    
    # Check if files exist
    test_csv = os.path.join(results_dir, "test_metrics_detailed.csv")
    val_csv = os.path.join(results_dir, "val_metrics_detailed.csv")
    
    if not os.path.exists(test_csv) or not os.path.exists(val_csv):
        continue
        
    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv)
    
    best_epoch = test_df.iloc[0]['epoch']
    
    # Get the best val row (epoch is 1-indexed in CSV)
    val_row = val_df[val_df['epoch'] == best_epoch].iloc[0]
    test_row = test_df.iloc[0]
    
    # Format Val Row
    val_data = [f"X3D-{var.upper()}", "Val", f"{val_row['mos_srcc']:.4f}", f"{val_row['mos_plcc']:.4f}"]
    for art in artifacts:
        auc = val_row.get(f'{art}_auc', 0.5)
        f1 = val_row[f'{art}_flag_f1']
        val_data.append(f"{auc:.4f} / {f1:.4f}")
    
    print("| " + " | ".join(val_data) + " |")
    
    # Format Test Row
    test_data = [f"X3D-{var.upper()}", "Test", f"{test_row['mos_srcc']:.4f}", f"{test_row['mos_plcc']:.4f}"]
    for art in artifacts:
        auc = test_row.get(f'{art}_auc', 0.5)
        f1 = test_row[f'{art}_flag_f1']
        test_data.append(f"{auc:.4f} / {f1:.4f}")
        
    print("| " + " | ".join(test_data) + " |")
