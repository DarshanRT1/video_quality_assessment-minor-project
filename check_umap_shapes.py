import torch
import os
import pandas as pd

for split in ['train', 'test']:
    df = pd.read_csv(f'/home/drive3/VQA_Demo/metadata/{split}.csv')
    count = 0
    layer = 'layer4'
    base_dir = '/home/drive3/low_encoder/features'
    for idx, row in df.iterrows():
        filename = row['filename']
        feat_path = os.path.join(base_dir, split, layer, f"{filename.replace('.mp4', '_features.pt')}")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(base_dir, split, layer, f"{filename.replace('.mp4', '_standardized_features.pt')}")
            if not os.path.exists(feat_path):
                feat_path = os.path.join(base_dir, split, layer, f"{filename.replace('.mp4', '_224p_features.pt')}")
        if os.path.exists(feat_path):
            count += 1
    print(f"Split: {split}")
    print(f"Total videos in {split}.csv: {len(df)}")
    print(f"Total .pt feature files found for {split}: {count}")
    print(f"The UMAP NumPy Array shape for this split is: [{count}, 192]")
    print("-" * 50)
