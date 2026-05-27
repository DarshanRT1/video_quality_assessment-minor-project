import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from tqdm import tqdm

def load_data(split):
    df = pd.read_csv(f'/home/drive3/VQA_Demo/metadata/{split}.csv')
    return df

def generate_layer_umaps():
    splits = ['train', 'test']
    layers = ['layer4']
    base_dir = '/home/drive3/low_encoder/features'
    
    # 9 Features to plot
    targets = [
        'mos', 'quality_class', 'spatial_flag', 'hallucination_flag',
        'lighting_flag', 'rendering_flag', 'physics_violation_flag',
        'motion_inconsistency_flag', 'object_flicker_flag'
    ]
    
    print("Preparing to generate UMAP embeddings. This may take a while.")
    os.makedirs('/home/drive3/low_encoder/results/umaps', exist_ok=True)
    
    for layer in layers:
        print(f"\nProcessing {layer}...")
        
        for split in splits:
            print(f"  Loading {split} split...")
            df = load_data(split)
            
            # Load embeddings
            features = []
            labels = {t: [] for t in targets}
            valid_indices = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    Reading {layer} tensors"):
                filename = row['filename']
                feat_path = os.path.join(base_dir, split, layer, f"{filename.replace('.mp4', '_features.pt')}")
                
                # Check for standardized naming
                if not os.path.exists(feat_path):
                    feat_path = os.path.join(base_dir, split, layer, f"{filename.replace('.mp4', '_standardized_features.pt')}")
                    if not os.path.exists(feat_path):
                        feat_path = os.path.join(base_dir, split, layer, f"{filename.replace('.mp4', '_224p_features.pt')}")
                
                if os.path.exists(feat_path):
                    feat = torch.load(feat_path)
                    # Use Global Average Pooling across (time, height, width) to match Model Datasets
                    feat = feat.mean(dim=(1, 2, 3)).numpy()
                    features.append(feat)
                    for t in targets:
                        labels[t].append(row[t])
                    valid_indices.append(idx)
            
            if len(features) == 0:
                print(f"    No features found for {layer} {split}")
                continue
                
            X = np.stack(features)
            
            # Reduce with UMAP
            print(f"    Calculating UMAP for {X.shape}...")
            reducer = umap.UMAP(random_state=42)
            embedding = reducer.fit_transform(X)
            
            # Plot the 9 targets in a 3x3 grid
            fig, axes = plt.subplots(3, 3, figsize=(20, 20))
            fig.suptitle(f'UMAP - {layer.capitalize()} - {split.capitalize()} Split', fontsize=20, y=1.02)
            axes = axes.flatten()
            
            for i, target in enumerate(targets):
                ax = axes[i]
                y = np.array(labels[target])
                
                if target == 'mos':
                    # Continuous variable -> colormap
                    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)
                    plt.colorbar(scatter, ax=ax, label=target.upper())
                else:
                    # Categorical / Binary variables
                    unique_classes = np.unique(y)
                    for cls in unique_classes:
                        mask = (y == cls)
                        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=f'Class {cls}', s=10, alpha=0.7)
                    ax.legend()
                    
                ax.set_title(f"Target: {target}")
                ax.set_xticks([])
                ax.set_yticks([])
                
            plt.tight_layout()
            out_path = f'/home/drive3/low_encoder/results/umaps/umap_{layer}_{split}.png'
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"    Saved grid to {out_path}")

if __name__ == '__main__':
    generate_layer_umaps()
