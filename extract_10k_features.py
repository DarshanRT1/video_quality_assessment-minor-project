import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '/home/drive3/low_encoder')
from x3d_backbone import X3D_Backbone, X3D_MultiLayerFeatureExtractor
import argparse

def get_video_path(filename, video_dir):
    # Try direct path
    path = os.path.join(video_dir, filename)
    if os.path.exists(path):
        return path
    
    # Try with .mp4 if not present
    if not filename.endswith('.mp4'):
        path = os.path.join(video_dir, filename + '.mp4')
        if os.path.exists(path):
            return path
            
    # Try standardized name
    standardized = filename.replace('.mp4', '_standardized.mp4')
    path = os.path.join(video_dir, standardized)
    if os.path.exists(path):
        return path
        
    return None

def extract_10k(variant='s'):
    print("=" * 60)
    print(f"FEATURE EXTRACTION: X3D-{variant.upper()} Kinetics-400 (Layer 4 only) on 10K Dataset")
    print("=" * 60)
    
    metadata_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    video_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"
    output_dir = f"/home/drive3/low_encoder/features_10k_{variant}"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = X3D_Backbone(variant=variant).to(device)
    model.eval()
    
    extractor = X3D_MultiLayerFeatureExtractor(
        variant=variant,
        model=model,
        device=device
    )
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        df = pd.read_csv(os.path.join(metadata_dir, f"{split}.csv"))
        
        split_out_dir = os.path.join(output_dir, split, 'layer4')
        os.makedirs(split_out_dir, exist_ok=True)
        
        success = 0
        failed = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
            filename = row['filename']
            base_name = filename.replace('.mp4', '').replace('_standardized', '')
            
            # Check if already extracted
            save_path = os.path.join(split_out_dir, f"{base_name}_features.pt")
            if os.path.exists(save_path):
                success += 1
                continue
                
            video_path = get_video_path(filename, video_dir)
            if video_path is None:
                # also try without _standardized if it is in the filename
                video_path = get_video_path(base_name + '.mp4', video_dir)
                if video_path is None:
                    failed += 1
                    continue
            
            features = extractor.extract_all_layers(video_path)
            
            if features is None or 'layer4' not in features:
                failed += 1
                continue
            
            torch.save(features['layer4'].cpu(), save_path)
            success += 1
            
        print(f"  {split}: {success} succeeded, {failed} failed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='s', choices=['xs', 's', 'm'], help='X3D variant to extract features from')
    args = parser.parse_args()
    extract_10k(args.variant)
