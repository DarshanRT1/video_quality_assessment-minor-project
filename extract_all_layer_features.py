"""
Feature Extraction Script - All Layers from X3D-S Backbone

This script extracts features from ALL layers of the X3D-S backbone model
for videos in the VQA dataset and saves them in separate folders for each split.

Features are saved in the following structure:
    features/
        train/
            layer0/
            layer1/
            layer2/
            layer3/
            layer4/
        val/
            (same structure)
        test/
            (same structure)
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import json

# Add current directory to path
sys.path.insert(0, '/home/drive3/low_encoder')

from x3d_s_backbone import X3D_S_Backbone, X3D_S_MultiLayerFeatureExtractor


# Configuration
VIDEO_DIR = "/home/drive3/VQA_Demo"
METADATA_DIR = os.path.join(VIDEO_DIR, "metadata")
VIDEO_SRC_DIR = os.path.join(VIDEO_DIR, "standardized_videos")
OUTPUT_BASE_DIR = "/home/drive3/low_encoder/features"

# Model configuration
CLIP_LENGTH = 13
CROP_SIZE = 160

# Layer names
LAYER_NAMES = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
LAYER_CHANNELS = {
    'layer0': 24,   # Stem output
    'layer1': 24,   # Stage 1 output
    'layer2': 48,   # Stage 2 output
    'layer3': 96,   # Stage 3 output
    'layer4': 192,  # Stage 4 output (final backbone)
}


def create_feature_extractor(device: str = "cuda"):
    """Create and initialize the feature extractor."""
    print(f"Using device: {device}")
    
    # Create model
    model = X3D_S_Backbone()
    model = model.to(device)
    model.eval()
    
    # Create extractor
    extractor = X3D_S_MultiLayerFeatureExtractor(
        model=model,
        device=device,
        clip_length=CLIP_LENGTH,
        crop_size=CROP_SIZE
    )
    
    return extractor


def get_video_path(filename: str) -> str:
    """Get the full path to a video file."""
    # The metadata filenames already include _standardized suffix
    video_path = os.path.join(VIDEO_SRC_DIR, filename)
    
    if os.path.exists(video_path):
        return video_path
    
    return None


def save_features(features: dict, output_dir: str, video_name: str):
    """Save features for each layer separately."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove .mp4 extension from video name
    base_name = video_name.replace(".mp4", "")
    
    # Save each layer's features
    for layer_name in LAYER_NAMES:
        if layer_name in features:
            layer_dir = os.path.join(output_dir, layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            
            feature_path = os.path.join(layer_dir, f"{base_name}_features.pt")
            # Save as PyTorch tensor
            torch.save(features[layer_name], feature_path)


def process_split(split_name: str, df: pd.DataFrame, extractor, max_videos: int = None):
    """Process all videos in a data split."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"Total videos in metadata: {len(df)}")
    
    # Output directory for this split
    split_output_dir = os.path.join(OUTPUT_BASE_DIR, split_name)
    
    # Count successful extractions
    success_count = 0
    fail_count = 0
    missing_count = 0
    
    # Process each video
    videos_to_process = df['filename'].tolist()
    if max_videos:
        videos_to_process = videos_to_process[:max_videos]
    
    print(f"Processing {len(videos_to_process)} videos...")
    
    for filename in tqdm(videos_to_process, desc=f"Extracting {split_name}"):
        # Get video path
        video_path = get_video_path(filename)
        
        if video_path is None:
            print(f"Warning: Video not found for {filename}")
            missing_count += 1
            continue
        
        # Extract features from all layers
        features = extractor.extract_all_layers(video_path)
        
        if features is None:
            print(f"Warning: Failed to extract features from {filename}")
            fail_count += 1
            continue
        
        # Save features
        save_features(features, split_output_dir, filename)
        success_count += 1
    
    print(f"\n{split_name.upper()} Results:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Missing: {missing_count}")
    
    return success_count, fail_count, missing_count


def save_feature_info(output_dir: str):
    """Save information about the extracted features."""
    info = {
        "model": "X3D-S Backbone (without MLP head)",
        "clip_length": CLIP_LENGTH,
        "crop_size": CROP_SIZE,
        "layers": LAYER_NAMES,
        "layer_channels": LAYER_CHANNELS,
        "layer_spatial_dims": {
            "layer0": "[13, 80, 80]",
            "layer1": "[13, 40, 40]", 
            "layer2": "[13, 20, 20]",
            "layer3": "[13, 10, 10]",
            "layer4": "[13, 5, 5]"
        },
        "flattened_features": {
            "layer0": 24 * 13 * 80 * 80,
            "layer1": 24 * 13 * 40 * 40,
            "layer2": 48 * 13 * 20 * 20,
            "layer3": 96 * 13 * 10 * 10,
            "layer4": 192 * 13 * 5 * 5
        }
    }
    
    info_path = os.path.join(output_dir, "feature_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nFeature info saved to: {info_path}")


def main():
    """Main function to extract features from all splits."""
    
    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create feature extractor
    extractor = create_feature_extractor(device)
    
    # Load metadata
    print("\nLoading metadata...")
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Save feature information
    save_feature_info(OUTPUT_BASE_DIR)
    
    # Process each split
    # Process train split
    train_success, train_fail, train_missing = process_split("train", train_df, extractor)
    
    # Process val split  
    val_success, val_fail, val_missing = process_split("val", val_df, extractor)
    
    # Process test split
    test_success, test_fail, test_missing = process_split("test", test_df, extractor)
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"\nTotal videos processed:")
    print(f"  Train: {train_success} (failed: {train_fail}, missing: {train_missing})")
    print(f"  Val:   {val_success} (failed: {val_fail}, missing: {val_missing})")
    print(f"  Test:  {test_success} (failed: {test_fail}, missing: {test_missing})")
    print(f"\nFeature dimensions per layer:")
    for layer in LAYER_NAMES:
        print(f"  {layer}: {LAYER_CHANNELS[layer]} channels")


if __name__ == "__main__":
    main()
