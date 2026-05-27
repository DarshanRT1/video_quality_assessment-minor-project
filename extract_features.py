"""
Feature Extraction Script using X3D-S Backbone

This script extracts features from videos in the VQA_Demo dataset using
the X3D-S model backbone (without MLP classification head).
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from typing import Tuple, Optional
from tqdm import tqdm

from x3d_s_backbone import X3D_S_Backbone


class VideoFeatureExtractor:
    """Extract features from videos using X3D-S backbone."""
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clip_length: int = 13,
        crop_size: int = 160
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model: X3D-S backbone model (if None, creates a new one)
            device: Device to run inference on
            clip_length: Number of frames to sample from each video
            crop_size: Spatial resolution to resize frames to
        """
        self.device = device
        self.clip_length = clip_length
        self.crop_size = crop_size
        
        if model is None:
            self.model = X3D_S_Backbone()
        else:
            self.model = model
            
        self.model = self.model.to(device)
        self.model.eval()
        
    def load_video(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Load a video file and extract frames.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tensor of shape [clip_length, C, H, W] or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly across the video
        if total_frames >= self.clip_length:
            indices = np.linspace(0, total_frames - 1, self.clip_length, dtype=int)
        else:
            # Repeat frames if video is shorter than clip_length
            indices = np.array([i % total_frames for i in range(self.clip_length)])
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.crop_size, self.crop_size))
                frames.append(frame)
            else:
                # If frame read fails, use black frame
                frames.append(np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8))
        
        cap.release()
        
        # Convert to tensor [T, H, W, C] -> [C, T, H, W]
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        
        # Normalize
        frames = frames / 255.0
        
        return frames
    
    def extract_features(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Extract features from a single video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Feature tensor of shape [1, 192, 13, 5, 5] or None if failed
        """
        frames = self.load_video(video_path)
        
        if frames is None:
            return None
            
        # Add batch dimension and move to device
        frames = frames.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(frames)
            
        return features
    
    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        video_dir: str,
        filename_col: str = "filename",
        output_dir: Optional[str] = None
    ) -> dict:
        """
        Extract features from a dataframe of video filenames.
        
        Args:
            df: DataFrame containing video filenames
            video_dir: Directory containing video files
            filename_col: Column name for video filenames
            output_dir: Optional directory to save features
            
        Returns:
            Dictionary mapping filenames to feature tensors
        """
        features_dict = {}
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            filename = row[filename_col]
            
            # Try different possible video paths (prioritize standardized_videos)
            possible_paths = [
                os.path.join(video_dir, "standardized_videos", filename),
                os.path.join(video_dir, "standardized_videos_lowres", filename.replace(".mp4", "_224p.mp4")),
                os.path.join(video_dir, filename),
            ]
            
            video_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    video_path = path
                    break
            
            if video_path is None:
                print(f"Warning: Could not find video for {filename}")
                continue
                
            features = self.extract_features(video_path)
            
            if features is not None:
                features_dict[filename] = features.cpu()
                
                # Optionally save features
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    feature_path = os.path.join(output_dir, f"{filename.replace('.mp4', '_features.pt')}")
                    torch.save(features.cpu(), feature_path)
        
        return features_dict


def main():
    """Main function to extract features from VQA dataset."""
    
    # Configuration
    VIDEO_DIR = "/home/drive3/VQA_Demo"
    METADATA_DIR = os.path.join(VIDEO_DIR, "metadata")
    OUTPUT_DIR = "/home/drive3/low_encoder/features"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create the X3D-S backbone model
    print("Loading X3D-S backbone model...")
    model = X3D_S_Backbone()
    model = model.to(device)
    model.eval()
    
    # Create feature extractor
    extractor = VideoFeatureExtractor(model=model, device=device)
    
    # Load metadata
    print("\nLoading metadata...")
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Extract features from a small sample for testing
    print("\nExtracting features from sample videos...")
    sample_df = train_df.head(5)
    
    features_dict = extractor.extract_from_dataframe(
        sample_df,
        VIDEO_DIR,
        filename_col="filename",
        output_dir=OUTPUT_DIR
    )
    
    print(f"\nExtracted features from {len(features_dict)} videos")
    
    # Print feature shapes
    for filename, features in list(features_dict.items())[:3]:
        print(f"  {filename}: {features.shape}")
    
    print(f"\nFeatures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
