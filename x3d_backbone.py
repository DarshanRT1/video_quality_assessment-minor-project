"""
X3D-S Model Backbone with Multi-Layer Feature Extraction

This module imports the x3d-s model from pytorchvideo and provides ability to 
extract features from all intermediate layers, not just the final output.

Original model: https://arxiv.org/abs/2004.04730
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pytorchvideo.models.hub import x3d_xs, x3d_s, x3d_m


class X3D_Backbone(nn.Module):
    """
    X3D-S model backbone without MLP classification head.
    
    This model can be used for feature extraction where you need the 
    intermediate representations from the video encoder.
    
    Architecture:
        - Block 0: ResNetBasicStem (input stem with spatial-temporal convolutions)
        - Block 1: ResStage (Stage 1 - 3 ResBlock units, 24 channels)
        - Block 2: ResStage (Stage 2 - 2 ResBlock units, 48 channels)
        - Block 3: ResStage (Stage 3 - 11 ResBlock units, 96 channels)
        - Block 4: ResStage (Stage 4 - 7 ResBlock units, 192 channels)
    
    Each ResBlock contains:
        - 1x1x1 convolution (dimension reduction)
        - 3x3x3 convolution (spatial-temporal feature extraction)
        - Squeeze-and-Excitation (SE) attention mechanism
        - 1x1x1 convolution (dimension expansion)
    
    Input shape:  [batch, 3, 13, 160, 160] (channels, time, height, width)
    Output shape: [batch, 192, 13, 5, 5] (channels, time, height, width)
    """
    
    def __init__(self, variant='s', original_model=None, pretrained=True):
        super().__init__()
        self.variant = variant
        
        if original_model is None:
            if variant == 'xs':
                original_model = x3d_xs(pretrained=pretrained)
            elif variant == 's':
                original_model = x3d_s(pretrained=pretrained)
            elif variant == 'm':
                original_model = x3d_m(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown variant {variant}")
        
        # Keep only blocks 0-4 (stem + 4 stages), remove the head (block 5)
        self.blocks = nn.ModuleList(list(original_model.blocks)[:5])
        
        # Store output channel information for each layer
        self._layer_channels = [24, 24, 48, 96, 192]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor of shape [batch, channels, time, height, width]
               Expected: [batch, 3, 13, 160, 160]
        
        Returns:
            Output tensor of shape [batch, 192, 13, 5, 5]
        """
        for block in self.blocks:
            x = block(x)
        return x
    
    def forward_with_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning features from all layers.
        
        Args:
            x: Input tensor of shape [batch, channels, time, height, width]
        
        Returns:
            Dictionary with intermediate features from each layer:
            - layer0: stem output (24 channels)
            - layer1: stage1 output (24 channels)  
            - layer2: stage2 output (48 channels)
            - layer3: stage3 output (96 channels)
            - layer4: stage4 output (192 channels) - final backbone output
        """
        features = {}
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            features[f'layer{i}'] = x
            
        return features
    
    @property
    def output_channels(self) -> int:
        """Return output channel dimension."""
        return 192
    
    @property
    def layer_channels(self) -> List[int]:
        """Return output channels for each layer."""
        return self._layer_channels
    
    @property
    def num_layers(self) -> int:
        """Return number of layers in the backbone."""
        return len(self.blocks)


class X3D_MultiLayerFeatureExtractor:
    """
    Feature extractor that returns features from all layers of X3D-S backbone.
    """
    
    def __init__(
        self,
        variant: str = 's',
        model: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clip_length: Optional[int] = None,
        crop_size: Optional[int] = None
    ):
        """
        Initialize the feature extractor.
        """
        self.device = device
        self.variant = variant
        
        # Set defaults based on variant if not explicitly provided
        if clip_length is None or crop_size is None:
            if variant == 'xs':
                self.clip_length = clip_length or 4
                self.crop_size = crop_size or 160
            elif variant == 's':
                self.clip_length = clip_length or 13
                self.crop_size = crop_size or 160
            elif variant == 'm':
                self.clip_length = clip_length or 16
                self.crop_size = crop_size or 256
            else:
                raise ValueError(f"Unknown variant {variant}")
        else:
            self.clip_length = clip_length
            self.crop_size = crop_size
        
        if model is None:
            self.model = X3D_Backbone(variant=variant)
        else:
            self.model = model
            
        self.model = self.model.to(device)
        self.model.eval()
        
    def extract_all_layers(
        self, 
        video_path: str,
        return_final: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract features from all layers of the backbone.
        
        Args:
            video_path: Path to the video file
            return_final: Whether to also return the final layer features
            
        Returns:
            Dictionary of layer features, or None if video cannot be loaded
        """
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly across the video
        if total_frames >= self.clip_length:
            indices = np.linspace(0, total_frames - 1, self.clip_length, dtype=int)
        else:
            indices = np.array([i % total_frames for i in range(self.clip_length)])
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.crop_size, self.crop_size))
                frames.append(frame)
            else:
                frames.append(np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8))
        
        cap.release()
        
        # Convert to tensor
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)
        frames = frames / 255.0
        
        # Add batch dimension and move to device
        frames = frames.unsqueeze(0).to(self.device)
        
        # Extract features from all layers
        with torch.no_grad():
            layer_features = self.model.forward_with_intermediate_features(frames)
        
        # Move to CPU and process
        result = {}
        for layer_name, features in layer_features.items():
            result[layer_name] = features.squeeze(0).cpu()
            
        return result


def create_x3d_backbone(variant='s') -> X3D_Backbone:
    """
    Create an X3D backbone model without the MLP classification head.
    """
    return X3D_Backbone(variant=variant)


# Test the model when run directly
if __name__ == "__main__":
    print("="*80)
    print("X3D-S BACKBONE MODEL - MULTI-LAYER FEATURE EXTRACTION")
    print("="*80)
    print()
    
    # Create the backbone model
    model = create_x3d_backbone()
    
    # Print model info
    print(f"Model type: {model.__class__.__name__}")
    print(f"Number of layers: {model.num_layers}")
    print(f"Layer output channels: {model.layer_channels}")
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 13, 160, 160)
    output = model(dummy_input)
    
    print(f"Input shape:  {list(dummy_input.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print()
    
    # Test multi-layer feature extraction
    print("="*80)
    print("MULTI-LAYER FEATURE EXTRACTION TEST")
    print("="*80)
    
    layer_features = model.forward_with_intermediate_features(dummy_input)
    
    for layer_name, features in layer_features.items():
        print(f"{layer_name}: shape = {list(features.shape)}, channels = {features.shape[0]}")
    
    print()
    
    # Test with actual video
    print("="*80)
    print("TESTING WITH ACTUAL VIDEO")
    print("="*80)
    
    extractor = X3D_MultiLayerFeatureExtractor(model=model)
    
    video_path = "/home/drive3/VQA_Demo/standardized_videos/00037_05_standardized.mp4"
    print(f"Testing with: {video_path}")
    
    features = extractor.extract_all_layers(video_path)
    
    if features:
        print("\nExtracted features from all layers:")
        for layer_name, feat in features.items():
            # Flatten spatial and temporal dimensions
            feat_flat = feat.flatten(start_dim=1)
            print(f"  {layer_name}: shape = {list(feat.shape)}, flattened = {list(feat_flat.shape)}")
