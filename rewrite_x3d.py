import re

with open('/home/drive3/low_encoder/x3d_backbone.py', 'r') as f:
    content = f.read()

# Replace imports
content = content.replace("from pytorchvideo.models.hub import x3d_s", "from pytorchvideo.models.hub import x3d_xs, x3d_s, x3d_m")

# Replace class names
content = content.replace("X3D_S_Backbone", "X3D_Backbone")
content = content.replace("X3D_S_MultiLayerFeatureExtractor", "X3D_MultiLayerFeatureExtractor")
content = content.replace("create_x3d_s_backbone", "create_x3d_backbone")

# Update X3D_Backbone __init__
init_old = """    def __init__(self, original_model=None, pretrained=True):
        super().__init__()
        
        if original_model is None:
            # Create a pretrained x3d-s model from PyTorchVideo hub
            original_model = x3d_s(pretrained=pretrained)"""

init_new = """    def __init__(self, variant='s', original_model=None, pretrained=True):
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
                raise ValueError(f"Unknown variant {variant}")"""
content = content.replace(init_old, init_new)

# Update Extractor __init__
extractor_init_old = """    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clip_length: int = 13,
        crop_size: int = 160
    ):
        \"\"\"
        Initialize the feature extractor.
        
        Args:
            model: X3D-S backbone model (if None, creates a new one)
            device: Device to run inference on
            clip_length: Number of frames to sample from each video
            crop_size: Spatial resolution to resize frames to
        \"\"\"
        self.device = device
        self.clip_length = clip_length
        self.crop_size = crop_size
        
        if model is None:
            self.model = X3D_Backbone()"""

extractor_init_new = """    def __init__(
        self,
        variant: str = 's',
        model: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clip_length: Optional[int] = None,
        crop_size: Optional[int] = None
    ):
        \"\"\"
        Initialize the feature extractor.
        \"\"\"
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
            self.model = X3D_Backbone(variant=variant)"""
content = content.replace(extractor_init_old, extractor_init_new)

# Update create_x3d_backbone
create_old = """def create_x3d_backbone() -> X3D_Backbone:
    \"\"\"
    Create an X3D-S backbone model without the MLP classification head.
    
    Returns:
        X3D_Backbone: The modified model with only the feature extraction backbone
    \"\"\"
    return X3D_Backbone()"""

create_new = """def create_x3d_backbone(variant='s') -> X3D_Backbone:
    \"\"\"
    Create an X3D backbone model without the MLP classification head.
    \"\"\"
    return X3D_Backbone(variant=variant)"""
content = content.replace(create_old, create_new)

with open('/home/drive3/low_encoder/x3d_backbone.py', 'w') as f:
    f.write(content)

print("x3d_backbone.py rewritten successfully!")
