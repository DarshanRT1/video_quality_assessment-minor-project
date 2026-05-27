import torch
import torch.nn as nn
import time
from x3d_backbone import create_x3d_backbone
from multitask_vqa_final import MultiTaskVQA
from torch.utils.flop_counter import FlopCounterMode

class EndToEndVQA(nn.Module):
    def __init__(self, variant='s'):
        super().__init__()
        self.backbone = create_x3d_backbone(variant=variant)
        # Force pretrained to False to prevent downloads
        self.backbone.blocks[0] # dummy test

        self.mlp = MultiTaskVQA(feature_dim=192, hidden_dim=512, hidden_dim2=256, dropout=0.0)
        
    def forward(self, x):
        features = self.backbone(x)
        # Global Average Pooling as done in dataset __getitem__
        features = features.mean(dim=(2, 3, 4))
        # Pass to MLP
        output = self.mlp(features)
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_variant(variant, clip_length, crop_size):
    print(f"\n{'='*40}")
    print(f"End-to-End Benchmarking X3D-{variant.upper()} + MLP")
    print(f"Input: {clip_length} frames, {crop_size}x{crop_size}")
    print(f"{'='*40}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Instantiate model
    model = EndToEndVQA(variant=variant).to(device)
    model.eval()
    
    # Count parameters
    params = count_parameters(model)
    print(f"Exact Parameters: {params:,}")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, clip_length, crop_size, crop_size).to(device)
    
    # Calculate GFLOPs
    with torch.no_grad():
        with FlopCounterMode(model, display=False) as flop_counter:
            model(dummy_input)
        
        flops = flop_counter.get_total_flops()
        print(f"Exact GFLOPs: {flops / 1e9:.6f}")
    
    # Benchmark Latency
    print("\nRunning latency benchmark...")
    with torch.no_grad():
        # Warmup
        for _ in range(50):
            model(dummy_input)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.time()
        num_iters = 100
        
        for _ in range(num_iters):
            model(dummy_input)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.time()
        
    avg_latency = ((end_time - start_time) / num_iters) * 1000 # in ms
    print(f"Exact Inference Latency (Batch Size 1): {avg_latency:.4f} ms")


if __name__ == "__main__":
    benchmark_variant('xs', 4, 160)
    benchmark_variant('s', 13, 160)
    benchmark_variant('m', 16, 256)
