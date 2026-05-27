import torch
import time
from x3d_backbone import create_x3d_backbone
from torch.utils.flop_counter import FlopCounterMode

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_variant(variant, clip_length, crop_size):
    print(f"\n{'='*40}")
    print(f"Benchmarking X3D-{variant.upper()}")
    print(f"Input: {clip_length} frames, {crop_size}x{crop_size}")
    print(f"{'='*40}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Instantiate model
    model = create_x3d_backbone(variant=variant).to(device)
    model.eval()
    
    # Count parameters
    params = count_parameters(model)
    print(f"Parameters: {params / 1e6:.2f} M")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, clip_length, crop_size, crop_size).to(device)
    
    # Calculate GFLOPs
    with torch.no_grad():
        with FlopCounterMode(model, display=False) as flop_counter:
            model(dummy_input)
        
        flops = flop_counter.get_total_flops()
        print(f"GFLOPs: {flops / 1e9:.3f}")
    
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
    print(f"Average Inference Latency (Batch Size 1): {avg_latency:.2f} ms")


if __name__ == "__main__":
    benchmark_variant('xs', 4, 160)
    benchmark_variant('s', 13, 160)
    benchmark_variant('m', 16, 256)
