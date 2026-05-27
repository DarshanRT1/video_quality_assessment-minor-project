import re

with open('/home/drive3/low_encoder/extract_10k_features.py', 'r') as f:
    content = f.read()

# Replace import
content = content.replace("from x3d_s_backbone import X3D_S_Backbone, X3D_S_MultiLayerFeatureExtractor", 
                          "from x3d_backbone import X3D_Backbone, X3D_MultiLayerFeatureExtractor\nimport argparse")

# Modify extract_10k to accept variant
def_old = "def extract_10k():"
def_new = "def extract_10k(variant='s'):"
content = content.replace(def_old, def_new)

# Update print statements
print_old = """    print("=" * 60)
    print("FEATURE EXTRACTION: X3D-S Kinetics-400 (Layer 4 only) on 10K Dataset")
    print("=" * 60)"""
print_new = """    print("=" * 60)
    print(f"FEATURE EXTRACTION: X3D-{variant.upper()} Kinetics-400 (Layer 4 only) on 10K Dataset")
    print("=" * 60)"""
content = content.replace(print_old, print_new)

# Update output directory dynamically
out_old = """    output_dir = "/home/drive3/low_encoder/features_10k\""""
out_new = """    output_dir = f"/home/drive3/low_encoder/features_10k_{variant}\""""
content = content.replace(out_old, out_new)

# Instantiate new backbone and extractor
inst_old = """    model = X3D_S_Backbone().to(device)
    model.eval()
    
    extractor = X3D_S_MultiLayerFeatureExtractor(
        model=model,
        device=device,
        clip_length=13,
        crop_size=160
    )"""
inst_new = """    model = X3D_Backbone(variant=variant).to(device)
    model.eval()
    
    extractor = X3D_MultiLayerFeatureExtractor(
        variant=variant,
        model=model,
        device=device
    )"""
content = content.replace(inst_old, inst_new)

# Update main block
main_old = """if __name__ == '__main__':
    extract_10k()"""
main_new = """if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='s', choices=['xs', 's', 'm'], help='X3D variant to extract features from')
    args = parser.parse_args()
    extract_10k(args.variant)"""
content = content.replace(main_old, main_new)

with open('/home/drive3/low_encoder/extract_10k_features.py', 'w') as f:
    f.write(content)

print("extract_10k_features.py rewritten successfully!")
