import re

with open('/home/drive3/low_encoder/multitask_vqa_final.py', 'r') as f:
    content = f.read()

# Add argparse import if not present
if "import argparse" not in content:
    content = content.replace("import torch\nimport torch.nn", "import torch\nimport torch.nn\nimport argparse")

# Update Config class defaults
config_old = """    FEATURES_DIR = "/home/drive3/low_encoder/features_10k"
    OUTPUT_DIR = "/home/drive3/low_encoder/results_10k\""""
config_new = """    VARIANT = 's'
    FEATURES_DIR = "/home/drive3/low_encoder/features_10k_s"
    OUTPUT_DIR = "/home/drive3/low_encoder/results_10k_s\""""
content = content.replace(config_old, config_new)

# Update main block
main_old = """if __name__ == '__main__':
    # Initialize configuration
    config = Config()"""
main_new = """if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='s', choices=['xs', 's', 'm'], help='X3D variant to train on')
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.VARIANT = args.variant
    config.FEATURES_DIR = f"/home/drive3/low_encoder/features_10k_{args.variant}"
    config.OUTPUT_DIR = f"/home/drive3/low_encoder/results_10k_{args.variant}\""""
content = content.replace(main_old, main_new)

with open('/home/drive3/low_encoder/multitask_vqa_final.py', 'w') as f:
    f.write(content)

print("multitask_vqa_final.py rewritten successfully!")
