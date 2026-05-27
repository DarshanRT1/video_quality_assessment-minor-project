#!/bin/bash

# Python Dependencies Installation Script
# This script provides instructions and commands for installing required packages

echo "========================================================================"
echo "Python Environment Setup for Stage-0 VQA Pipeline"
echo "========================================================================"
echo ""

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    echo "⚠️  pip is NOT installed with python3"
    echo ""
    echo "To install pip, run ONE of the following:"
    echo ""
    echo "Option 1 - Using apt (requires sudo):"
    echo "  sudo apt update"
    echo "  sudo apt install python3-pip"
    echo ""
    echo "Option 2 - Using get-pip.py:"
    echo "  wget https://bootstrap.pypa.io/get-pip.py"
    echo "  python3 get-pip.py --user"
    echo ""
    echo "After installing pip, run this script again."
    echo "========================================================================"
    exit 1
fi

echo "✓ pip is available"
echo ""
echo "Installing Python packages..."
echo "This may take 10-30 minutes depending on your system."
echo ""

# Install packages
python3 -m pip install --user \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    pytorchvideo \
    pandas \
    numpy \
    scikit-learn \
    opencv-python \
    umap-learn \
    matplotlib \
    seaborn \
    scipy \
    tqdm

# Check installation
echo ""
echo "========================================================================"
echo "Verifying installation..."
echo "========================================================================"

python3 -c "
import torch
import torchvision
import pandas
import numpy
import cv2
print('✓ All core packages imported successfully')
print(f'  PyTorch: {torch.__version__}')
print(f'  NumPy: {numpy.__version__}')
print(f'  Pandas: {pandas.__version__}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Installation Complete!"
    echo "You can now run the pipeline:"
    echo "  python3 run_pipeline.py"
    echo "========================================================================"
else
    echo ""
    echo "⚠️  Some packages failed to import. Please check the error messages above."
fi
