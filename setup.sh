#!/bin/bash

# Setup script for Stage-0 VQA Pipeline

echo "Setting up Stage-0 VQA Pipeline..."

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "Creating directory structure..."
mkdir -p results/features/train
mkdir -p results/features/val
mkdir -p results/features/test
mkdir -p results/umap
mkdir -p results/models

echo "Setup complete!"
echo "Directory structure:"
tree results/ 2>/dev/null || find results/ -type d
