#!/bin/bash
echo "Starting X3D-XS Extraction..."
python3 extract_10k_features.py --variant xs > extraction_xs.log 2>&1
echo "X3D-XS Extraction completed."

echo "Starting X3D-M Extraction..."
python3 extract_10k_features.py --variant m > extraction_m.log 2>&1
echo "X3D-M Extraction completed."
