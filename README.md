# X3D-S Multi-Task Video Quality Assessment Pipeline

Complete multi-task video quality assessment (VQA) pipeline leveraging an X3D-S backbone. The actual functioning model predicts a continuous MOS score, 5 quality classes, and 7 binary artifact flags.

## 📁 Project Structure

```
/home/drive3/low_encoder/
├── x3d_s_backbone.py                  # X3D-S feature extraction backbone
├── extract_features.py                # Single-layer feature extraction script
├── extract_all_layer_features.py      # Multi-layer feature extraction script
├── multitask_vqa_model.py             # Previous VQA model iteration
├── multitask_vqa_model_v2.py          # Previous VQA model iteration (v2)
├── multitask_vqa_final.py             # Main multi-task training script with Focal Loss
├── X3D_S_MULTI_TASK_VQA_PIPELINE.md   # Detailed model architecture documentation
├── features/                          # Directory for extracted X3D-S embeddings
└── results/                           # Directory containing model checkpoints and metrics
```

## 🚀 Quick Start Pipeline

### 1. Feature Extraction

Features must be extracted utilizing the custom X3D-S backbone prior to training. The pipeline supports extracting all layers (layer0 through layer4).

```bash
python3 extract_all_layer_features.py
```
This script reads the `/home/drive3/VQA_Demo` dataset, passes videos through `x3d_s_backbone.py` (which is kept completely frozen and untrained/randomly initialized), and saves embeddings in the `features/` directory.

### 2. Model Training

The downstream Multi-Task MLP takes the extracted features and learns to predict all specified VQA quality tasks. The final script utilizes **Focal Loss** to handle extreme class imbalances for rare artifact detections (like Physics Violations).

```bash
python3 multitask_vqa_final.py
```

Results, including the `best_model.pt` checkpoint and CSV metrics (such as `train_metrics_detailed.csv`, `val_metrics_detailed.csv`, and `test_metrics_detailed.csv`), are automatically saved iteratively to the `results/` directory.

## 📊 Pipeline Characteristics

### Dataset Path Requirements
- Requires base video data properly structured under `/home/drive3/VQA_Demo`.
- Metadata CSVs should be contained in `/home/drive3/VQA_Demo/metadata/`.
- Video files directly accessed from `/home/drive3/VQA_Demo/standardized_videos/`.

### Architecture Highlights
- **Backbone**: X3D-S (Frozen)
- **Shared MLP Trunk**: 192 (or layer dependent) → 512 → 256. Includes LayerNorm and Dropout(0.4).
- **Task Specific Heads**:
  - continuous MOS regression head.
  - 5-class Quality classification head.
  - Multi-label Shared Artifact head predicting: Spatial, Hallucination, Lighting, Rendering, Physics Violation, Motion Inconsistency, and Object Flicker.

For more granular details regarding layer dims, batch scaling, and per-artifact focal loss parameters, consult the specific implementation within `multitask_vqa_final.py`.
