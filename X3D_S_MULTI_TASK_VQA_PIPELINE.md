# X3D-S Multi-Task VQA Pipeline Configuration

## Overview

This pipeline implements a multi-task learning model for Video Quality Assessment (VQA) using X3D-S as the backbone feature extractor. The model predicts:
- **MOS (Mean Opinion Score)** regression
- **Quality classification** (5 classes: Bad, Poor, Fair, Good, Excellent)
- **7 Binary artifact classifications**

---

## Model Architecture

### Feature Extractor
- **Model**: X3D-S (from PyTorchVideo)
- **Feature Layer**: layer4 (192 channels)
- **Pooling**: Global average pooling over temporal and spatial dimensions
- **Output Dimension**: 192

### Multi-Task MLP

```
Input (192)
    ↓
Shared Trunk (MLP)
    ├─ Linear(192 → 512) + LayerNorm + ReLU + Dropout(0.4)
    ├─ Linear(512 → 256) + LayerNorm + ReLU + Dropout(0.4)
    ↓
Task-Specific Heads
    ├─ MOS Head: 256 → 128 → 1 (regression)
    ├─ Quality Head: 256 → 128 → 5 (classification)
    └─ Shared Artifact Head: 256 → 256 → 128 → 2 (binary) → shared by all 7 artifacts
```

**Total Parameters**: ~530,000

### Key Architecture Decision
- **Shared artifact head** used for all 7 artifact types to prevent class collapse
- Each artifact uses the same head weights but receives gradients from its own Focal Loss

---

## Configuration

### Training Parameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 200 (with early stopping at 30) |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Dropout | 0.4 |
| Early Stopping Patience | 30 |

### Optimizer & Scheduler
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=30, T_mult=2, eta_min=1e-6)
- **Gradient Clipping**: max_norm=1.0

### Loss Function

#### Focal Loss for Artifact Detection
Focal Loss is used to handle severe class imbalance in artifact detection:
- **Gamma**: 2.0-3.0 (focuses on hard examples)
- **Class Weights**: Aggressive weights based on dataset distribution

| Artifact | Class 0 (%) | Class 1 (%) | Alpha (weight for 1) | Gamma |
|----------|-------------|------------|---------------------|-------|
| Spatial | 50.5 | 49.5 | [1.0, 1.0] | 2.0 |
| Hallucination | 53.3 | 46.7 | [1.0, 1.0] | 2.0 |
| Lighting | 75.2 | 24.8 | [0.25, 0.75] | 2.5 |
| Rendering | 70.6 | 29.4 | [0.30, 0.70] | 2.5 |
| Physics | 90.6 | 9.4 | [0.10, 0.90] | 3.0 |
| Motion | 79.8 | 20.2 | [0.20, 0.80] | 2.5 |
| Flicker | 81.6 | 18.4 | [0.15, 0.85] | 2.5 |

#### Loss Weights
| Task | Weight |
|------|--------|
| MOS | 1.5 |
| Quality | 1.0 |
| Spatial | 1.0 |
| Hallucination | 1.0 |
| Lighting | 2.0 |
| Rendering | 2.5 |
| Physics | 3.0 |
| Motion | 2.5 |
| Flicker | 1.5 |

---

## Dataset

### Data Split
| Split | Samples |
|-------|---------|
| Train | 1,398 |
| Val | 297 |
| Test | 305 |
| **Total** | **2,000** |

### Feature Files Location
```
/home/drive3/low_encoder/features/
├── train/
│   ├── layer0/  (1,398 files)
│   ├── layer1/  (1,398 files)
│   ├── layer2/  (1,398 files)
│   ├── layer3/  (1,398 files)
│   └── layer4/  (1,398 files)
├── val/
│   └── ...
└── test/
    └── ...
```

### Metadata Location
```
/home/drive3/VQA_Demo/metadata/
├── train.csv
├── val.csv
└── test.csv
```

---

## Output Results

### Best Model: Epoch 90

#### Validation Performance
| Metric | Value |
|--------|-------|
| Loss | 251.87 |
| MOS RMSE | 12.85 |
| MOS PLCC | 0.493 |
| MOS SRCC | 0.472 |

#### Test Performance

##### MOS Regression
| Metric | Value |
|--------|-------|
| RMSE | 13.93 |
| PLCC | 0.406 |
| SRCC | 0.396 |

##### Quality Classification
| Metric | Value |
|--------|-------|
| Accuracy | 48.5% |

##### Artifact Detection
| Artifact | Accuracy | F1 Score |
|----------|----------|----------|
| Spatial | 40.0% | 0.399 |
| Hallucination | 62.0% | 0.617 |
| Lighting | 62.0% | 0.583 |
| Rendering | 50.2% | 0.462 |
| Physics | 56.1% | 0.437 |
| Motion | 58.0% | 0.529 |
| Flicker | 65.6% | 0.602 |

---

## Key Improvements (Focal Loss)

### Problem Solved
- **Before**: Model collapsed to single-class predictions (e.g., physics always predicted 0)
- **After**: Focal Loss enables diverse predictions for all artifacts

### Results Comparison
| Artifact | Before (F1) | After (F1) | Improvement |
|----------|-------------|------------|-------------|
| Hallucination | 0.316 | 0.617 | +95% |
| Lighting | 0.493 | 0.583 | +18% |
| Flicker | 0.629 | 0.602 | -4% |
| Physics | 0.479 | 0.437 | -9% |

---

## Files

### Core Scripts
| File | Description |
|------|-------------|
| `x3d_s_backbone.py` | X3D-S model definition |
| `extract_features.py` | Feature extraction script |
| `extract_all_layer_features.py` | Multi-layer feature extraction |
| `multitask_vqa_final.py` | Main training script with Focal Loss |

### Output Files
```
/home/drive3/low_encoder/results/
├── best_model.pt              # Best model checkpoint (epoch 90)
├── train_metrics_detailed.csv
├── val_metrics_detailed.csv
├── val_metrics_important.csv  # 8 key epochs
├── test_metrics_detailed.csv
├── validation_metrics.png     # SRCC, PLCC, RMSE plots
└── artifact_accuracy_histogram.png
```

---

## Usage

### 1. Extract Features
```bash
python3 extract_features.py
```

### 2. Train Model
```bash
python3 multitask_vqa_final.py
```

### 3. Evaluate
Results are automatically saved to `results/` directory.

---

## Notes

1. **Feature Dimension**: Uses layer4 features (192 channels) after global average pooling
2. **Focal Loss**: Handles severe class imbalance (physics has only 8% positive class)
3. **Shared Artifact Head**: Prevents individual heads from collapsing to single classes
4. **Early Stopping**: Monitors validation loss with patience of 30 epochs
5. **Best Model Selection**: Based on lowest validation loss (epoch 90)
