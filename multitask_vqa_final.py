"""
Multi-Task Learning Model for VQA - Focal Loss + Aggressive Class Weights
==========================================================================
Using Shared MLP Trunk with Focal Loss for artifact detection
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/drive3/low_encoder')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    VIDEO_DIR = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"
    METADATA_DIR = "/home/user3/AI_VQA(10000)/10Ksplits"
    VARIANT = 's'
    FEATURES_DIR = "/home/drive3/low_encoder/features_10k_s"
    OUTPUT_DIR = "/home/drive3/low_encoder/results_10k_s"
    
    FEATURE_DIM = 192
    HIDDEN_DIM = 512
    HIDDEN_DIM2 = 256
    DROPOUT = 0.4
    
    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 30
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()


# ============================================================================
# FOCAL LOSS - Better for class imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma   # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Standard cross entropy
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p_t = torch.exp(-ce_loss)
        
        # Apply focal loss weighting
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# MODEL - Shared MLP Trunk with separate heads
# ============================================================================

class MultiTaskVQA(nn.Module):
    def __init__(self, feature_dim=192, hidden_dim=512, hidden_dim2=256, dropout=0.4, num_quality_classes=5):
        super().__init__()
        
        # SHARED MLP TRUNK
        self.shared_trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # MOS head (regression)
        self.mos_head = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim2 // 2, 1)
        )
        
        # Quality head (classification)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim2 // 2, num_quality_classes)
        )
        
        # Artifact heads (binary classification) - INDEPENDENT heads for each artifact
        # We must initialize 7 separate heads so they can learn different weights!
        artifact_hidden = hidden_dim2
        
        def create_artifact_head():
            return nn.Sequential(
                nn.Linear(hidden_dim2, artifact_hidden),
                nn.LayerNorm(artifact_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(artifact_hidden, artifact_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(artifact_hidden // 2, 2)
            )
            
        self.artifact_heads = nn.ModuleDict({
            'spatial': create_artifact_head(),
            'hallucination': create_artifact_head(),
            'lighting': create_artifact_head(),
            'rendering': create_artifact_head(),
            'physics': create_artifact_head(),
            'motion': create_artifact_head(),
            'flicker': create_artifact_head()
        })
        
    def forward(self, x):
        shared_features = self.shared_trunk(x)
        
        return {
            'mos': self.mos_head(shared_features).squeeze(-1),
            'quality': self.quality_head(shared_features),
            'spatial': self.artifact_heads['spatial'](shared_features),
            'hallucination': self.artifact_heads['hallucination'](shared_features),
            'lighting': self.artifact_heads['lighting'](shared_features),
            'rendering': self.artifact_heads['rendering'](shared_features),
            'physics': self.artifact_heads['physics'](shared_features),
            'motion': self.artifact_heads['motion'](shared_features),
            'flicker': self.artifact_heads['flicker'](shared_features)
        }


# ============================================================================
# DATASET
# ============================================================================

class VQADataset(Dataset):
    QUALITY_TO_IDX = {'Bad': 0, 'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    
    def __init__(self, features_dir, metadata_path, split):
        self.features_dir = os.path.join(features_dir, split, 'layer4')
        self.split = split
        self.df = pd.read_csv(metadata_path)
        
        self.feature_files = {}
        for filename in self.df['filename']:
            base_name = filename.replace('.mp4', '').replace('_standardized', '')
            feature_path = os.path.join(self.features_dir, f'{base_name}_features.pt')
            if os.path.exists(feature_path):
                self.feature_files[filename] = feature_path
        
        self.df = self.df[self.df['filename'].isin(self.feature_files.keys())]
        print(f"Loaded {len(self.df)} samples for {split}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = self.feature_files[row['filename']]
        
        feat = torch.load(feature_path)
        feat = feat.mean(dim=(1, 2, 3))
        
        return {
            'features': feat,
            'mos': torch.tensor(row['mos'], dtype=torch.float32),
            'quality': torch.tensor(self.QUALITY_TO_IDX.get(row['quality_class'], 2), dtype=torch.long),
            'spatial': torch.tensor(int(row['spatial_flag']), dtype=torch.long),
            'hallucination': torch.tensor(int(row['hallucination_flag']), dtype=torch.long),
            'lighting': torch.tensor(int(row['lighting_flag']), dtype=torch.long),
            'rendering': torch.tensor(int(row['rendering_flag']), dtype=torch.long),
            'physics': torch.tensor(int(row['physics_violation_flag']), dtype=torch.long),
            'motion': torch.tensor(int(row['motion_inconsistency_flag']), dtype=torch.long),
            'flicker': torch.tensor(int(row['object_flicker_flag']), dtype=torch.long),
        }


# ============================================================================
# LOSS - Using Focal Loss for artifact tasks
# ============================================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mos_loss_fn = nn.MSELoss()
        self.quality_loss_fn = nn.CrossEntropyLoss()
        
        # AGGRESSIVE class weights based on dataset distribution
        # physics: 92% class 0 -> weight class 1 heavily
        # Use Focal Loss with gamma=2.0 for severe imbalance
        
        # For highly imbalanced (physics, motion, lighting, rendering): gamma=3.0
        # For moderate imbalance (spatial, hallucination, flicker): gamma=2.0
        
        # Class weights: [weight_for_0, weight_for_1]
        # For physics (92/8): need ~11x weight for class 1
        self.physics_loss_fn = FocalLoss(
            alpha=torch.tensor([0.1, 0.9]).to(device),  # Very aggressive
            gamma=3.0
        )
        
        # For motion (78/22): need ~3.5x weight for class 1
        self.motion_loss_fn = FocalLoss(
            alpha=torch.tensor([0.2, 0.8]).to(device),
            gamma=2.5
        )
        
        # For lighting (75/25): need ~3x weight for class 1
        self.lighting_loss_fn = FocalLoss(
            alpha=torch.tensor([0.25, 0.75]).to(device),
            gamma=2.5
        )
        
        # For rendering (72/28): need ~2.5x weight for class 1
        self.rendering_loss_fn = FocalLoss(
            alpha=torch.tensor([0.3, 0.7]).to(device),
            gamma=2.5
        )
        
        # For flicker (82/18): need ~4.5x weight for class 1
        self.flicker_loss_fn = FocalLoss(
            alpha=torch.tensor([0.15, 0.85]).to(device),
            gamma=2.5
        )
        
        # For spatial/hallucination (50/50): use standard CE
        self.spatial_loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0]).to(device)
        )
        self.hallucination_loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0]).to(device)
        )
        
    def forward(self, predictions, targets):
        mos_loss = self.mos_loss_fn(predictions['mos'], targets['mos'])
        quality_loss = self.quality_loss_fn(predictions['quality'], targets['quality'])
        
        # Loss weights for different tasks
        total_loss = (
            1.5 * mos_loss + 
            1.0 * quality_loss +
            1.0 * self.spatial_loss_fn(predictions['spatial'], targets['spatial']) +
            1.0 * self.hallucination_loss_fn(predictions['hallucination'], targets['hallucination']) +
            2.0 * self.lighting_loss_fn(predictions['lighting'], targets['lighting']) +
            2.5 * self.rendering_loss_fn(predictions['rendering'], targets['rendering']) +
            3.0 * self.physics_loss_fn(predictions['physics'], targets['physics']) +
            2.5 * self.motion_loss_fn(predictions['motion'], targets['motion']) +
            1.5 * self.flicker_loss_fn(predictions['flicker'], targets['flicker'])
        )
        
        return {'total': total_loss}


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(predictions, targets):
    metrics = {}
    
    # MOS metrics
    mos_pred = predictions['mos'].detach().cpu().numpy()
    mos_true = targets['mos'].detach().cpu().numpy()
    metrics['mos_rmse'] = float(np.sqrt(np.mean((mos_pred - mos_true) ** 2)))
    
    try:
        metrics['mos_plcc'] = float(pearsonr(mos_pred, mos_true)[0])
    except:
        metrics['mos_plcc'] = 0.0
    
    try:
        metrics['mos_srcc'] = float(spearmanr(mos_pred, mos_true)[0])
    except:
        metrics['mos_srcc'] = 0.0
    
    # Quality accuracy
    quality_pred = predictions['quality'].argmax(dim=1)
    quality_true = targets['quality']
    metrics['quality_accuracy'] = float((quality_pred == quality_true).float().mean())
    
    # Artifact metrics
    artifact_mapping = {
        'spatial': 'spatial',
        'hallucination': 'hallucination',
        'lighting': 'lighting',
        'rendering': 'rendering',
        'physics': 'physics',
        'motion': 'motion',
        'flicker': 'flicker'
    }
    
    for name, csv_name in artifact_mapping.items():
        pred = predictions[name].argmax(dim=1).detach().cpu().numpy()
        true = targets[name].detach().cpu().numpy()
        
        metrics[f'{csv_name}_accuracy'] = float((pred == true).mean())
        
        # Per-class F1
        f1_scores = []
        for cls in [0, 1]:
            tp = int(((pred == cls) & (true == cls)).sum())
            fp = int(((pred == cls) & (true != cls)).sum())
            fn = int(((pred != cls) & (true == cls)).sum())
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(f1)
        
        metrics[f'{csv_name}_flag_f1'] = float(np.mean(f1_scores))
        
        # Calculate AUC
        try:
            probs = torch.softmax(predictions[name], dim=1)[:, 1].detach().cpu().numpy()
            auc = roc_auc_score(true, probs)
            metrics[f'{csv_name}_auc'] = float(auc)
        except ValueError:
            # Occurs if only one class is present in the batch
            metrics[f'{csv_name}_auc'] = 0.5
    
    return metrics


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    all_preds = {k: [] for k in ['mos', 'quality', 'spatial', 'hallucination', 'lighting', 'rendering', 'physics', 'motion', 'flicker']}
    all_targets = {k: [] for k in all_preds.keys()}
    
    for batch in tqdm(dataloader, desc="Training"):
        features = batch['features'].to(device)
        
        targets = {
            'mos': batch['mos'].to(device),
            'quality': batch['quality'].to(device),
            'spatial': batch['spatial'].to(device),
            'hallucination': batch['hallucination'].to(device),
            'lighting': batch['lighting'].to(device),
            'rendering': batch['rendering'].to(device),
            'physics': batch['physics'].to(device),
            'motion': batch['motion'].to(device),
            'flicker': batch['flicker'].to(device)
        }
        
        optimizer.zero_grad()
        predictions = model(features)
        losses = loss_fn(predictions, targets)
        
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
    
    # Compute metrics with dropout OFF
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = {k: batch[k].to(device) for k in ['mos', 'quality', 'spatial', 'hallucination', 'lighting', 'rendering', 'physics', 'motion', 'flicker']}
            predictions = model(features)
            
            for key in predictions.keys():
                all_preds[key].append(predictions[key])
                all_targets[key].append(targets[key])
    
    for key in all_preds:
        all_preds[key] = torch.cat(all_preds[key], dim=0)
        all_targets[key] = torch.cat(all_targets[key], dim=0)
    
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader)
    
    model.train()
    
    return metrics


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    all_preds = {k: [] for k in ['mos', 'quality', 'spatial', 'hallucination', 'lighting', 'rendering', 'physics', 'motion', 'flicker']}
    all_targets = {k: [] for k in all_preds.keys()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch['features'].to(device)
            
            targets = {
                'mos': batch['mos'].to(device),
                'quality': batch['quality'].to(device),
                'spatial': batch['spatial'].to(device),
                'hallucination': batch['hallucination'].to(device),
                'lighting': batch['lighting'].to(device),
                'rendering': batch['rendering'].to(device),
                'physics': batch['physics'].to(device),
                'motion': batch['motion'].to(device),
                'flicker': batch['flicker'].to(device)
            }
            
            predictions = model(features)
            losses = loss_fn(predictions, targets)
            
            total_loss += losses['total'].item()
            
            for key in predictions.keys():
                all_preds[key].append(predictions[key])
                all_targets[key].append(targets[key])
    
    for key in all_preds:
        all_preds[key] = torch.cat(all_preds[key], dim=0)
        all_targets[key] = torch.cat(all_targets[key], dim=0)
    
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    device = config.DEVICE
    print(f"Using device: {device}")
    print("Training with FOCAL LOSS for artifact detection")
    
    print("\nLoading datasets...")
    train_dataset = VQADataset(config.FEATURES_DIR, os.path.join(config.METADATA_DIR, 'train.csv'), 'train')
    val_dataset = VQADataset(config.FEATURES_DIR, os.path.join(config.METADATA_DIR, 'val.csv'), 'val')
    test_dataset = VQADataset(config.FEATURES_DIR, os.path.join(config.METADATA_DIR, 'test.csv'), 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print("\nCreating model...")
    model = MultiTaskVQA(config.FEATURE_DIM, config.HIDDEN_DIM, config.HIDDEN_DIM2, config.DROPOUT).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = MultiTaskLoss(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    print("\nStarting training...")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, MOS RMSE: {train_metrics['mos_rmse']:.2f}, PLCC: {train_metrics['mos_plcc']:.3f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, MOS RMSE: {val_metrics['mos_rmse']:.2f}, PLCC: {val_metrics['mos_plcc']:.3f}")
        
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                      os.path.join(config.OUTPUT_DIR, 'best_model.pt'))
            print(f"✓ Saved best model (epoch {epoch+1})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement: {epochs_without_improvement}/{config.EARLY_STOPPING_PATIENCE}")
            
        # Force save intermediate epochs for UMAP generation
        if (epoch + 1) == 1 or (epoch + 1) % 15 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                      os.path.join(config.OUTPUT_DIR, f'model_epoch_{epoch+1}.pt'))
        
        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}!")
            break
    
    # Save CSVs
    train_df = pd.DataFrame(train_history)
    train_df['epoch'] = range(1, len(train_history) + 1)
    train_df.to_csv(os.path.join(config.OUTPUT_DIR, 'train_metrics_detailed.csv'), index=False)
    
    val_df = pd.DataFrame(val_history)
    val_df['epoch'] = range(1, len(val_history) + 1)
    val_df.to_csv(os.path.join(config.OUTPUT_DIR, 'val_metrics_detailed.csv'), index=False)
    
    # Test evaluation
    print("\n" + "="*60)
    print("Evaluating on test set...")
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    
    print("\n" + "="*60)
    print("TEST RESULTS (FOCAL LOSS):")
    print("="*60)
    print(f"\n📊 MOS Regression:")
    print(f"   RMSE: {test_metrics['mos_rmse']:.4f}")
    print(f"   PLCC: {test_metrics['mos_plcc']:.4f}")
    print(f"   SRCC: {test_metrics['mos_srcc']:.4f}")
    
    print(f"\n📈 Quality Classification:")
    print(f"   Accuracy: {test_metrics['quality_accuracy']*100:.2f}%")
    
    print(f"\n🎯 Artifact Detection:")
    artifacts = ['spatial', 'hallucination', 'lighting', 'rendering', 'physics', 'motion', 'flicker']
    for art in artifacts:
        acc = test_metrics[f'{art}_accuracy'] * 100
        f1 = test_metrics[f'{art}_flag_f1']
        auc = test_metrics.get(f'{art}_auc', 0.5)
        print(f"   {art.capitalize():15s} - Accuracy: {acc:6.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Save test results
    test_row = {
        'epoch': best_epoch + 1,
        'mos_rmse': test_metrics['mos_rmse'],
        'mos_plcc': test_metrics['mos_plcc'],
        'mos_srcc': test_metrics['mos_srcc'],
        'quality_accuracy': test_metrics['quality_accuracy'],
    }
    for art in artifacts:
        test_row[f'{art}_accuracy'] = test_metrics[f'{art}_accuracy']
        test_row[f'{art}_flag_f1'] = test_metrics[f'{art}_flag_f1']
        test_row[f'{art}_auc'] = test_metrics.get(f'{art}_auc', 0.5)
    
    test_df = pd.DataFrame([test_row])
    test_df.to_csv(os.path.join(config.OUTPUT_DIR, 'test_metrics_detailed.csv'), index=False)
    
    print("\n✓ Results saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='s', choices=['xs', 's', 'm'], help='X3D variant to train on')
    args = parser.parse_args()
    
    config.VARIANT = args.variant
    config.FEATURES_DIR = f"/home/drive3/low_encoder/features_10k_{args.variant}"
    config.OUTPUT_DIR = f"/home/drive3/low_encoder/results_10k_{args.variant}"
    
    main()
