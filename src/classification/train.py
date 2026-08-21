import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.classification.dataset import VehicleMultiLabelDataset, get_transforms, CLASSES
from src.classification.model import get_vehicle_classifier

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_pos_weight(df, classes=CLASSES):
    """
    Computes exact positive weight (neg_count / pos_count) per class for BCEWithLogitsLoss.
    """
    num_samples = len(df)
    pos_counts = df[classes].astype('float32').sum().values
    neg_counts = num_samples - pos_counts
    pos_weight = neg_counts / pos_counts
    return torch.tensor(pos_weight, dtype=torch.float32)

def evaluate(model, val_loader, criterion, device, thresholds=[0.3, 0.4, 0.5, 0.6]):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)

    results_per_threshold = {}
    for th in thresholds:
        preds = (all_probs >= th).astype(int)
        
        per_class_f1 = f1_score(all_targets, preds, average=None, zero_division=0)
        per_class_prec = precision_score(all_targets, preds, average=None, zero_division=0)
        per_class_rec = recall_score(all_targets, preds, average=None, zero_division=0)
        
        macro_f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
        h_loss = hamming_loss(all_targets, preds)

        results_per_threshold[th] = {
            'macro_f1': macro_f1,
            'hamming_loss': h_loss,
            'per_class_f1': {cls: f1 for cls, f1 in zip(CLASSES, per_class_f1)},
            'per_class_prec': {cls: p for cls, p in zip(CLASSES, per_class_prec)},
            'per_class_rec': {cls: r for cls, r in zip(CLASSES, per_class_rec)}
        }

    return val_loss, results_per_threshold

def train_model(mode='finetuned', max_epochs=15, batch_size=64, lr_head=1e-3, lr_backbone=1e-4, seed=42):
    seed_everything(seed)
    
    # Verify MPS device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\n==========================================")
    print(f"Starting Training Run: Mode={mode.upper()}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"==========================================")

    # Paths
    train_csv = 'data/vehicles-coco/train/_classes.csv'
    train_dir = 'data/vehicles-coco/train'
    val_csv = 'data/vehicles-coco/valid/_classes.csv'
    val_dir = 'data/vehicles-coco/valid'

    train_transform, val_transform = get_transforms(img_size=224)

    train_dataset = VehicleMultiLabelDataset(train_csv, train_dir, transform=train_transform)
    val_dataset = VehicleMultiLabelDataset(val_csv, val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Calculate exact pos_weight
    pos_weight = compute_pos_weight(train_dataset.df)
    print("Exact pos_weight computed from train split:")
    for cls_name, pw in zip(CLASSES, pos_weight):
        print(f"  {cls_name:12s}: {pw.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Initialize model
    is_pretrained = (mode == 'finetuned')
    model = get_vehicle_classifier(pretrained=is_pretrained, num_classes=4).to(device)

    # Model save path
    os.makedirs('models', exist_ok=True)
    model_save_path = f"models/classifier_{mode}.pt"

    # Setup Optimizer & Freeze Strategy
    if mode == 'finetuned':
        # Freeze backbone initially (Epoch 1-2)
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr_head, weight_decay=1e-4)
        print("Initial state: Backbone frozen, training head only.")
    else:
        # From scratch: all parameters active
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_head, weight_decay=1e-4)
        print("Initial state: All layers trainable (Random Initialization).")

    history = []
    best_macro_f1 = 0.0
    patience = 3
    patience_counter = 0
    start_wall_time = time.time()
    max_wall_time_sec = 35 * 60  # 35 minutes cap

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        
        # Unfreeze backbone after epoch 2 for finetuned mode with discriminative LR
        if mode == 'finetuned' and epoch == 3:
            print("\n>>> Unfreezing backbone for fine-tuning with discriminative learning rates...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': lr_backbone},
                {'params': model.fc.parameters(), 'lr': lr_head}
            ], weight_decay=1e-4)

        # Training Phase
        model.train()
        running_loss = 0.0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)

        # Validation Phase
        val_loss, th_results = evaluate(model, val_loader, criterion, device)
        epoch_sec = time.time() - epoch_start
        total_elapsed = time.time() - start_wall_time

        macro_f1_50 = th_results[0.5]['macro_f1']
        h_loss_50 = th_results[0.5]['hamming_loss']

        print(f"Epoch [{epoch:02d}/{max_epochs:02d}] ({epoch_sec:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Macro-F1 (0.5): {macro_f1_50:.4f} | Val Hamming Loss: {h_loss_50:.4f}")
        for th in [0.3, 0.4, 0.5]:
            f1s = th_results[th]['per_class_f1']
            print(f"   th={th:.1f} -> Macro-F1: {th_results[th]['macro_f1']:.4f} | "
                  f"bus: {f1s['bus']:.3f}, car: {f1s['car']:.3f}, moto: {f1s['motorcycle']:.3f}, truck: {f1s['truck']:.3f}")

        # Record history
        record = {
            'mode': mode,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time_sec': epoch_sec,
            'cum_time_sec': total_elapsed,
            'val_macro_f1_0.5': macro_f1_50,
            'val_macro_f1_0.4': th_results[0.4]['macro_f1'],
            'val_macro_f1_0.3': th_results[0.3]['macro_f1'],
            'val_hamming_loss_0.5': h_loss_50,
        }
        for cls in CLASSES:
            record[f'val_f1_{cls}_0.5'] = th_results[0.5]['per_class_f1'][cls]
            record[f'val_prec_{cls}_0.5'] = th_results[0.5]['per_class_prec'][cls]
            record[f'val_rec_{cls}_0.5'] = th_results[0.5]['per_class_rec'][cls]
            record[f'val_f1_{cls}_0.4'] = th_results[0.4]['per_class_f1'][cls]

        history.append(record)

        # Incrementally update CSV log file
        df_hist = pd.DataFrame(history)
        csv_path = 'docs/training_curves.csv'
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            # Remove existing rows for this mode to avoid duplicate records on resume
            df_old = df_old[df_old['mode'] != mode]
            df_combined = pd.concat([df_old, df_hist], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        else:
            df_hist.to_csv(csv_path, index=False)

        # Save Best Model & Check Early Stopping
        if macro_f1_50 > best_macro_f1:
            best_macro_f1 = macro_f1_50
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"   --> Best model saved to {model_save_path} (Val Macro-F1: {best_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in validation Macro-F1 for {patience} consecutive epochs.")
                break

        if total_elapsed >= max_wall_time_sec:
            print(f"\n[Wall-Clock Limit] Hard limit of {max_wall_time_sec/60:.1f} mins reached. Stopping training.")
            break

    print(f"\nTraining finished for {mode.upper()}. Best Val Macro-F1 (0.5): {best_macro_f1:.4f}")
    return pd.DataFrame(history)

def plot_curves():
    csv_path = 'docs/training_curves.csv'
    if not os.path.exists(csv_path):
        return
    df_all = pd.read_csv(csv_path)
    df_ft = df_all[df_all['mode'] == 'finetuned']
    df_sc = df_all[df_all['mode'] == 'scratch']

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    if len(df_ft) > 0:
        plt.plot(df_ft['epoch'], df_ft['train_loss'], 'b-o', label='FT Train Loss')
        plt.plot(df_ft['epoch'], df_ft['val_loss'], 'b--s', label='FT Val Loss')
    if len(df_sc) > 0:
        plt.plot(df_sc['epoch'], df_sc['train_loss'], 'r-o', label='Scratch Train Loss')
        plt.plot(df_sc['epoch'], df_sc['val_loss'], 'r--s', label='Scratch Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Macro-F1 plot
    plt.subplot(1, 2, 2)
    if len(df_ft) > 0:
        plt.plot(df_ft['epoch'], df_ft['val_macro_f1_0.5'], 'b-o', label='FT Macro-F1 (th=0.5)')
        plt.plot(df_ft['epoch'], df_ft['val_macro_f1_0.4'], 'b:', label='FT Macro-F1 (th=0.4)')
    if len(df_sc) > 0:
        plt.plot(df_sc['epoch'], df_sc['val_macro_f1_0.5'], 'r-o', label='Scratch Macro-F1 (th=0.5)')
        plt.plot(df_sc['epoch'], df_sc['val_macro_f1_0.4'], 'r:', label='Scratch Macro-F1 (th=0.4)')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.title('Validation Macro-F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'docs/training_curves.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Training curves plot saved to {plot_path}")

def main():
    os.makedirs('docs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Run Fine-Tuned Model (6 Epochs)
    df_ft = train_model(mode='finetuned', max_epochs=6, batch_size=64, seed=42)
    plot_curves()
    
    # 2. Run From-Scratch Model (15 Epochs)
    df_sc = train_model(mode='scratch', max_epochs=15, batch_size=64, seed=42)
    plot_curves()

if __name__ == '__main__':
    main()

