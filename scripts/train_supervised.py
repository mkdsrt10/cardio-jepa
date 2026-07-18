import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from src.data.dataset import load_ptbxl_metadata, PTBXLDataset
from src.models.encoder import ECGEncoder1D, ECGClassifier1D

def calculate_metrics(y_true, y_probs):
    """Calculates multi-label AUROC and macro F1 scores robustly."""
    # Convert to numpy
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_preds = (y_probs >= 0.5).astype(int)
    
    # Calculate AUROC per class (handling classes without positive/negative samples gracefully)
    auc_scores = []
    num_classes = y_true.shape[1]
    for c in range(num_classes):
        if len(np.unique(y_true[:, c])) > 1:
            auc = roc_auc_score(y_true[:, c], y_probs[:, c])
            auc_scores.append(auc)
        else:
            # Fallback for small synthetic datasets where some folds might miss a class
            auc_scores.append(0.5)
            
    macro_auroc = np.mean(auc_scores)
    
    # Calculate Macro F1 score
    macro_f1 = f1_score(y_true, y_preds, average="macro", zero_division=0)
    
    return macro_auroc, macro_f1, auc_scores

def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Automatic Mixed Precision for High-Performance acceleration
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
                
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent gradient explosions
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        total_loss += loss.item() * x.size(0)
        
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_probs = []
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)
            
        total_loss += loss.item() * x.size(0)
        
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        all_true.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        
    all_true = np.concatenate(all_true, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    auroc, f1, class_aucs = calculate_metrics(all_true, all_probs)
    
    return avg_loss, auroc, f1, class_aucs

def main():
    parser = argparse.ArgumentParser(description="PTB-XL 1D ResNet Supervised Baseline")
    parser.add_argument("--data_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--latent_dim", type=int, default=256, help="Encoder embedding size")
    parser.add_argument("--save_path", type=str, default="checkpoints/supervised_baseline.pt", help="Path to save model checkpoint")
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print("🚀 PTB-XL Supervised Baseline training pipeline")
    print("--------------------------------------------------")
    
    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    # Create save directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 1. Load Metadata & Class mappings
    try:
        df, superclasses = load_ptbxl_metadata(args.data_dir)
        print(f"Metadata loaded. Found {len(df)} records. Diagnostic superclasses: {superclasses}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
        
    # 2. Setup Splits (1-8 Train, 9 Val, 10 Test)
    train_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=list(range(1, 9)), augment=True)
    val_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=[9], augment=False)
    test_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=[10], augment=False)
    
    # 3. Create High-Performance Dataloaders (using Playbook tuning)
    num_workers = 4 if device.type == "cuda" else 0
    persistent_workers = True if num_workers > 0 else False
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )
    
    print(f"Splits set up successfully:")
    print(f"  - Train: {len(train_dataset)} records ({len(train_loader)} batches)")
    print(f"  - Val:   {len(val_dataset)} records ({len(val_loader)} batches)")
    print(f"  - Test:  {len(test_dataset)} records ({len(test_loader)} batches)")
    
    # 4. Instantiate Model, Loss, Optimizer, and Scaler
    encoder = ECGEncoder1D(in_channels=12, latent_dim=args.latent_dim)
    model = ECGClassifier1D(encoder=encoder, num_classes=len(superclasses))
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    # 5. Training loop
    best_val_loss = float("inf")
    best_val_auroc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_auroc, val_f1, val_class_aucs = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUROC: {val_auroc:.4f} | "
              f"Val F1: {val_f1:.4f}")
              
        # Track best model checkpoint
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "superclasses": superclasses,
                "val_auroc": val_auroc,
                "val_f1": val_f1
            }, args.save_path)
            
    print(f"\nTraining complete. Best validation AUROC: {best_val_auroc:.4f}")
    print(f"Best model checkpoint saved to: {args.save_path}")
    
    # 6. Evaluate on Test Split
    print("\nEvaluating best checkpoint on Test set (Fold 10)...")
    if os.path.exists(args.save_path):
        checkpoint = torch.load(args.save_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
    test_loss, test_auroc, test_f1, test_class_aucs = evaluate(model, test_loader, criterion, device)
    print("--------------------------------------------------")
    print(f"📋 TEST SET EVALUATION RESULTS (Fold 10)")
    print("--------------------------------------------------")
    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1:    {test_f1:.4f}")
    print("Class-wise AUROC scores:")
    for i, s_class in enumerate(superclasses):
        print(f"  - {s_class}: {test_class_aucs[i]:.4f}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
