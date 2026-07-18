import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.dataset import load_ptbxl_metadata, PTBXLDataset
from src.models.encoder import ECGEncoder1D, ECGClassifier1D

def calculate_auroc(y_true, y_probs):
    """Calculates multi-label macro-averaged AUROC score."""
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    auc_scores = []
    num_classes = y_true.shape[1]
    for c in range(num_classes):
        if len(np.unique(y_true[:, c])) > 1:
            auc = roc_auc_score(y_true[:, c], y_probs[:, c])
            auc_scores.append(auc)
        else:
            auc_scores.append(0.5)
            
    return np.mean(auc_scores)

def train_and_evaluate(model, train_dataset, test_loader, epochs, lr, batch_size, device):
    """Trains a model on a training dataset and evaluates AUROC on the test set."""
    num_workers = 4 if device.type == "cuda" else 0
    persistent_workers = True if num_workers > 0 else False
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    # Simple, high-performance training loop
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
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
                
    # Evaluation on Test set
    model.eval()
    all_true = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(x)
            else:
                logits = model(x)
                
            probs = torch.sigmoid(logits)
            all_true.append(y.numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_true = np.concatenate(all_true, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    return calculate_auroc(all_true, all_probs)

def draw_ascii_chart(fractions, random_scores, jepa_scores):
    """Draws a highly visual ASCII line chart comparing label efficiency of Random vs CardioRep."""
    # Build a grid of size 12 rows (vertical) by 45 columns (horizontal)
    # y ranges from 0.50 to 1.00
    rows = 12
    cols = 50
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    
    y_min, y_max = 0.50, 1.00
    
    # Map coordinates
    # x goes from index 0 to 5 (corresponding to the 6 fractions)
    # We map x index [0, 5] -> col indices in [5, 45]
    x_col_indices = np.linspace(5, cols - 5, len(fractions), dtype=int)
    
    # Plot curves
    for i in range(len(fractions)):
        # Random Init
        y_r = random_scores[i]
        r_row = rows - 1 - int((y_r - y_min) / (y_max - y_min) * (rows - 1))
        r_row = max(0, min(rows - 1, r_row))
        grid[r_row][x_col_indices[i]] = "R"
        
        # CardioRep
        y_j = jepa_scores[i]
        j_row = rows - 1 - int((y_j - y_min) / (y_max - y_min) * (rows - 1))
        j_row = max(0, min(rows - 1, j_row))
        grid[j_row][x_col_indices[i]] = "C"
        
        # Overlap
        if r_row == j_row:
            grid[r_row][x_col_indices[i]] = "X"
            
    # Print chart
    print("\n📈 CardioRep Label Efficiency AUROC Curve")
    print("--------------------------------------------------")
    for r in range(rows):
        y_val = y_max - r * (y_max - y_min) / (rows - 1)
        # Left Y axis
        line = f"{y_val:.2f} | "
        for c in range(cols):
            line += grid[r][c]
        print(line)
        
    # Bottom X axis
    bottom_line = "     +--" + ("-" * cols)
    print(bottom_line)
    
    # X axis labels
    label_line = "        "
    for idx, f in enumerate(fractions):
        f_percent = f"{int(f*100)}%"
        # Calculate padding to align under columns
        target_pos = x_col_indices[idx]
        current_pos = len(label_line) - 8
        label_line += " " * (target_pos - current_pos) + f_percent
    print(label_line)
    print("\nLegend:  [C] = CardioRep (Pretrained)  [R] = Random Init  [X] = Overlap")
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="CardioRep Label Efficiency Experiment")
    parser.add_argument("--data_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_sig1.pt", help="Path to pretrained JEPA weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per fraction")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print("🧪 Running CardioRep Label Efficiency Experiment")
    print("--------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    # 1. Load dataset metadata
    df, superclasses = load_ptbxl_metadata(args.data_dir)
    train_df_all = df[df["strat_fold"].isin(list(range(1, 9)))].copy()
    
    # Load Test dataset
    test_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=[10], augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4 if device.type == "cuda" else 0, pin_memory=True)
    
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    random_auroc_results = []
    jepa_auroc_results = []
    
    print(f"Fractions to evaluate: {[f'{int(f*100)}%' for f in fractions]}")
    
    for frac in fractions:
        print(f"\n⚡ Evaluating subset fraction: {int(frac*100)}% ...")
        
        # Sub-sample training metadata DataFrame deterministically
        frac_train_df = train_df_all.sample(frac=frac, random_state=42)
        print(f"  - Subset contains {len(frac_train_df)} training records.")
        
        # Instantiate Train Dataset for this subset
        train_dataset_sub = PTBXLDataset(args.data_dir, frac_train_df, superclasses, fold_list=list(range(1, 9)), augment=True)
        
        # A. Evaluate Random Initialization
        print("  - [Random Init] Training from scratch...")
        encoder_rand = ECGEncoder1D(in_channels=12, latent_dim=256)
        model_rand = ECGClassifier1D(encoder=encoder_rand, num_classes=len(superclasses)).to(device)
        rand_auroc = train_and_evaluate(model_rand, train_dataset_sub, test_loader, args.epochs, args.lr, args.batch_size, device)
        random_auroc_results.append(rand_auroc)
        print(f"    -> Test AUROC: {rand_auroc:.4f}")
        
        # B. Evaluate CardioRep Pretrained
        print("  - [CardioRep Pretrained] Load weights & fine-tuning...")
        encoder_jepa = ECGEncoder1D(in_channels=12, latent_dim=256)
        # Load weights
        checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)
        encoder_jepa.load_state_dict(checkpoint["encoder_state_dict"])
        model_jepa = ECGClassifier1D(encoder=encoder_jepa, num_classes=len(superclasses)).to(device)
        
        jepa_auroc = train_and_evaluate(model_jepa, train_dataset_sub, test_loader, args.epochs, args.lr, args.batch_size, device)
        jepa_auroc_results.append(jepa_auroc)
        print(f"    -> Test AUROC: {jepa_auroc:.4f}")
        
    # Print Summary Table
    print("\n" + "=" * 50)
    print("📊 LABEL EFFICIENCY EXPERIMENT COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Fraction':<10} | {'Random Init AUROC':<18} | {'CardioRep AUROC':<18} | {'Absolute Gain':<10}")
    print("-" * 55)
    for i, frac in enumerate(fractions):
        gain = jepa_auroc_results[i] - random_auroc_results[i]
        print(f"{int(frac*100):>3}%        | {random_auroc_results[i]:.4f}             | {jepa_auroc_results[i]:.4f}           | {gain:+.4f}")
    print("=" * 50)
    
    # Save results to CSV
    results_csv = "checkpoints/label_efficiency_results.csv"
    import csv
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fraction", "random_init_auroc", "cardiorep_auroc", "absolute_gain"])
        for i in range(len(fractions)):
            writer.writerow([fractions[i], random_auroc_results[i], jepa_auroc_results[i], jepa_auroc_results[i] - random_auroc_results[i]])
    print(f"Results CSV saved to: {results_csv}")
    
    # Plot ASCII visual chart
    draw_ascii_chart(fractions, random_auroc_results, jepa_auroc_results)

if __name__ == "__main__":
    main()
