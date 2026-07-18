import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import load_ptbxl_metadata, PTBXLDataset
from src.models.encoder import ECGEncoder1D
from src.models.jepa import ECGJEPA, apply_jepa_masking, sigreg_loss

@torch.no_grad()
def calculate_effective_rank(z: torch.Tensor, max_samples=1024) -> float:
    """Calculates the Effective Rank of representation embeddings.
    
    Guarantees stable FP32 diagnostics on CPU for a limited sample size,
    following the High-Performance Playbook.
    """
    # Limit sample size to avoid O(D^3) SVD bottle-necks
    if z.size(0) > max_samples:
        indices = torch.randperm(z.size(0))[:max_samples]
        z = z[indices]
        
    # Cast representation to float32 on CPU to prevent FP16 SVD support issues
    z_f32 = z.float().cpu()
    
    # Singular Value Decomposition
    try:
        singular_values = torch.linalg.svdvals(z_f32)
        sum_sv = torch.sum(singular_values)
        if sum_sv > 1e-8:
            p = singular_values / sum_sv
            # Shannon entropy
            entropy = -torch.sum(p * torch.log(p + 1e-10))
            # Effective Rank is exp(H)
            eff_rank = torch.exp(entropy).item()
            return eff_rank
    except Exception as e:
        # Fallback in case of numerical instability
        pass
    return 1.0

def pretrain_epoch(model, dataloader, optimizer, sigreg_weight, device, scaler):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_sig = 0.0
    
    for x, _ in dataloader:
        x = x.to(device)
        
        # Apply JEPA Context-Target masking
        # Context view has masked time steps and/or leads
        x_masked = apply_jepa_masking(x, mask_type="both", mask_ratio_time=0.25, mask_leads_count=3)
        
        optimizer.zero_grad()
        
        # High-Performance AMP pretraining forward-pass
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                # Forward Context Encoder & Predictor
                z_predicted, z_context = model.forward_context(x_masked)
                
                # Forward Target Encoder (using clean augmented x, weights are EMA)
                z_target = model.forward_target(x)
                
                # 1. Prediction (MSE) loss
                mse_loss = nn.functional.mse_loss(z_predicted, z_target)
                
                # 2. Signature Regularization (SIGReg) loss to prevent representation collapse
                sig_loss_pred = sigreg_loss(z_predicted, lambd=1.0)
                sig_loss_target = sigreg_loss(z_target, lambd=1.0)
                total_sig_loss = sig_loss_pred + sig_loss_target
                
                # Combined JEPA Loss
                loss = mse_loss + sigreg_weight * total_sig_loss
                
            scaler.scale(loss).backward()
            
            # Gradient clipping to stabilize pretraining
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            z_predicted, z_context = model.forward_context(x_masked)
            z_target = model.forward_target(x)
            
            mse_loss = nn.functional.mse_loss(z_predicted, z_target)
            sig_loss_pred = sigreg_loss(z_predicted, lambd=1.0)
            sig_loss_target = sigreg_loss(z_target, lambd=1.0)
            total_sig_loss = sig_loss_pred + sig_loss_target
            
            loss = mse_loss + sigreg_weight * total_sig_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Target Encoder EMA parameter weight copy update
        model.update_target_ema()
        
        total_loss += loss.item() * x.size(0)
        total_mse += mse_loss.item() * x.size(0)
        total_sig += total_sig_loss.item() * x.size(0)
        
    n = len(dataloader.dataset)
    return total_loss / n, total_mse / n, total_sig / n

@torch.no_grad()
def evaluate_pretraining(model, dataloader, sigreg_weight, device):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_sig = 0.0
    all_representations = []
    
    for x, _ in dataloader:
        x = x.to(device)
        x_masked = apply_jepa_masking(x, mask_type="both", mask_ratio_time=0.25, mask_leads_count=3)
        
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                z_predicted, z_context = model.forward_context(x_masked)
                z_target = model.forward_target(x)
                
                mse_loss = nn.functional.mse_loss(z_predicted, z_target)
                sig_loss_pred = sigreg_loss(z_predicted, lambd=1.0)
                sig_loss_target = sigreg_loss(z_target, lambd=1.0)
                total_sig_loss = sig_loss_pred + sig_loss_target
                loss = mse_loss + sigreg_weight * total_sig_loss
        else:
            z_predicted, z_context = model.forward_context(x_masked)
            z_target = model.forward_target(x)
            
            mse_loss = nn.functional.mse_loss(z_predicted, z_target)
            sig_loss_pred = sigreg_loss(z_predicted, lambd=1.0)
            sig_loss_target = sigreg_loss(z_target, lambd=1.0)
            total_sig_loss = sig_loss_pred + sig_loss_target
            loss = mse_loss + sigreg_weight * total_sig_loss
            
        total_loss += loss.item() * x.size(0)
        total_mse += mse_loss.item() * x.size(0)
        total_sig += total_sig_loss.item() * x.size(0)
        
        all_representations.append(z_context)
        
    all_representations = torch.cat(all_representations, dim=0)
    
    # Calculate Effective Rank (high dimensional check)
    eff_rank = calculate_effective_rank(all_representations)
    
    n = len(dataloader.dataset)
    return total_loss / n, total_mse / n, total_sig / n, eff_rank

def main():
    parser = argparse.ArgumentParser(description="ECG-JEPA Self-Supervised Pretraining")
    parser.add_argument("--data_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (from Playbook)")
    parser.add_argument("--sigreg_weight", type=float, default=20.0, help="SIGReg Loss scaling parameter (from Playbook)")
    parser.add_argument("--ema_decay", type=float, default=0.996, help="Target encoder EMA decay constant")
    parser.add_argument("--latent_dim", type=int, default=256, help="Representation dimension")
    parser.add_argument("--save_path", type=str, default="checkpoints/ecg_jepa_encoder.pt", help="Path to save representation weights")
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print("🧠 ECG-JEPA Self-Supervised Pretraining Pipeline")
    print("--------------------------------------------------")
    
    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    # Create save directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 1. Load Metadata & Datasets
    try:
        df, superclasses = load_ptbxl_metadata(args.data_dir)
        print(f"Metadata loaded successfully. Diagnostic superclasses: {superclasses}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
        
    train_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=list(range(1, 9)), augment=True)
    val_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=[9], augment=False)
    
    # 2. Dataloaders (with High-Performance Playbook threading configuration)
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
    
    print(f"Splits initialized:")
    print(f"  - Pretrain Train (Folds 1-8): {len(train_dataset)} records ({len(train_loader)} batches)")
    print(f"  - Pretrain Val (Fold 9):      {len(val_dataset)} records ({len(val_loader)} batches)")
    
    # 3. Instantiate model
    context_encoder = ECGEncoder1D(in_channels=12, latent_dim=args.latent_dim)
    model = ECGJEPA(
        context_encoder=context_encoder,
        latent_dim=args.latent_dim,
        ema_decay=args.ema_decay
    )
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Linear Warmup scheduler (5 epochs of warmup as per Playbook)
    # Cosine Annealing decay afterwards
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
            
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    # 4. Pretraining Loop
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_mse, train_sig = pretrain_epoch(
            model, train_loader, optimizer, args.sigreg_weight, device, scaler
        )
        # Val
        val_loss, val_mse, val_sig, eff_rank = evaluate_pretraining(
            model, val_loader, args.sigreg_weight, device
        )
        
        scheduler.step()
        
        # Display Epoch log
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, SIG: {train_sig:.4f}) | "
              f"Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, SIG: {val_sig:.4f}) | "
              f"Effective Rank: {eff_rank:.1f}")
              
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save strictly the representation context encoder weights
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": context_encoder.state_dict(),
                "latent_dim": args.latent_dim,
                "effective_rank": eff_rank,
                "val_loss": val_loss
            }, args.save_path)
            
    print(f"\nPretraining successfully completed!")
    print(f"Best ECG-JEPA Context Encoder saved to: {args.save_path}")

if __name__ == "__main__":
    main()
