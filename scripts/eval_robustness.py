import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import load_ptbxl_metadata, PTBXLDataset
from src.models.encoder import ECGEncoder1D

# Custom Robustness evaluation Dataset that applies specific systematic physical corruptions
class CorruptedDataset(Dataset):
    def __init__(self, clean_dataset, corruption_type):
        self.dataset = clean_dataset
        self.corruption_type = corruption_type
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Retrieve clean waveform and label
        x, y = self.dataset[idx]
        x = x.clone()
        
        # Apply specific systematic physical clinical corruption
        if self.corruption_type == "clean":
            pass
            
        elif self.corruption_type == "missing_3_leads":
            # Set 3 random leads to 0 completely
            drop_leads = torch.randperm(12)[:3]
            x[drop_leads] = 0.0
            
        elif self.corruption_type == "missing_6_leads":
            # Set 6 random leads (50% of connections!) to 0 completely
            drop_leads = torch.randperm(12)[:6]
            x[drop_leads] = 0.0
            
        elif self.corruption_type == "high_freq_noise":
            # Severe high-frequency Gaussian noise (typical clinical electromyogram noise)
            noise = 0.08 * torch.randn_like(x)
            x = x + noise
            
        elif self.corruption_type == "baseline_drift":
            # Severe baseline breathing wander (slow low-frequency sinusoid)
            length = x.shape[-1]
            t = torch.linspace(0, 10.0, length)
            freq = 1.25 # 1.25 Hz breathing cycles
            amp = 0.35  # 0.35 mV high drift
            wander = amp * torch.sin(2 * torch.pi * freq * t)
            x = x + wander.unsqueeze(0)
            
        elif self.corruption_type == "motion_artifact":
            # Localized transient burst (high amplitude motion shakes) on 3 random leads
            corrupt_leads = torch.randperm(12)[:3]
            for lead in corrupt_leads:
                # Random 200ms segment (20 timesteps at 100Hz)
                start_idx = torch.randint(0, x.shape[-1] - 20, (1,)).item()
                x[lead, start_idx : start_idx + 20] += 0.5 * torch.randn(20)
                
        return x, y

@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    """Extracts latent embeddings and labels."""
    encoder.eval()
    all_embeddings = []
    all_labels = []
    for x, y in dataloader:
        x = x.to(device)
        z = encoder(x)
        all_embeddings.append(z.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

def evaluate_retrieval_precision(train_embeds, train_labels, test_embeds, test_labels, k=3):
    """Calculates Mean Retrieval Precision @ k using multi-label matching."""
    # L2 normalize embeddings for fast cosine similarity
    train_norms = np.linalg.norm(train_embeds, axis=1, keepdims=True) + 1e-8
    test_norms = np.linalg.norm(test_embeds, axis=1, keepdims=True) + 1e-8
    
    norm_train = train_embeds / train_norms
    norm_test = test_embeds / test_norms
    
    # Cosine Similarity: [NumTest, NumTrain]
    similarity = norm_test @ norm_train.T
    
    precisions = []
    for i in range(len(test_embeds)):
        query_labels = test_labels[i]
        top_k_indices = np.argsort(similarity[i])[::-1][:k]
        neighbor_labels = train_labels[top_k_indices]
        
        relevance_counts = []
        for n_idx in range(k):
            # A retrieved neighbor is relevant if it shares at least one clinical statement with the query
            intersection = np.logical_and(query_labels, neighbor_labels[n_idx])
            if np.any(intersection) or (np.sum(query_labels) == 0 and np.sum(neighbor_labels[n_idx]) == 0):
                relevance_counts.append(1.0)
            else:
                relevance_counts.append(0.0)
        precisions.append(np.mean(relevance_counts))
        
    return np.mean(precisions)

def draw_robustness_chart(scenarios, precisions):
    """Draws a beautiful horizontal ASCII bar chart showing retrieval degradation."""
    print("\n📊 Case Retrieval Precision @ 3 under Clinical Corruptions")
    print("--------------------------------------------------")
    for s, p in zip(scenarios, precisions):
        percent = p * 100
        # Build bar (length proportional to precision, e.g. 1% = 0.4 bar columns)
        bar_len = int(p * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"{s:<20} | {bar} {percent:.2f}%")
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="CardioRep Retrieval Robustness Benchmarking")
    parser.add_argument("--data_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_sig1.pt", help="Path to pretrained JEPA weights")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding extraction")
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print("🩺 CardioRep Case Retrieval Robustness Benchmarking")
    print("--------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    # 1. Load Metadata & Datasets
    df, superclasses = load_ptbxl_metadata(args.data_dir)
    
    # Clean train dataset (Folds 1-8) serving as search database index
    train_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=list(range(1, 9)), augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4 if device.type == "cuda" else 0, pin_memory=True)
    
    # Clean test dataset (Fold 10)
    test_dataset_clean = PTBXLDataset(args.data_dir, df, superclasses, fold_list=[10], augment=False)
    
    # 2. Instantiate and load pre-trained CardioRep encoder
    encoder = ECGEncoder1D(in_channels=12, latent_dim=256).to(device)
    checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    
    # 3. Extract Clean Search Index Embeddings
    print("Extracting clean training database index embeddings...")
    train_embeds, train_labels = extract_embeddings(encoder, train_loader, device)
    
    scenarios = [
        "clean",
        "missing_3_leads",
        "missing_6_leads",
        "high_freq_noise",
        "baseline_drift",
        "motion_artifact"
    ]
    
    scenario_display_names = [
        "Clean Baseline",
        "Missing 3 Leads",
        "Missing 6 Leads (50%)",
        "High-Freq Noise",
        "Baseline Drift",
        "Motion Artifacts"
    ]
    
    precisions_k3 = []
    precisions_k5 = []
    
    # Evaluate across systematic clinical corruptions
    for scenario, name in zip(scenarios, scenario_display_names):
        print(f"\n⚡ Evaluating scenario: '{name}'...")
        corrupt_dataset = CorruptedDataset(test_dataset_clean, corruption_type=scenario)
        test_loader = DataLoader(corrupt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4 if device.type == "cuda" else 0, pin_memory=True)
        
        # Extract corrupted query embeddings
        test_embeds, test_labels = extract_embeddings(encoder, test_loader, device)
        
        p3 = evaluate_retrieval_precision(train_embeds, train_labels, test_embeds, test_labels, k=3)
        p5 = evaluate_retrieval_precision(train_embeds, train_labels, test_embeds, test_labels, k=5)
        
        precisions_k3.append(p3)
        precisions_k5.append(p5)
        
        print(f"  - Mean Precision @ 3: {p3:.4f}")
        print(f"  - Mean Precision @ 5: {p5:.4f}")
        
    # Print Comparative Robustness Report
    print("\n" + "=" * 55)
    print("📋 CLINICAL ROBUSTNESS AND RETRIEVAL DEGRADATION SUMMARY")
    print("=" * 55)
    print(f"{'Clinical Scenario':<22} | {'Precision @ 3':<13} | {'Precision @ 5':<13} | {'Degradation'}")
    print("-" * 60)
    baseline_p3 = precisions_k3[0]
    for i, name in enumerate(scenario_display_names):
        deg = (precisions_k3[i] - baseline_p3) / baseline_p3 * 100
        deg_str = f"{deg:+.2f}%" if i > 0 else "0.00% (Base)"
        print(f"{name:<22} | {precisions_k3[i]:.4f}        | {precisions_k5[i]:.4f}        | {deg_str}")
    print("=" * 55)
    
    # Plot ASCII visual chart
    draw_robustness_chart(scenario_display_names, precisions_k3)

if __name__ == "__main__":
    main()
