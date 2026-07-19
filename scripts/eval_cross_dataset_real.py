import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import wfdb

from src.models.encoder import ECGEncoder1D

# SNOMED-CT mappings to our 5 diagnostic superclasses (from PhysioNet Challenge 2020)
SNOMED_MAPPING = {
    # 1. Normal (NORM)
    "426783006": "NORM", "428750005": "NORM", "164947007": "NORM",
    # 2. Conduction Disturbance (CD)
    "164867002": "CD", "164865005": "CD", "164909002": "CD", "164951009": "CD", 
    "164873001": "CD", "270492004": "CD", "427544004": "CD", "28189009": "CD",
    # 3. Myocardial Infarction (MI)
    "164861001": "MI", "59118001": "MI", "164865005": "MI", "713427006": "MI",
    # 4. ST/T Changes (STTC)
    "164931005": "STTC", "55930002": "STTC", "164934002": "STTC", "251146004": "STTC",
    # 5. Hypertrophy (HYP)
    "251146004": "HYP", "89792004": "HYP", "164873001": "HYP"
}

def parse_snomed_labels_from_header(hea_path):
    """Parses SNOMED-CT clinical diagnostic codes from a .hea file and maps them to NORM, MI, STTC, CD, HYP."""
    superclasses = ["NORM", "MI", "STTC", "CD", "HYP"]
    label = np.zeros(len(superclasses), dtype=np.float32)
    
    if not os.path.exists(hea_path):
        return None
        
    snomed_codes = []
    with open(hea_path, "r") as f:
        for line in f:
            if line.startswith("#Dx:") or line.startswith("# Dx:"):
                # Extract comma-separated codes
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    codes = parts[1].strip().split(",")
                    snomed_codes.extend([c.strip() for c in codes])
                    
    # Map to superclasses
    mapped_count = 0
    for code in snomed_codes:
        if code in SNOMED_MAPPING:
            s_class = SNOMED_MAPPING[code]
            s_class_idx = superclasses.index(s_class)
            label[s_class_idx] = 1.0
            mapped_count += 1
            
    # Fallback: if we found codes but they didn't map to our 5 classes, skip
    if len(snomed_codes) > 0 and mapped_count == 0:
        return None
        
    return label

class RealTransferDataset(Dataset):
    def __init__(self, record_paths, labels, target_length=1000):
        self.record_paths = record_paths
        self.labels = labels
        self.target_length = target_length
        
    def __len__(self):
        return len(self.record_paths)
        
    def __getitem__(self, idx):
        path = self.record_paths[idx]
        
        # Load raw signal
        signal, meta = wfdb.rdsamp(path)
        fs = meta["fs"]
        
        raw_x = signal.astype(np.float32).T
        
        # Downsample 500Hz -> 100Hz
        if fs == 500:
            x = raw_x[:, ::5]
        else:
            x = raw_x
            
        # Align length to exactly 10 seconds
        if x.shape[1] < self.target_length:
            pad_len = self.target_length - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_len)), "constant")
        else:
            x = x[:, :self.target_length]
            
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return x_tensor, y_tensor

@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    """Extracts representation embeddings from frozen encoder."""
    encoder.eval()
    embeddings = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        z = encoder(x)
        embeddings.append(z.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)

def calculate_auroc(y_true, y_probs):
    """Calculates macro AUROC score robustly."""
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    auc_scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            auc_scores.append(roc_auc_score(y_true[:, c], y_probs[:, c]))
        else:
            auc_scores.append(0.5)
    return np.mean(auc_scores)

def train_linear_probe(train_embeds, train_labels, test_embeds, test_labels, num_classes, epochs=150, lr=1e-3):
    """Trains a multi-label linear probe classifier on top of frozen embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier = nn.Linear(train_embeds.shape[1], num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    
    x_train, y_train = torch.from_numpy(train_embeds).to(device), torch.from_numpy(train_labels).to(device)
    x_test, y_test = torch.from_numpy(test_embeds).to(device), torch.from_numpy(test_labels).to(device)
    
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(x_test)
        probs = torch.sigmoid(test_logits).cpu().numpy()
        
    return calculate_auroc(test_labels, probs)

def scan_real_dataset_with_labels(dataset_dir, max_records=500):
    """Scans the directory of target records, parses SNOMED-CT labels, and returns clean records."""
    record_paths = []
    labels = []
    
    # List all headers
    hea_files = [f for f in os.listdir(dataset_dir) if f.endswith(".hea")]
    print(f"Scanning '{dataset_dir}': Found {len(hea_files)} total records...")
    
    count = 0
    for h in hea_files:
        record_name = h.split(".")[0]
        abs_hea_path = os.path.join(dataset_dir, h)
        abs_record_path = os.path.join(dataset_dir, record_name)
        
        # Parse SNOMED clinical labels
        label = parse_snomed_labels_from_header(abs_hea_path)
        if label is not None:
            record_paths.append(abs_record_path)
            labels.append(label)
            count += 1
            if count >= max_records:
                break
                
    print(f"  - Extracted {len(record_paths)} clean clinical records with valid ground-truth mappings.")
    return record_paths, labels

def main():
    parser = argparse.ArgumentParser(description="CardioRep Real Cross-Dataset Transfer Benchmark")
    parser.add_argument("--chapman_dir", type=str, default="data/chapman_real", help="Path to real Chapman database")
    parser.add_argument("--cpsc_dir", type=str, default="data/cpsc2018_real", help="Path to real CPSC2018 database")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_sig1.pt", help="Path to pretrained CardioRep weights")
    parser.add_argument("--epochs", type=int, default=150, help="Linear probing training epochs")
    parser.add_argument("--max_records", type=int, default=500, help="Max records to load per dataset")
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print("🚀 CardioRep REAL Cross-Dataset Generalization Benchmark")
    print("--------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup encoders
    encoder_jepa = ECGEncoder1D(in_channels=12, latent_dim=256).to(device)
    checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)
    encoder_jepa.load_state_dict(checkpoint["encoder_state_dict"])
    
    encoder_rand = ECGEncoder1D(in_channels=12, latent_dim=256).to(device)
    
    target_datasets = {
        "Chapman (Real)": args.chapman_dir,
        "CPSC2018 (Real)": args.cpsc_dir
    }
    
    results = {}
    
    for name, directory in target_datasets.items():
        print(f"\n==================================================")
        print(f"📁 Evaluating on REAL Dataset: {name}")
        print(f"==================================================")
        
        record_paths, labels = scan_real_dataset_with_labels(directory, max_records=args.max_records)
        
        if len(record_paths) == 0:
            print(f"Skipping {name}: no records parsed successfully.")
            continue
            
        # 80/20 train/test split
        split_idx = int(0.8 * len(record_paths))
        train_paths, train_labels = record_paths[:split_idx], labels[:split_idx]
        test_paths, test_labels = record_paths[split_idx:], labels[split_idx:]
        
        train_ds = RealTransferDataset(train_paths, train_labels)
        test_ds = RealTransferDataset(test_paths, test_labels)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        # A. Evaluate Frozen Random Init Encoder
        print("  - Evaluating Frozen Random Init Encoder...")
        r_train_emb, r_train_lbl = extract_embeddings(encoder_rand, train_loader, device)
        r_test_emb, r_test_lbl = extract_embeddings(encoder_rand, test_loader, device)
        rand_auroc = train_linear_probe(r_train_emb, r_train_lbl, r_test_emb, r_test_lbl, num_classes=5, epochs=args.epochs)
        print(f"    -> Random Init Transfer Test AUROC: {rand_auroc:.4f}")
        
        # B. Evaluate CardioRep Pretrained
        print("  - Evaluating Frozen CardioRep Pretrained Encoder...")
        j_train_emb, j_train_lbl = extract_embeddings(encoder_jepa, train_loader, device)
        j_test_emb, j_test_lbl = extract_embeddings(encoder_jepa, test_loader, device)
        jepa_auroc = train_linear_probe(j_train_emb, j_train_lbl, j_test_emb, j_test_lbl, num_classes=5, epochs=args.epochs)
        print(f"    -> CardioRep Transfer Test AUROC: {jepa_auroc:.4f}")
        
        results[name] = {
            "random_auroc": rand_auroc,
            "cardiorep_auroc": jepa_auroc,
            "absolute_gain": jepa_auroc - rand_auroc
        }
        
    # Print Final Summary Comparison
    print("\n" + "=" * 55)
    print("📊 CARDIOREP REAL CROSS-DATASET GENERALIZATION SUMMARY")
    print("=" * 55)
    print(f"{'Target Dataset':<18} | {'Random Init':<12} | {'CardioRep':<12} | {'Absolute Gain'}")
    print("-" * 60)
    for name, scores in results.items():
        print(f"{name:<18} | {scores['random_auroc']:.4f}       | {scores['cardiorep_auroc']:.4f}       | {scores['absolute_gain']:+.4f}")
    print("=" * 55)

if __name__ == "__main__":
    main()
