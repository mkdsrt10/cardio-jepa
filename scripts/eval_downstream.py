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
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_preds = (y_probs >= 0.5).astype(int)
    
    auc_scores = []
    num_classes = y_true.shape[1]
    for c in range(num_classes):
        if len(np.unique(y_true[:, c])) > 1:
            auc = roc_auc_score(y_true[:, c], y_probs[:, c])
            auc_scores.append(auc)
        else:
            auc_scores.append(0.5)
            
    macro_auroc = np.mean(auc_scores)
    macro_f1 = f1_score(y_true, y_preds, average="macro", zero_division=0)
    
    return macro_auroc, macro_f1, auc_scores

@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    """Extracts representation embeddings and labels for an entire dataloader."""
    encoder.eval()
    all_embeddings = []
    all_labels = []
    
    for x, y in dataloader:
        x = x.to(device)
        # Extract representation
        z = encoder(x)
        all_embeddings.append(z.cpu().numpy())
        all_labels.append(y.numpy())
        
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

def train_linear_probe(train_embeds, train_labels, val_embeds, val_labels, test_embeds, test_labels, num_classes, epochs=15, lr=1e-3):
    """Trains a multi-label linear classifier probe on top of frozen representation embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple linear probe model
    classifier = nn.Linear(train_embeds.shape[1], num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    
    # Prepare PyTorch Tensors
    x_train, y_train = torch.from_numpy(train_embeds).to(device), torch.from_numpy(train_labels).to(device)
    x_val, y_val = torch.from_numpy(val_embeds).to(device), torch.from_numpy(val_labels).to(device)
    x_test, y_test = torch.from_numpy(test_embeds).to(device), torch.from_numpy(test_labels).to(device)
    
    best_val_auroc = 0.0
    best_weights = None
    
    # Fast linear training loop
    for epoch in range(1, epochs + 1):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        # Eval
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(x_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auroc, _, _ = calculate_metrics(val_labels, val_probs)
            
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_weights = classifier.state_dict().copy()
                
    # Load best weights
    classifier.load_state_dict(best_weights)
    classifier.eval()
    
    with torch.no_grad():
        test_logits = classifier(x_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_auroc, test_f1, class_aucs = calculate_metrics(test_labels, test_probs)
        
    return test_auroc, test_f1, class_aucs

def evaluate_retrieval_precision(train_embeds, train_labels, test_embeds, test_labels, k=5):
    """Evaluates case retrieval precision using Cosine Similarity of JEPA embeddings."""
    print(f"\n🔍 Evaluating Case Retrieval Performance (k={k})...")
    
    # L2 normalize embeddings for cosine similarity
    train_norms = np.linalg.norm(train_embeds, axis=1, keepdims=True) + 1e-8
    test_norms = np.linalg.norm(test_embeds, axis=1, keepdims=True) + 1e-8
    
    norm_train = train_embeds / train_norms
    norm_test = test_embeds / test_norms
    
    # Compute Cosine Similarity Matrix: [NumTest, NumTrain]
    similarity = norm_test @ norm_train.T
    
    precisions = []
    # For each query in the test set
    for i in range(len(test_embeds)):
        query_labels = test_labels[i]
        
        # Get indices of top-k most similar records in the train set
        top_k_indices = np.argsort(similarity[i])[::-1][:k]
        
        # Retrieve neighbor labels
        neighbor_labels = train_labels[top_k_indices]
        
        # Multi-label intersection precision:
        # A retrieved neighbor is "relevant" if it shares at least one positive label with the query.
        # For NORM queries, they must share the NORM label.
        relevance_counts = []
        for n_idx in range(k):
            intersection = np.logical_and(query_labels, neighbor_labels[n_idx])
            if np.any(intersection) or (np.sum(query_labels) == 0 and np.sum(neighbor_labels[n_idx]) == 0):
                relevance_counts.append(1.0)
            else:
                relevance_counts.append(0.0)
                
        precisions.append(np.mean(relevance_counts))
        
    mean_precision = np.mean(precisions)
    print(f"  - Mean retrieval precision @ {k}: {mean_precision:.4f}")
    return mean_precision

def main():
    parser = argparse.ArgumentParser(description="ECG-JEPA Downstream Evaluation")
    parser.add_argument("--data_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/ecg_jepa_encoder.pt", help="Path to pre-trained JEPA encoder")
    parser.add_argument("--epochs", type=int, default=100, help="Linear probing epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding extraction")
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print("🎯 ECG-JEPA Downstream Linear Probing & Case Retrieval")
    print("--------------------------------------------------")
    
    # Check for pre-trained weights
    if not os.path.exists(args.jepa_path):
        print(f"Error: Pre-trained JEPA weights not found at '{args.jepa_path}'")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Metadata & Datasets
    df, superclasses = load_ptbxl_metadata(args.data_dir)
    train_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=list(range(1, 9)), augment=False)
    val_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=[9], augment=False)
    test_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=[10], augment=False)
    
    # Dataloaders (no shuffling, high performance)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4 if device.type == "cuda" else 0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4 if device.type == "cuda" else 0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4 if device.type == "cuda" else 0, pin_memory=True)
    
    # 2. Instantiate and load pre-trained JEPA Context Encoder
    encoder = ECGEncoder1D(in_channels=12, latent_dim=256)
    checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(device)
    
    print("Extracting frozen representation vectors...")
    train_embeds, train_labels = extract_embeddings(encoder, train_loader, device)
    val_embeds, val_labels = extract_embeddings(encoder, val_loader, device)
    test_embeds, test_labels = extract_embeddings(encoder, test_loader, device)
    
    print(f"Embeddings extracted:")
    print(f"  - Train: {train_embeds.shape}")
    print(f"  - Val:   {val_embeds.shape}")
    print(f"  - Test:  {test_embeds.shape}")
    
    # 3. Train Downstream Linear Classifier Probe
    print(f"\n⚡ Training linear probing classifier (BCE loss, epochs={args.epochs})...")
    test_auroc, test_f1, class_aucs = train_linear_probe(
        train_embeds, train_labels, val_embeds, val_labels, test_embeds, test_labels,
        num_classes=len(superclasses), epochs=args.epochs, lr=1e-3
    )
    
    print("--------------------------------------------------")
    print(f"📋 DOWNSTREAM LINEAR PROBING RESULTS (Fold 10)")
    print("--------------------------------------------------")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1:    {test_f1:.4f}")
    print("Class-wise AUROC scores:")
    for i, s_class in enumerate(superclasses):
        print(f"  - {s_class}: {class_aucs[i]:.4f}")
    print("--------------------------------------------------")
    
    # 4. Similar Case Retrieval Engine Evaluation
    evaluate_retrieval_precision(train_embeds, train_labels, test_embeds, test_labels, k=3)
    evaluate_retrieval_precision(train_embeds, train_labels, test_embeds, test_labels, k=5)
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
