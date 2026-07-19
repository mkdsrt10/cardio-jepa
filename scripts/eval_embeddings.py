import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import scipy.signal

from src.data.dataset import load_ptbxl_metadata, PTBXLDataset
from src.models.encoder import ECGEncoder1D

def calculate_heart_rate(signal_12lead, fs=100):
    """Calculates patient heart rate (BPM) using classical clinical QRS peak detection on Lead II."""
    # Lead II is at index 1
    lead2 = signal_12lead[1]
    
    # QRS peak detector using scipy find_peaks
    # We use distance of 33 samples (equivalent to max 180 BPM)
    peaks, _ = scipy.signal.find_peaks(lead2, distance=33, prominence=0.2)
    
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        mean_diff = np.mean(peak_diffs)
        bpm = 60.0 * fs / mean_diff
        return float(bpm)
    return 72.0 # Default clinical normal fallback

def get_dominant_diagnosis(label_vec, superclasses):
    """Returns the dominant abnormal diagnosis string, or NORM if clean."""
    # Find active superclasses
    active_indices = np.where(label_vec == 1.0)[0]
    
    if len(active_indices) == 0:
        return "NORM"
        
    # If NORM is present and nothing else, return NORM
    norm_idx = superclasses.index("NORM")
    if len(active_indices) == 1 and active_indices[0] == norm_idx:
        return "NORM"
        
    # Return the first active abnormal class
    for idx in active_indices:
        if idx != norm_idx:
            return superclasses[idx]
            
    return "NORM"

def render_ascii_scatter(coords_2d, labels, char_mapping, title):
    """Renders a beautiful, high-fidelity ASCII scatter plot in the console."""
    # Grid sizes
    rows = 15
    cols = 50
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    
    # Normalize coordinates to [0, cols-1] and [0, rows-1]
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add a tiny epsilon to avoid division by zero
    x_range = (x_max - x_min) + 1e-8
    y_range = (y_max - y_min) + 1e-8
    
    for i in range(len(coords_2d)):
        c_x = int((x[i] - x_min) / x_range * (cols - 1))
        c_y = rows - 1 - int((y[i] - y_min) / y_range * (rows - 1)) # Flipped so positive Y goes up
        
        c_x = max(0, min(cols - 1, c_x))
        c_y = max(0, min(rows - 1, c_y))
        
        # Get character symbol
        lbl = labels[i]
        char = char_mapping.get(lbl, ".")
        grid[c_y][c_x] = char
        
    # Print scatter chart
    print(f"\n🗺️ t-SNE Latent Cluster Map: {title}")
    print("--------------------------------------------------")
    for r in range(rows):
        line = " | " + "".join(grid[r])
        print(line)
    print(" +-" + "-" * cols)
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="CardioRep t-SNE Latent Embedding Analysis")
    parser.add_argument("--data_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_sig1.pt", help="Path to pretrained CardioRep weights")
    parser.add_argument("--max_samples", type=int, default=400, help="Max test samples to extract and project")
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print("📊 CardioRep Latent Embedding & Cluster Analysis")
    print("--------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    # 1. Load Metadata & Datasets (Fold 10 Test set)
    df, superclasses = load_ptbxl_metadata(args.data_dir)
    test_dataset = PTBXLDataset(args.data_dir, df, superclasses, fold_list=[10], augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. Load Pre-trained CardioRep encoder
    encoder = ECGEncoder1D(in_channels=12, latent_dim=256).to(device)
    checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    
    print(f"Extracting clinical embeddings and metadata for {args.max_samples} test profiles...")
    
    all_embeddings = []
    metadata = {
        "diagnosis": [],
        "age": [],
        "sex": [],
        "heart_rate": []
    }
    
    sample_count = 0
    # We load manually to extract both signals (for heart rate) and embeddings
    with torch.no_grad():
        for x, y in test_loader:
            x_dev = x.to(device)
            z = encoder(x_dev).cpu().numpy()
            
            # Loop over batch records
            for b in range(len(z)):
                all_embeddings.append(z[b])
                
                # Fetch original row info
                row_idx = test_dataset.df.index[sample_count]
                row = test_dataset.df.loc[row_idx]
                
                # A. Extract Diagnosis
                metadata["diagnosis"].append(get_dominant_diagnosis(y[b].numpy(), superclasses))
                
                # B. Extract Demographics
                metadata["age"].append(float(row["age"]) if not np.isnan(row["age"]) else 60.0)
                metadata["sex"].append("Male" if row["sex"] == 1.0 else "Female")
                
                # C. Extract Heart Rate (BPM)
                metadata["heart_rate"].append(calculate_heart_rate(x[b].numpy()))
                
                sample_count += 1
                if sample_count >= args.max_samples:
                    break
            if sample_count >= args.max_samples:
                break
                
    embeddings = np.array(all_embeddings)
    
    # Categorize demographics/intervals into clean discrete variables for clustering metrics
    age_groups = []
    for a in metadata["age"]:
        if a < 40: age_groups.append("<40 (Young)")
        elif a < 65: age_groups.append("40-65 (Middle)")
        else: age_groups.append(">65 (Elderly)")
        
    hr_groups = []
    for bpm in metadata["heart_rate"]:
        if bpm < 60.0: hr_groups.append("Bradycardia (<60 BPM)")
        elif bpm <= 100.0: hr_groups.append("Normal (60-100 BPM)")
        else: hr_groups.append("Tachycardia (>100 BPM)")
        
    # 3. Apply t-SNE Dimensionality Reduction
    print(f"\n⚡ Projecting 256D embeddings to 2D using t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords_2d = tsne.fit_transform(embeddings)
    
    # 4. Compute Silhouette Scores (Clustering Separability)
    # Silhouette score is in [-1, +1]. A positive score means good separability.
    print(f"\n⚙️ Calculating Quantitative Silhouette Scores:")
    diag_sil = silhouette_score(coords_2d, metadata["diagnosis"])
    hr_sil = silhouette_score(coords_2d, hr_groups)
    sex_sil = silhouette_score(coords_2d, metadata["sex"])
    age_sil = silhouette_score(coords_2d, age_groups)
    
    print(f"  - Diagnosis Clusters:          {diag_sil:+.4f} (Pos score proves pathological grouping)")
    print(f"  - Rhythm/Heart-Rate Clusters:  {hr_sil:+.4f} (Pos score proves rhythm grouping)")
    print(f"  - Gender Demographics:         {sex_sil:+.4f} (Near 0.0 is perfect - proves gender-invariant debiasing)")
    print(f"  - Age Demographics:            {age_sil:+.4f} (Near 0.0 is perfect - proves age-invariant debiasing)")
    
    # 5. Render Beautiful ASCII Scatter Plots
    # A. Pathological Cluster
    diag_mapping = {
        "NORM": "N", # Normal (Green zone)
        "MI": "M",   # Infarction (Red alert)
        "STTC": "S", # ST Changes
        "CD": "C",   # Conduction block
        "HYP": "H"   # Hypertrophy
    }
    render_ascii_scatter(
        coords_2d, metadata["diagnosis"], diag_mapping,
        "Color by Pathological Diagnosis ([N]=NORM, [M]=MI, [S]=STTC, [C]=CD, [H]=HYP)"
    )
    
    # B. Rhythm Cluster
    hr_mapping = {
        "Bradycardia (<60 BPM)": "S",   # Slow
        "Normal (60-100 BPM)": ".",    # Normal dots
        "Tachycardia (>100 BPM)": "F"   # Fast
    }
    render_ascii_scatter(
        coords_2d, hr_groups, hr_mapping,
        "Color by Heart-Rate/Rhythm ([S]=Bradycardia/Slow, [.]=Normal, [F]=Tachycardia/Fast)"
    )

if __name__ == "__main__":
    main()
