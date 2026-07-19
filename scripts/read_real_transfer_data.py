import os
import numpy as np
import torch
import wfdb

from src.models.encoder import ECGEncoder1D

def read_and_embed_real_ecg(record_path, encoder, device, target_length=1000):
    """Loads a real clinical 12-lead ECG, downsamples it from 500Hz to 100Hz, and extracts its CardioRep embedding."""
    # 1. Load raw record using wfdb
    # This automatically reads the .hea header and the respective binary signal file (.mat or .dat)
    signal, meta = wfdb.rdsamp(record_path)
    fs = meta["fs"]
    leads = meta["sig_name"]
    
    # Shape: [Length, Leads] -> Transpose to [Leads, Length]
    raw_x = signal.astype(np.float32).T
    
    # 2. Downsample from 500Hz to 100Hz on-the-fly to align with pretraining receptive field
    if fs == 500:
        x = raw_x[:, ::5] # Take every 5th sample
    else:
        x = raw_x
        
    # 3. Align length to exactly 1000 samples (10 seconds)
    if x.shape[1] < target_length:
        pad_len = target_length - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_len)), "constant")
    else:
        x = x[:, :target_length]
        
    # 4. Extract embedding using frozen pretrained CardioRep encoder
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device) # [1, 12, 1000]
    
    with torch.no_grad():
        z = encoder(x_tensor).squeeze(0).cpu().numpy()
        
    return z, x, fs, leads

def main():
    print("--------------------------------------------------")
    # Real path resolution
    chapman_dir = "data/chapman_real"
    cpsc_dir = "data/cpsc2018_real"
    
    print("📂 Loading Real Hospital ECG Databases...")
    print(f"Chapman Real Database: {os.path.abspath(chapman_dir)}")
    print(f"CPSC2018 Real Database: {os.path.abspath(cpsc_dir)}")
    print("--------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained CardioRep encoder
    encoder = ECGEncoder1D(in_channels=12, latent_dim=256).to(device)
    checkpoint = torch.load("checkpoints/jepa_sig1.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    print("Pre-trained CardioRep encoder weights loaded successfully!")
    
    # A. Read a real clinical record from Chapman University Database
    print("\n[Clinical Record 1] Loading from Chapman (Shaoxing Hospital)...")
    # Chapman files are named like 'A0001', 'A0002', etc. Let's find first available file
    chapman_files = [f.split(".")[0] for f in os.listdir(chapman_dir) if f.endswith(".hea")]
    chapman_record = os.path.join(chapman_dir, chapman_files[0])
    
    z_c, x_c, fs_c, leads_c = read_and_embed_real_ecg(chapman_record, encoder, device)
    print(f"  - Record Name:      {chapman_record}")
    print(f"  - Original Freq:    {fs_c} Hz")
    print(f"  - Leads Present:    {leads_c}")
    print(f"  - Waveform Shape:   {x_c.shape} (Downsampled to 100Hz)")
    print(f"  - Latent Vector z:  Shape {z_c.shape} | Mean value: {z_c.mean():.4f}")
    
    # B. Read a real clinical record from CPSC2018 China Database
    print("\n[Clinical Record 2] Loading from CPSC2018 (China Physiological Challenge)...")
    cpsc_files = [f.split(".")[0] for f in os.listdir(cpsc_dir) if f.endswith(".hea")]
    cpsc_record = os.path.join(cpsc_dir, cpsc_files[0])
    
    z_p, x_p, fs_p, leads_p = read_and_embed_real_ecg(cpsc_record, encoder, device)
    print(f"  - Record Name:      {cpsc_record}")
    print(f"  - Original Freq:    {fs_p} Hz")
    print(f"  - Leads Present:    {leads_p}")
    print(f"  - Waveform Shape:   {x_p.shape} (Downsampled to 100Hz)")
    print(f"  - Latent Vector z:  Shape {z_p.shape} | Mean value: {z_p.mean():.4f}")
    print("--------------------------------------------------")
    print("🎉 Successfully loaded and processed actual, real-world clinical hospital waveforms end-to-end!")

if __name__ == "__main__":
    main()
