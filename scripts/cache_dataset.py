import os
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from src.data.dataset import load_ptbxl_metadata

def cache_signals(data_dir, output_file, target_length=1000):
    """Loads all 12-lead ECG signals from individual WFDB records and caches them into a single binary numpy file."""
    print("--------------------------------------------------")
    # Real path resolution
    data_dir = os.path.abspath(data_dir)
    output_file = os.path.abspath(output_file)
    print(f"📦 Starting ECG Dataset caching pipeline")
    print(f"Dataset Dir: {data_dir}")
    print(f"Output File: {output_file}")
    
    # Load metadata
    df, _ = load_ptbxl_metadata(data_dir)
    num_records = len(df)
    print(f"Loaded database. Processing {num_records} records...")
    
    # Pre-allocate high-performance float32 array
    # Shape: [NumRecords, Leads, Length]
    all_signals = np.zeros((num_records, 12, target_length), dtype=np.float32)
    
    # We map ecg_id to index in our array. 
    # PTB-XL ecg_id starts at 1, so index is ecg_id - 1.
    for idx, (ecg_id, row) in enumerate(tqdm(df.iterrows(), total=num_records)):
        filename = row["filename_lr"]
        abs_path = os.path.join(data_dir, filename)
        
        # Load waveform
        try:
            signal, _ = wfdb.rdsamp(abs_path)
            # Signal shape: [Length, Leads] -> Transpose to [Leads, Length]
            x = signal.astype(np.float32).T
            
            # Align length to target_length
            if x.shape[1] < target_length:
                pad_len = target_length - x.shape[1]
                x = np.pad(x, ((0, 0), (0, pad_len)), "constant")
            elif x.shape[1] > target_length:
                x = x[:, :target_length]
                
            all_signals[idx] = x
        except Exception as e:
            print(f"Error loading record {ecg_id} at {abs_path}: {e}")
            
    # Save cache file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, all_signals)
    print(f"🎉 Successfully cached {num_records} signals to {output_file} (Size: {all_signals.nbytes / (1024**2):.2f} MB)")
    print("--------------------------------------------------")

if __name__ == "__main__":
    cache_signals("data/ptbxl", "data/ptbxl_cached_signals.npy")
