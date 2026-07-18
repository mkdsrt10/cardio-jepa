import os
import argparse
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import wfdb

def generate_synthetic_ecg(length=1000, num_leads=12, heart_rate=75, fs=100):
    """Generates a high-fidelity synthetic 12-lead ECG signal."""
    t = np.linspace(0, length / fs, length, endpoint=False)
    signal = np.zeros((length, num_leads))
    
    # Generate a periodic cardiac cycle (QRS + P + T waves)
    # Heart rate is in beats per minute
    bps = heart_rate / 60.0
    period = 1.0 / bps
    
    # Time-based gating for heartbeat peaks
    phase = (t % period) / period
    
    for lead in range(num_leads):
        # Base rhythm: P wave
        p_wave = 0.05 * np.exp(-((phase - 0.2) / 0.04) ** 2)
        # QRS complex (main heartbeat spike)
        qrs = 0.8 * np.exp(-((phase - 0.4) / 0.015) ** 2)
        # T wave
        t_wave = 0.2 * np.exp(-((phase - 0.65) / 0.06) ** 2)
        
        # Add lead-specific scaling & variations
        lead_scale = 1.0 - 0.1 * lead if lead < 6 else 0.5 + 0.1 * (lead - 6)
        signal[:, lead] = (p_wave + qrs + t_wave) * lead_scale
        
        # Add slow baseline drift
        drift = 0.15 * np.sin(2 * np.pi * 0.15 * t + lead)
        signal[:, lead] += drift
        
        # Add high-frequency noise
        noise = 0.02 * np.random.randn(length)
        signal[:, lead] += noise
        
    return signal

def create_synthetic_dataset(output_dir, num_samples=250):
    """Creates a complete synthetic PTB-XL dataset with WFDB records."""
    print(f"Generating high-fidelity synthetic PTB-XL dataset in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create scp_statements.csv
    scp_data = {
        "description": [
            "Normal ECG", "Anterior Myocardial Infarction", "Inferior Myocardial Infarction",
            "Left Bundle Branch Block", "Right Bundle Branch Block", "ST/T Change",
            "Left Ventricular Hypertrophy"
        ],
        "diagnostic": [1, 1, 1, 1, 1, 1, 1],
        "rhythm": [0, 0, 0, 0, 0, 0, 0],
        "diagnostic_class": ["NORM", "MI", "MI", "CD", "CD", "STTC", "HYP"],
        "diagnostic_subclass": ["NORM", "AMI", "IMI", "LBBB", "RBBB", "STTC", "LVH"]
    }
    scp_df = pd.DataFrame(scp_data, index=["NORM", "AMI", "IMI", "LBBB", "RBBB", "STTC", "LVH"])
    scp_df.index.name = "scp_code"
    scp_df.to_csv(os.path.join(output_dir, "scp_statements.csv"))
    
    # 2. Generate records and build ptbxl_database.csv
    database_rows = []
    
    leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # Potential codes for synthetic generation
    possible_codes = [
        {"NORM": 100.0},
        {"AMI": 100.0, "STTC": 50.0},
        {"IMI": 100.0},
        {"LBBB": 100.0, "STTC": 30.0},
        {"RBBB": 100.0},
        {"STTC": 100.0},
        {"LVH": 100.0, "HYP": 50.0}
    ]
    
    for idx in range(1, num_samples + 1):
        ecg_id = idx
        patient_id = 10000 + (idx % 40) # 40 patients
        age = float(np.random.randint(18, 90))
        sex = np.random.choice([0, 1])
        height = float(np.random.randint(150, 195)) if sex == 1 else float(np.random.randint(140, 180))
        weight = float(np.random.randint(50, 110))
        
        # Pick a random set of scp codes
        scp_dict = possible_codes[idx % len(possible_codes)]
        strat_fold = ((idx - 1) % 10) + 1 # Folds 1-10 distributed evenly
        
        # Define relative file paths (PTB-XL style)
        folder_num = (idx // 1000) * 1000
        folder_num_str = f"{folder_num:05d}"
        file_name = f"{idx:05d}_lr"
        
        rel_dir = os.path.join("records100", folder_num_str)
        abs_record_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(abs_record_dir, exist_ok=True)
        
        # Generate the raw signal
        # Map pathology to heart rate variations
        hr = 75
        if "AMI" in scp_dict or "IMI" in scp_dict:
            hr = 90  # Tachycardia in acute MI
        elif "LBBB" in scp_dict or "RBBB" in scp_dict:
            hr = 60  # Conduction delays
            
        signal_100hz = generate_synthetic_ecg(length=1000, num_leads=12, heart_rate=hr, fs=100)
        
        # Write WFDB records (.dat and .hea)
        wfdb.wrsamp(
            record_name=file_name,
            fs=100,
            units=["mV"] * 12,
            sig_name=leads,
            p_signal=signal_100hz,
            fmt=["16"] * 12,
            write_dir=abs_record_dir
        )
        
        # Store database rows
        database_rows.append({
            "ecg_id": ecg_id,
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "height": height,
            "weight": weight,
            "scp_codes": str(scp_dict),
            "strat_fold": strat_fold,
            "filename_lr": os.path.join(rel_dir, file_name),
            "filename_hr": os.path.join(rel_dir.replace("records100", "records500"), file_name.replace("_lr", "_hr"))
        })
        
    db_df = pd.DataFrame(database_rows)
    db_df.set_index("ecg_id", inplace=True)
    db_df.to_csv(os.path.join(output_dir, "ptbxl_database.csv"))
    print(f"Synthetic dataset generation complete! Generated {num_samples} records.")

def download_real_ptbxl(output_dir):
    """Downloads and extracts the real PTB-XL dataset from PhysioNet."""
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "ptb-xl.zip")
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    
    print(f"Downloading PTB-XL from PhysioNet...")
    print(f"URL: {url}")
    print(f"Saving to: {zip_path}")
    
    # Progress callback helper
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}% completed", end="")
        
    urllib.request.urlretrieve(url, zip_path, progress_hook)
    print("\nDownload complete! Extracting files...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
    print("Extraction complete! Organizing directories...")
    # The zip contains a folder named "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    extracted_folder_name = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    extracted_path = os.path.join(output_dir, extracted_folder_name)
    
    if os.path.exists(extracted_path):
        # Move all contents from extracted_path to output_dir
        for item in os.listdir(extracted_path):
            os.rename(os.path.join(extracted_path, item), os.path.join(output_dir, item))
        os.rmdir(extracted_path)
        
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print("Real PTB-XL dataset is ready to use!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTB-XL Dataset Downloader & Synthetic Generator")
    parser.add_argument("--dir", type=str, default="data/ptbxl", help="Output directory path")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic demo data instead of downloading")
    parser.add_argument("--samples", type=int, default=250, help="Number of synthetic samples to generate")
    args = parser.parse_args()
    
    # Construct absolute path for safety
    abs_dir = os.path.abspath(args.dir)
    
    if args.synthetic:
        create_synthetic_dataset(abs_dir, args.samples)
    else:
        try:
            download_real_ptbxl(abs_dir)
        except Exception as e:
            print(f"\nError downloading real dataset: {e}")
            print("Falling back to high-fidelity synthetic data generation...")
            create_synthetic_dataset(abs_dir, args.samples)
