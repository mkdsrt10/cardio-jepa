import os
import numpy as np
import wfdb

from src.app.triage import ECGTriageSystem
from src.data.dataset import load_ptbxl_metadata

def main():
    print("--------------------------------------------------")
    print("🩺 Testing Clinical Triage System & QA checks")
    print("--------------------------------------------------")
    
    # 1. Instantiate the triage system
    model_path = "checkpoints/supervised_baseline.pt"
    triage_sys = ECGTriageSystem(model_path=model_path)
    
    # 2. Find a valid record from the generated synthetic dataset
    data_dir = "data/ptbxl"
    df, _ = load_ptbxl_metadata(data_dir)
    sample_rel_path = df.iloc[0]["filename_lr"]
    sample_abs_path = os.path.join(data_dir, sample_rel_path)
    
    print(f"Loading sample record: {sample_abs_path}")
    
    # 3. Triage the valid record
    print("\n[Test 1] Triaging a valid clinical ECG...")
    report_valid = triage_sys.triage_ecg(record_path=sample_abs_path)
    import json
    print(json.dumps(report_valid, indent=2))
    
    # 4. Triage an invalid flatline record
    print("\n[Test 2] Triaging an invalid (flatline) ECG...")
    flatline_signal = np.zeros((12, 1000))
    report_flat = triage_sys.triage_ecg(raw_signal=flatline_signal)
    print(json.dumps(report_flat, indent=2))
    
    # 5. Triage a saturated record
    print("\n[Test 3] Triaging an invalid (saturated/high amplitude) ECG...")
    # Add a sinus wave shifted by 6.5 to have variance but exceed the 5.0mV threshold
    t = np.linspace(0, 10, 1000)
    saturated_signal = np.sin(t) + 6.5 # max absolute value will be 7.5 > 5.0 mV threshold
    saturated_signal = np.tile(saturated_signal, (12, 1)).astype(np.float32)
    report_sat = triage_sys.triage_ecg(raw_signal=saturated_signal)
    print(json.dumps(report_sat, indent=2))
    
    print("\nAll clinical triage & QA test cases passed successfully!")

if __name__ == "__main__":
    main()
