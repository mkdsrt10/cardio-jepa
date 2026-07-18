import os
import torch
import numpy as np
import wfdb

from src.models.encoder import ECGEncoder1D, ECGClassifier1D

class ECGTriageSystem:
    """Production-grade Clinical Triage & Quality Assurance system for 12-lead ECGs."""
    
    def __init__(self, model_path=None, device=None, latent_dim=256):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.latent_dim = latent_dim
        self.superclasses = ["NORM", "MI", "STTC", "CD", "HYP"]
        
        # Build network architecture
        self.encoder = ECGEncoder1D(in_channels=12, latent_dim=latent_dim)
        self.model = ECGClassifier1D(encoder=self.encoder, num_classes=len(self.superclasses))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.model_loaded = False
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
            
    def load_weights(self, model_path):
        """Loads trained weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.superclasses = checkpoint.get("superclasses", self.superclasses)
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model_loaded = True
        print(f"ECG Triage System weights successfully loaded from '{model_path}'")

    def run_signal_quality_check(self, x: np.ndarray):
        """Performs robust physical checks on raw ECG waveforms.
        
        Input x shape: [12, Length]
        Returns: (is_valid, reason)
        """
        # 1. Flatline / Lead detachment check
        variances = np.var(x, axis=1)
        flat_leads = np.where(variances < 1e-5)[0]
        if len(flat_leads) > 0:
            return False, f"Flatline detected: Lead(s) {flat_leads.tolist()} are inactive or detached."
            
        # 2. Extreme saturation check (voltage values usually shouldn't exceed 5.0 mV in clinical recordings)
        max_vals = np.abs(x)
        saturated_leads = np.where(np.any(max_vals > 5.0, axis=1))[0]
        if len(saturated_leads) > 0:
            return False, f"Saturation alert: Lead(s) {saturated_leads.tolist()} exceed clinical range (>5.0mV)."
            
        # 3. High-frequency muscle artifact / noise check
        # We can calculate high-frequency noise by looking at differences between successive samples
        successive_diff_var = np.var(np.diff(x, axis=1), axis=1)
        noisy_leads = np.where(successive_diff_var > 0.8)[0]
        if len(noisy_leads) > 0:
            return False, f"Excessive noise check failed: High-frequency noise on Lead(s) {noisy_leads.tolist()}."
            
        return True, "Passed signal quality assurance."

    @torch.no_grad()
    def triage_ecg(self, record_path=None, raw_signal=None):
        """Triages a 12-lead ECG waveform.
        
        Outputs multi-label probability predictions, confidence metrics, and clinical decision support logs.
        """
        if raw_signal is None and record_path is not None:
            # Load from file using wfdb
            signal, _ = wfdb.rdsamp(record_path)
            # Transpose to [12, Length]
            raw_signal = signal.astype(np.float32).T
            
        if raw_signal is None:
            raise ValueError("Must provide either a file path or raw ECG signal.")
            
        # 1. Physical Signal Quality QA check
        is_valid, quality_reason = self.run_signal_quality_check(raw_signal)
        
        # 2. Pad / Truncate signal to 1000 samples for the model
        length = raw_signal.shape[1]
        target_len = 1000
        if length < target_len:
            processed_signal = np.pad(raw_signal, ((0, 0), (0, target_len - length)), "constant")
        else:
            processed_signal = raw_signal[:, :target_len]
            
        # 3. Model Inference (if weights are loaded)
        probabilities = {}
        triage_status = "PENDING_REVIEW"
        abnormality_score = 0.0
        confidence = 0.0
        
        if self.model_loaded and is_valid:
            x_tensor = torch.from_numpy(processed_signal).float().unsqueeze(0).to(self.device) # [1, 12, 1000]
            logits = self.model(x_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            # Map superclasses to their probability
            probabilities = {self.superclasses[i]: float(probs[i]) for i in range(len(self.superclasses))}
            
            # Abnormality score is the max of the abnormal probabilities
            abnormal_probs = [probabilities[c] for c in self.superclasses if c != "NORM"]
            abnormality_score = float(np.max(abnormal_probs))
            
            # Normal class probability
            norm_prob = probabilities.get("NORM", 0.0)
            
            # Triage decision rule:
            # Normal triage: NORM prob is >= 0.50, and all other abnormal classes are < 0.40
            if norm_prob >= 0.50 and abnormality_score < 0.40:
                triage_status = "NORMAL (Low Priority)"
                confidence = float(norm_prob)
            else:
                triage_status = "REQUIRES REVIEW (High Priority)"
                confidence = float(np.max([abnormality_score, 1.0 - norm_prob]))
        elif not self.model_loaded and is_valid:
            quality_reason += " (Model weights not loaded; triage screening skipped)"
            
        # 4. Compile Structured Machine-Readable Decision Report
        report = {
            "status": "SUCCESS" if is_valid else "FAILED_QA",
            "quality_assurance": {
                "passed": is_valid,
                "detail": quality_reason,
            },
            "triage_results": {
                "decision": triage_status if is_valid else "REJECTED_BY_QA",
                "abnormality_score": round(abnormality_score, 4) if is_valid else 0.0,
                "confidence_score": round(confidence, 4) if is_valid else 0.0,
                "findings": {k: round(v, 4) for k, v in probabilities.items()} if is_valid else {}
            },
            "waveform_metadata": {
                "leads": raw_signal.shape[0],
                "samples": length,
            }
        }
        
        return report
