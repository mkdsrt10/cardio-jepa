import os
import torch
import numpy as np
import wfdb

from src.models.encoder import ECGEncoder1D, ECGClassifier1D
from src.evaluation.interpret import compute_integrated_gradients

class ECGTriageSystem:
    """Production-grade Clinical Triage & Quality Assurance system for 12-lead ECGs."""
    
    def __init__(self, model_path=None, jepa_path=None, data_dir=None, device=None, latent_dim=256):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.latent_dim = latent_dim
        self.superclasses = ["NORM", "MI", "STTC", "CD", "HYP"]
        
        # 1. Build and load Supervised Classifier model
        self.encoder = ECGEncoder1D(in_channels=12, latent_dim=latent_dim)
        self.model = ECGClassifier1D(encoder=self.encoder, num_classes=len(self.superclasses))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.model_loaded = False
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
            
        # 2. Build and load JEPA Feature Extractor for Similarity Engine
        self.jepa_encoder = ECGEncoder1D(in_channels=12, latent_dim=latent_dim)
        self.jepa_encoder = self.jepa_encoder.to(self.device)
        self.jepa_encoder.eval()
        
        self.jepa_loaded = False
        if jepa_path and os.path.exists(jepa_path):
            self.load_jepa_weights(jepa_path)
            
        # 3. Setup retrieval indexing if training data is available
        self.retrieval_enabled = False
        if self.jepa_loaded and data_dir:
            self.setup_retrieval(data_dir)
            
    def load_weights(self, model_path):
        """Loads trained supervised weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.superclasses = checkpoint.get("superclasses", self.superclasses)
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model_loaded = True
        print(f"ECG Triage System weights successfully loaded from '{model_path}'")

    def load_jepa_weights(self, jepa_path):
        """Loads pre-trained JEPA weights from checkpoint."""
        checkpoint = torch.load(jepa_path, map_location=self.device, weights_only=False)
        if "encoder_state_dict" in checkpoint:
            self.jepa_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        else:
            self.jepa_encoder.load_state_dict(checkpoint)
        self.jepa_encoder.eval()
        self.jepa_loaded = True
        print(f"JEPA representation encoder successfully loaded from '{jepa_path}'")

    def setup_retrieval(self, data_dir):
        """Loads training set metadata and caches JEPA representation vectors for similar-case lookup."""
        from src.data.dataset import load_ptbxl_metadata, PTBXLDataset
        from torch.utils.data import DataLoader
        
        try:
            self.df, _ = load_ptbxl_metadata(data_dir)
            # Filter folds 1-8 (Train split)
            self.train_df = self.df[self.df["strat_fold"].isin(list(range(1, 9)))].copy()
            
            self.train_records_metadata = []
            for idx, row in self.train_df.iterrows():
                self.train_records_metadata.append({
                    "ecg_id": int(idx),
                    "patient_id": int(row["patient_id"]),
                    "age": float(row["age"]) if not np.isnan(row["age"]) else None,
                    "sex": int(row["sex"]) if not np.isnan(row["sex"]) else None,
                    "labels": [self.superclasses[i] for i in range(len(self.superclasses)) if row["labels"][i] == 1.0]
                })
                
            # Load or pre-compute training set embeddings
            cache_embed_path = "checkpoints/train_jepa_embeddings.npy"
            if os.path.exists(cache_embed_path):
                self.train_embeddings = np.load(cache_embed_path)
            else:
                print("Pre-computing JEPA training embeddings for similarity indexing...")
                train_dataset = PTBXLDataset(data_dir, self.df, self.superclasses, fold_list=list(range(1, 9)), augment=False)
                loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
                embeddings = []
                with torch.no_grad():
                    for x, _ in loader:
                        x = x.to(self.device)
                        z = self.jepa_encoder(x)
                        embeddings.append(z.cpu().numpy())
                self.train_embeddings = np.concatenate(embeddings, axis=0)
                os.makedirs(os.path.dirname(cache_embed_path), exist_ok=True)
                np.save(cache_embed_path, self.train_embeddings)
                print(f"Similarity index built with {len(self.train_embeddings)} vectors.")
                
            # Pre-normalize for fast Cosine similarity
            norms = np.linalg.norm(self.train_embeddings, axis=1, keepdims=True) + 1e-8
            self.norm_train_embeddings = self.train_embeddings / norms
            self.retrieval_enabled = True
        except Exception as e:
            print(f"Retrieval Engine setup failed: {e}")

    def run_signal_quality_check(self, x: np.ndarray):
        """Performs physical checks on raw ECG waveforms.
        
        Input x shape: [12, Length]
        Returns: (is_valid, reason)
        """
        variances = np.var(x, axis=1)
        flat_leads = np.where(variances < 1e-5)[0]
        if len(flat_leads) > 0:
            return False, f"Flatline detected: Lead(s) {flat_leads.tolist()} are inactive or detached."
            
        max_vals = np.abs(x)
        saturated_leads = np.where(np.any(max_vals > 5.0, axis=1))[0]
        if len(saturated_leads) > 0:
            return False, f"Saturation alert: Lead(s) {saturated_leads.tolist()} exceed clinical range (>5.0mV)."
            
        successive_diff_var = np.var(np.diff(x, axis=1), axis=1)
        noisy_leads = np.where(successive_diff_var > 0.8)[0]
        if len(noisy_leads) > 0:
            return False, f"Excessive noise check failed: High-frequency noise on Lead(s) {noisy_leads.tolist()}."
            
        return True, "Passed signal quality assurance."

    def triage_ecg(self, record_path=None, raw_signal=None):
        """Triages a 12-lead ECG waveform.
        
        Outputs multi-label predictions, confidence levels, Integrated Gradients attribution,
        similar historical cases, and compiles a structured decision report.
        """
        if raw_signal is None and record_path is not None:
            signal, _ = wfdb.rdsamp(record_path)
            raw_signal = signal.astype(np.float32).T
            
        if raw_signal is None:
            raise ValueError("Must provide either a file path or raw ECG signal.")
            
        # 1. Physical Signal Quality QA check
        is_valid, quality_reason = self.run_signal_quality_check(raw_signal)
        
        # 2. Pad / Truncate signal to 1000 samples for model inference
        length = raw_signal.shape[1]
        target_len = 1000
        if length < target_len:
            processed_signal = np.pad(raw_signal, ((0, 0), (0, target_len - length)), "constant")
        else:
            processed_signal = raw_signal[:, :target_len]
            
        # 3. Model Inference, Attribution and Case Retrieval
        probabilities = {}
        triage_status = "PENDING_REVIEW"
        abnormality_score = 0.0
        confidence = 0.0
        lead_influence = {}
        temporal_influence = []
        similar_cases = []
        
        if is_valid:
            x_tensor = torch.from_numpy(processed_signal).float().unsqueeze(0).to(self.device) # [1, 12, 1000]
            
            # A. Probability Classification
            if self.model_loaded:
                with torch.no_grad():
                    logits = self.model(x_tensor)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                probabilities = {self.superclasses[i]: float(probs[i]) for i in range(len(self.superclasses))}
                
                # Determine abnormality score and triage decision
                abnormal_probs = [probabilities[c] for c in self.superclasses if c != "NORM"]
                abnormality_score = float(np.max(abnormal_probs)) if len(abnormal_probs) > 0 else 0.0
                norm_prob = probabilities.get("NORM", 0.0)
                
                if norm_prob >= 0.50 and abnormality_score < 0.40:
                    triage_status = "NORMAL (Low Priority)"
                    confidence = float(norm_prob)
                else:
                    triage_status = "REQUIRES REVIEW (High Priority)"
                    confidence = float(np.max([abnormality_score, 1.0 - norm_prob]))
                    
                # B. Integrated Gradients Saliency/Attribution
                try:
                    # Target the class that represents the main abnormality or NORM
                    target_class_idx = np.argmax(probs)
                    saliency = compute_integrated_gradients(self.model, torch.from_numpy(processed_signal).float(), target_class_idx, steps=10, device=self.device)
                    # Absolute magnitude of attribution
                    abs_saliency = np.abs(saliency)
                    
                    # Compute lead-wise attribution rankings
                    lead_attribution = np.sum(abs_saliency, axis=1) # [12]
                    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                    lead_att_scaled = lead_attribution / (np.sum(lead_attribution) + 1e-8)
                    lead_influence = {lead_names[i]: float(lead_att_scaled[i]) for i in range(12)}
                    # Sort lead influence descending
                    lead_influence = dict(sorted(lead_influence.items(), key=lambda item: item[1], reverse=True))
                    
                    # Compute temporal attribution pooling (summarize into 10 intervals of length 100)
                    temp_attribution = np.sum(abs_saliency, axis=0) # [1000]
                    for seg in range(10):
                        interval_sum = np.sum(temp_attribution[seg*100 : (seg+1)*100])
                        temporal_influence.append({
                            "seconds_start": float(seg),
                            "seconds_end": float(seg + 1),
                            "saliency_score": float(interval_sum)
                        })
                    # Scale temporal influence scores to sum to 1
                    total_temp_sum = sum(item["saliency_score"] for item in temporal_influence) + 1e-8
                    for item in temporal_influence:
                        item["saliency_score"] /= total_temp_sum
                        item["saliency_score"] = round(item["saliency_score"], 4)
                except Exception as e:
                    print(f"Attribution mapping failed: {e}")
            else:
                quality_reason += " (Supervised weights not loaded; triage classification and attribution skipped)"
                
            # C. Similarity-based Case Retrieval
            if self.jepa_loaded and self.retrieval_enabled:
                try:
                    with torch.no_grad():
                        z_query = self.jepa_encoder(x_tensor).squeeze(0).cpu().numpy()
                        
                    # L2 normalize query
                    z_query_norm = z_query / (np.linalg.norm(z_query) + 1e-8)
                    
                    # Compute cosine similarity
                    similarities = self.norm_train_embeddings @ z_query_norm
                    
                    # Get top 3 nearest historical records
                    top_k = 3
                    top_indices = np.argsort(similarities)[::-1][:top_k]
                    
                    for idx in top_indices:
                        record_info = self.train_records_metadata[idx]
                        similar_cases.append({
                            "ecg_id": record_info["ecg_id"],
                            "patient_id": record_info["patient_id"],
                            "age": record_info["age"],
                            "sex": "Male" if record_info["sex"] == 1 else "Female" if record_info["sex"] == 0 else None,
                            "clinical_findings": record_info["labels"],
                            "similarity_score": float(similarities[idx])
                        })
                except Exception as e:
                    print(f"Similar cases retrieval failed: {e}")
                    
        # 4. Compile Structured Machine-Readable Clinical Decision Report
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
            "explainability_insights": {
                "top_influencing_leads": list(lead_influence.keys())[:3] if is_valid and lead_influence else [],
                "lead_saliency_distribution": {k: round(v, 4) for k, v in lead_influence.items()} if is_valid and lead_influence else {},
                "temporal_saliency_intervals": temporal_influence if is_valid and temporal_influence else []
            },
            "retrieved_historical_cases": similar_cases if is_valid and similar_cases else [],
            "waveform_metadata": {
                "leads": raw_signal.shape[0],
                "samples": length,
            }
        }
        
        return report
