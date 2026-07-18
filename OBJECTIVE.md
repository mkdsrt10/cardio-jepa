# 🎯 Project Objective: ECG-JEPA Representation & Triage System

The primary objective of this project is to build a self-supervised representation learning system for 12-lead electrocardiogram (ECG) waveforms based on the Joint Embedding Predictive Architecture (JEPA). This system is designed as clinical decision support for automated, robust triage and similar-case retrieval rather than fully autonomous diagnosis.

---

## 🏥 Clinical Screening Goals

1. **Abnormality Screening & Triage**:
   - Classify 12-lead ECGs as "Normal" or "Requires Review" with high sensitivity.
   - Serve as an initial safety filter in high-volume settings (such as emergency departments or remote cardiac monitoring platforms) to surface critical abnormalities to cardiologists.

2. **Multilabel Diagnostic Group Screening**:
   - Screen for five diagnostic superclasses defined by the PTB-XL benchmark:
     - **NORM**: Normal ECG
     - **MI**: Myocardial Infarction
     - **STTC**: ST/T Change
     - **CD**: Conduction Disturbance
     - **HYP**: Hypertrophy
   - Support multilabel annotation as individual clinical cases often present with co-occurring pathologies.

3. **Interpretability & Clinician-in-the-Loop Decision Support**:
   - **Signal Quality Check**: Assess input waveform quality and flag artifacts, noise, or lead detachment.
   - **Similar-Case Retrieval**: Query a database of historical ECGs to retrieve cases with highly similar latent representations and clinical outcomes.
   - **Lead/Time Attribution**: Highlight the specific leads and temporal segments (e.g., QRS complexes, ST segments) that heavily influenced the system's risk score.
   - **Machine-Readable Report**: Generate structured JSON/HL7-compliant triage reports with abnormality scores, confidence intervals, and reference historical matches.

---

## 🔬 Technical Targets

1. **Self-Supervised Pretraining via ECG-JEPA**:
   - **Context-Target Masking**: Train an encoder to predict the representation of masked temporal intervals or masked leads (e.g., predicting Lead V1-V6 using Lead I, II, III, aVR, aVL, aVF) within a joint embedding space.
   - **Noise & Drift Invariance**: Force representations to be invariant to typical clinical artifacts (such as baseline wander, electromyographical high-frequency noise, and scaling artifacts) through temporal slicing and synthetic signal distortions.
   - **Prevention of Dimensional Collapse**: Use signature regularization (SIGReg with $\lambda = 20.0$) and coordinate-wise variance constraints to maintain high effective rank in the embedding space ($\text{Effective Rank} \ge 60$).

2. **Data Pipeline Optimization**:
   - Efficiently load, process, and split the **PTB-XL dataset** containing 21,800 clinical 10-second ECG recordings at 100Hz/500Hz.
   - Keep dataset assets in highly serialized, memory-mapped formats (such as Arrow or Parquet) to eliminate CPU-to-GPU data transfer bottlenecks.
   - Maximize GPU throughput with custom 1D data loading pipelines using multi-threaded PyTorch DataLoader settings (`num_workers`, `persistent_workers`, and `pin_memory`).

3. **Downstream Tasks & Probing**:
   - **Linear Probing**: Freeze the JEPA encoder weights and train a lightweight linear classifier to measure semantic separability of representations.
   - **Retrieval Engine**: Build an efficient vector database lookup using FAISS or PyTorch cosine-similarity matching for similar-case retrieval.
   - **Attribution & Visualization**: Implement gradient-based attribution (such as Integrated Gradients or Grad-CAM adapted for 1D signals) to identify localized abnormality regions.
