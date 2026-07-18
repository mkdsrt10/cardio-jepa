# 🎯 Project Objective: CardioRep

**CardioRep** is a self-supervised representation learning and clinical decision support platform for 12-lead electrocardiogram (ECG) waveforms. Based on the Joint Embedding Predictive Architecture (JEPA), CardioRep pretrains directly on raw waveforms to learn robust, semantic latent representations that empower downstream screening and case-retrieval clinical tools.

CardioRep is designed strictly as clinical decision support for automated, robust triage and similar-case retrieval rather than fully autonomous diagnosis.

---

## 🏥 Clinical Screening Goals

1. **Abnormality Screening & Triage**:
   - Classify 12-lead ECGs as "Normal" or "Requires Review" with high sensitivity.
   - Serve as an initial safety filter in high-volume environments (such as emergency departments or ambulatory clinics) to surface critical abnormalities to cardiologists.

2. **Multilabel Diagnostic Group Screening**:
   - Screen for five key diagnostic superclasses defined by the PTB-XL benchmark:
     - **NORM**: Normal ECG
     - **MI**: Myocardial Infarction
     - **STTC**: ST/T Change
     - **CD**: Conduction Disturbance
     - **HYP**: Hypertrophy
   - Support multilabel annotations to match co-occurring clinical pathologies.

3. **Interpretability & Clinician-in-the-Loop Decision Support**:
   - **Signal Quality Check**: Assess input waveform quality and flag artifacts, noise, or lead detachment.
   - **Similar-Case Retrieval**: Query a database of historical ECGs to retrieve cases with highly similar latent representations and clinical outcomes.
   - **Lead/Time Attribution**: Highlight the specific leads and temporal segments (e.g., QRS complexes, ST segments) that heavily influenced the risk score.

---

## 🔬 Representation Health & Technical Targets

1. **Self-Supervised Pretraining via ECG-JEPA**:
   - **Context-Target Masking**: Train an encoder to predict the representation of masked temporal intervals or masked leads (e.g., predicting Lead V1-V6 using Lead I, II, III, aVR, aVL, aVF) within a joint embedding space.
   - **Noise & Drift Invariance**: Force representations to be invariant to physical clinical artifacts (baseline wander, high-frequency sensor noise) through temporal slicing and synthetic signal distortions.

2. **Holy Trinity of Representation Diagnostics**:
   To guarantee a healthy, high-dimensional latent space free of collapse, CardioRep integrates three diagnostic metrics:
   - **Effective Rank**: Detects *dimensional collapse* (determines whether representation information is spread across orthogonal dimensions).
   - **Feature Standard Deviation (`feature_std`)**: Detects *variance collapse* (determines whether feature dimensions maintain sufficient variance).
   - **Pairwise Cosine Similarity (`pairwise_cosine_similarity`)**: Detects *embedding crowding* / point collapse (ensures different patients are mapped to distinct angles rather than crowding together).
