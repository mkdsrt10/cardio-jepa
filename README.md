# 🫀 CardioRep: A Self-Supervised ECG Representation Learning and Clinical Decision Support Platform

An advanced clinical decision support platform that leverages Joint Embedding Predictive Architecture (JEPA) style self-supervised learning to extract highly robust, reusable representations of 12-lead Electrocardiograms (ECGs).

Instead of treating ECGs as generic images or building brittle classifiers, CardioRep models temporal and lead-based relationships from raw waveforms to provide robust abnormality screening, similar-case retrieval, and interpretable clinical decision support.

---

## 🏗️ System Architecture

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                            INFERENCE PIPELINE                           │
 └─────────────────────────────────────────────────────────────────────────┘
  [ 12-lead ECG Upload ]
           │
           ▼
  [ Signal Quality Check ] ──────► High noise / Lead detachment alert
           │
           ▼
  [ CardioRep 1D Encoder ] ──────► Patient ECG Embedding (Latent Space)
           │
           ├───► [ Abnormality Triage Classifier ] ────► Normal vs. Abnormal + Superclasses
           │
           ├───► [ Vector DB Similarity Search ] ─────► Similar Historical Cases Retrieved
           │
           └───► [ 1D Attribution Engine ] ──────────► Lead & Time Segment Highlights
           │
           ▼
  [ Clinical Decision Report ]
```

### 🧠 The JEPA-style Self-Supervised Objective
In real-world settings, ECG waveforms are mostly unlabelled. CardioRep teaches the model physical cardiac properties by predicting:
- **Masked Time Intervals**: Predicting the representation of a masked 2-second rhythm segment using the surrounding context.
- **Masked Leads**: Predicting representations of missing chest leads ($V_1$-$V_6$) from active limb leads ($I, II, III$).
- **Noise & Drift Invariance**: Forcing representation alignment between clean and synthetically distorted views.

---

## 🔬 The Holy Trinity of Representation Diagnostics

During pretraining, CardioRep monitors three distinct mathematical metrics to gauge embedding space health and prevent all modes of collapse:

| Metric | Diagnostics Target | Failure Mode Addressed | Actionable Threshold |
| :--- | :--- | :--- | :---: |
| **Effective Rank** | Dimensional Collapse | Representation content collapsing into a low-dimensional subspace. | $\ge 60.0$ |
| **Feature Std (`feature_std`)** | Variance Collapse | Individual feature dimensions collapsing to zero variance across the batch. | $\ge 0.5$ |
| **Pairwise Cosine (`pairwise_cosine`)** | Embedding Crowding | All samples clumping into a single dense direction (point collapse). | $\le 0.1$ |

---

## 📁 Repository Structure

```text
ecg-jepa/
├── config/                  # Configuration files for training & inference
├── data/                    # Dataset cache (e.g., PTB-XL download and cache files)
├── src/                     # Source code directory
│   ├── __init__.py
│   ├── data/                # Data pipelines
│   │   ├── __init__.py
│   │   ├── dataset.py       # PTB-XL custom PyTorch Dataset with 1D augmentations
│   │   └── pipeline.py      # High-performance dataloader, Arrow/Parquet conversions
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   ├── encoder.py       # ECGEncoder1D (1D ResNet with large receptive field)
│   │   ├── jepa.py          # JEPA Pretraining loss, Context-Target masking
│   │   └── classifier.py    # Multi-label diagnostic probing and classification heads
│   ├── evaluation/          # Evaluation and attribution
│   │   ├── __init__.py
│   │   ├── metrics.py       # Effective Rank, Linear Probing, Retrieval Precision
│   │   └── interpret.py     # 1D Attribution & Lead/Time Saliency Visualizations
│   └── app/                 # Triage product application
│       ├── __init__.py
│       ├── quality.py       # Signal quality assurance and artifact detection
│       ├── database.py      # Indexing and retrieving similar historical ECG vectors
│       └── report.py        # Automated PDF/JSON clinical report builder
├── tests/                   # Unit and integration tests
├── OBJECTIVE.md             # Clinical screening goals and technical targets
├── VALUE_PROPOSITION.md     # Real-world clinical benefit and self-supervised efficiency
└── README.md                # This file
```
