# 🫀 Self-Supervised ECG Representation & Triage System

An advanced clinical decision support system that leverages Joint Embedding Predictive Architecture (JEPA) style self-supervised learning to learn highly robust, reusable representations of 12-lead Electrocardiograms (ECGs). 

Instead of treating ECGs as generic images or building brittle classifiers, this system models temporal and lead-based relationships from raw waveforms to provide robust abnormality screening, similar-case retrieval, and interpretable clinical decision support.

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
  [ ECG-JEPA 1D Encoder ] ───────► Patient ECG Embedding (Latent Space)
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
In real-world settings, ECG waveforms are mostly unlabelled. The self-supervised objective teaches the model physical cardiac properties by predicting:
- **Masked Time Intervals**: Predicting the representation of a masked 2-second rhythm segment using the surrounding context.
- **Masked Leads**: Predicting representations of missing leads (such as chest leads $V_1$-$V_6$) from the active limb leads ($I, II, III$).
- **Noise & Drift Invariance**: Forcing representation alignment between clean and synthetically distorted views (e.g., adding low-frequency breathing baseline wander or high-frequency muscle artifacts).

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

---

## ⚡ Key Engineering Playbook Alignment
To maximize GPU training speeds and ensure stable representation convergence, this repository is built in strict adherence to the project's **High-Performance SSL Playbook**:
1. **Memory-Mapped Data Pipelines**: Keeps waveforms in Arrow/Parquet format to bypass slow CPU-bound serialization locks.
2. **Optimal Dataloader Threading**: Employs `num_workers=4`, `pin_memory=True`, and `persistent_workers=True` to fully saturate the GPU.
3. **Preventing Representation Collapse**: Uses signature regularization (`SIGReg` loss weight $\lambda = 20.0$) to guarantee high representation dimensions ($\text{Effective Rank} \ge 60$) and avoid point/dimensional collapse.
4. **Stable FP32 Diagnostics**: Performs singular value decomposition (SVD) and Effective Rank computations exclusively in `FP32` on CPU subsets of size $\le 1024$ to preserve numerical stability and eliminate calculation stalls.
