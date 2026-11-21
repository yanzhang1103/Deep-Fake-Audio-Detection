# Deepfake Audio Detection with Signal Processing & Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yanzhang1103/Deep-Fake-Audio-Detection/blob/main/deepfake_audio_detection.ipynb)

**Davi Dai, Yan Zhang** · Course project, PSYCH 186B

Voice-cloning tools can now produce speech that is hard to tell from the real thing. That makes fraud, impersonation and misinformation easier.
This project tests whether **26 interpretable acoustic features** (MFCCs, spectral shape, energy, chroma) plus standard ML models can detect AI-generated speech. It then asks the harder question: **does the detector still work on audio from a different source?**

<p align="center"><img src="assets/poster.jpg" width="520" alt="Project poster"></p>

## Key findings

| | XGBoost | Random Forest | LSTM |
|---|---|---|---|
| Accuracy on DEEP-VOICE (held-out test) | **99.3%** | 98.9% | 98.5% |
| Training time | **0.9 s** | 3.3 s | 128.8 s |

1. **In-distribution, simple features are enough.** All three models reach about 99% accuracy. XGBoost is the most accurate *and* the fastest, about 140× faster to train than the LSTM.
2. **Tree models are robust to feature ablation.** Removing the most discriminative features (MFCC2, MFCC4, MFCC18, RMS) barely changes XGBoost's accuracy. The MFCC and spectral features are highly correlated, so the model shifts onto substitutes. High importance doesn't make a feature irreplaceable.
3. **Generalization is the real problem.** Applied without retraining to the external **Fake-or-Real** dataset, XGBoost falls to **about 62% accuracy**, and it labels most *real* speakers in **VCTK** as fake. The model learns the recording conditions and synthesis method of one dataset, not a general signature of synthetic speech.
4. **Data diversity, not the feature set, is the bottleneck.** Adding a slice of the outside dataset to training largely restores performance on it (Section 8 of the notebook).

> Numbers above are from our project run. The notebook re-derives all of them with fixed seeds, so exact values can shift slightly between runs and library versions.

## Approach

```
audio ──► 1-s segments ──► Librosa features (26) ──► XGBoost / Random Forest / LSTM ──► REAL vs FAKE
```

| Step | Details |
|---|---|
| **Data** | [DEEP-VOICE](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition): real speech and voice-converted deepfakes, as 1-second segments (about 11.8k rows, balanced) |
| **Features** | MFCC 1–20, chroma, RMS energy, spectral centroid, bandwidth and rolloff, zero-crossing rate |
| **Models** | XGBoost (early stopping on a validation set), Random Forest (100 trees), 2-layer LSTM over 17 overlapping windows of the feature vector |
| **Evaluation** | Stratified 60/20/20 train/val/test split; accuracy, precision, recall, F1, MCC, ROC-AUC (from probabilities); mean ± SD over 5 splits |
| **Robustness checks** | Block-split leakage check, label-shuffle sanity check (should give chance), feature ablation |
| **Interpretability** | Chi² ranking, XGBoost gain importance, SHAP, partial dependence |
| **Generalization** | Zero-shot evaluation on [Fake-or-Real](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) and [VCTK](https://www.kaggle.com/datasets/colestan/vctk-corpus), with features extracted the same way; distribution-shift heatmap; mixed-data retraining |

## Run it

The easiest way is the **Open in Colab** badge above. All three datasets download automatically through `kagglehub` (about 4 GB in total).

Locally:

```bash
pip install -r requirements.txt
jupyter notebook deepfake_audio_detection.ipynb
```

The full run takes about 25 to 40 minutes on CPU. Set `RUN_LSTM = False` at the top to skip the LSTM experiments.

## Repository

```
├── deepfake_audio_detection.ipynb   # full analysis
├── requirements.txt
└── assets/
    ├── poster.pdf                   # project poster
    └── poster.jpg
```

## Limitations & next steps

- DEEP-VOICE is small (8 speakers, one voice-conversion system), and its segments come from a few long recordings, so in-dataset scores are likely optimistic.
- Frame-averaged features discard temporal structure that modern detectors exploit.
- **Next:** train across multiple synthesis systems (ASVspoof plus FoR), use speaker-disjoint splits, and compare against learned representations (raw-waveform CNNs, wav2vec 2.0).

## References

- Bird, J. J., & Lotfi, A. (2023). *Real-time Detection of AI-Generated Speech for DeepFake Voice Conversion.* arXiv:2308.12734.
- Reimao, R., & Tzerpos, V. (2019). *FoR: A Dataset for Synthetic Speech Detection.* SpeD 2019.
- Yamagishi, J., Veaux, C., & MacDonald, K. (2019). *CSTR VCTK Corpus.* University of Edinburgh.
