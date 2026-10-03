# VerifyVoice

### AI-Powered Deepfake Audio Detection

VerifyVoice is an AI-powered audio verification system that analyzes speech recordings and predicts whether the voice is **REAL** or **FAKE** using a deep convolutional neural network trained on Mel Spectrogram representations.

The system combines a **React/Next.js frontend**, **FastAPI ML inference backend**, **PyTorch CNN model**, and **Grad-CAM explainability** to provide both a prediction and visual insight into the model's decision.

---

## Overview

AI-generated voices are becoming increasingly realistic, making it difficult to determine whether an audio recording is authentic.

VerifyVoice addresses this problem by converting speech into a time-frequency representation and using a CNN to detect patterns associated with synthetic audio.

### Core Pipeline

```text
Audio Upload
     │
     ▼
Audio Decoding
     │
     ▼
16 kHz Mono Conversion
     │
     ▼
2-Second Audio Chunks
     │
     ▼
50% Overlap
     │
     ▼
Mel Spectrogram
     │
     ▼
CNN Classification
     │
     ├───────────────┐
     ▼               ▼
 REAL              FAKE
     │               │
     └───────┬───────┘
             ▼
     Audio-Level Probability
             │
             ▼
        Grad-CAM
             │
             ▼
       Visual Result
