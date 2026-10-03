# VerifyVoice

### AI-Powered Deepfake Audio Detection & Explainability Platform

VerifyVoice is an end-to-end audio verification system that analyzes speech recordings and predicts whether a voice is **REAL** (organic human voice) or **FAKE** (synthetic/AI-generated voice) using a trained 4-layer Convolutional Neural Network (CNN) operating on normalized Mel-spectrogram time-frequency features. It incorporates **Grad-CAM** (Gradient-weighted Class Activation Mapping) on the final 512-channel convolutional layer to provide visual, interpretable spectrogram attribution for every prediction.

---

## Architecture

The system is built on a decoupled, production-ready full-stack architecture:

```text
┌────────────────────────────────────────────────────────┐
│                   Next.js Frontend                     │
│    React 19 • Tailwind CSS • Framer Motion • Web Audio │
└───────────────────────────┬────────────────────────────┘
                            │ Multipart/form-data
                            │ POST /api/predict
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│        Uvicorn • Librosa • Soundfile • NumPy           │
└───────────────────────────┬────────────────────────────┘
                            │ Standardized Tensor
                            │ [1, 1, 128, 63]
                            ▼
┌────────────────────────────────────────────────────────┐
│              4-Layer Deep Convolutional Network        │
│          Conv2D (64 → 128 → 256 → 512) • PyTorch       │
└───────────────────────────┬────────────────────────────┘
                            │ Logits + Layer 15 Attributions
                            ▼
┌────────────────────────────────────────────────────────┐
│         Grad-CAM Heatmap & Aggregate Prediction        │
│      Dark-Themed Spectrogram Overlay & Verdict Card    │
└────────────────────────────────────────────────────────┘
```

---

## Features

- **Accurate Audio Deepfake Detection:** Evaluates acoustic micro-timing, phase alignment, and spectral vocoder artifacts.
- **Grad-CAM Spectrogram Explainability:** Visualizes the exact temporal frames and Mel-frequency bins (0–8000 Hz) that drove the model's prediction.
- **Robust Chunk-Based Inference:** Slices audio of any length into 2-second windows with 50% overlap, aggregating confidence scores across all segments.
- **Universal Audio Format Support:** Decodes and verifies WAV, MP3, FLAC, OGG, and M4A audio files up to 50MB.
- **Client-Side History & Preview:** Built-in IndexedDB audio history and verification playback with animated waveforms.
- **Cross-Platform & Cloud-Ready:** Hardened CORS, decoupled environment configurations, and resilient dual-engine execution.

---

## Preprocessing Pipeline

The model expects acoustic inputs preprocessed according to strict specifications:

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Target Sample Rate** | `16,000 Hz` | Audio is resampled and converted to single-channel mono |
| **Window Duration** | `2.0 seconds` | Exactly `32,000` samples per window |
| **Overlap** | `50%` | Step stride of `16,000` samples between adjacent chunks |
| **Mel Bins** | `128` | Computed via `librosa.feature.melspectrogram` (`n_fft=2048`, `hop_length=512`) |
| **Amplitude Scaling** | `dB` | `mel_db = librosa.power_to_db(mel, ref=np.max)` |
| **Standardization** | Per-chunk | `mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)` |
| **Expected Tensor Shape** | `(1, 1, 128, 63)` | `(Batch, Channel, Mel_Bins, Frames)` |

---

## Model Architecture

The production classifier uses a specialized 4-stage convolutional acoustic feature extractor:

```text
Input: [1, 1, 128, 63]
  │
  ├── Block 1: Conv2D(1 → 64, k=3, p=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2) → Dropout2d(0.20)
  ├── Block 2: Conv2D(64 → 128, k=3, p=1) → BatchNorm2d(128) → ReLU → MaxPool2d(2) → Dropout2d(0.20)
  ├── Block 3: Conv2D(128 → 256, k=3, p=1) → BatchNorm2d(256) → ReLU → MaxPool2d(2) → Dropout2d(0.20)
  ├── Block 4: Conv2D(256 → 512, k=3, p=1) → BatchNorm2d(512) → ReLU → MaxPool2d(2) → Dropout2d(0.20)
  │            *(Layer 15 Conv2D is the target for Grad-CAM attribution)*
  │
  ├── AdaptiveAvgPool2d((1, 1))
  ├── Flatten
  ├── Linear(512 → 128) → ReLU → Dropout(0.30)
  └── Linear(128 → 1) → Scalar Logit
```

- **Inference Mode:** `model.eval()` with `torch.no_grad()`.
- **Classification Threshold:** Sigmoid probability $\ge 0.5 \implies$ **`FAKE`**, $< 0.5 \implies$ **`REAL`**.

---

## Project Structure

```text
├── backend/
│   ├── main.py             # FastAPI REST service & lifecycle management
│   ├── model.py            # CNN architecture definition & checkpoint loader
│   ├── pipeline.py         # 16 kHz audio chunking, Mel-spectrogram & inference
│   ├── gradcam.py          # Layer 15 Grad-CAM attribution heatmap generator
│   └── temp_gradcam/       # Ephemeral directory for serving Grad-CAM plots
├── Model/
│   └── CNN_4Layer_64_128_256_512_BN_Dropout_best.pth  # Trained weights
├── app/
│   ├── globals.css         # Design system & dark-glass aesthetics
│   ├── layout.tsx          # Root layout
│   └── page.tsx            # Main application UI, upload, analysis & errors
├── components/
│   ├── AudioPlayer.tsx     # Custom audio playback & waveforms
│   ├── HistoryView.tsx     # IndexedDB analysis history view
│   ├── PredictionExplanation.tsx # Grad-CAM explanation renderer
│   └── ResultCard.tsx      # Metrics grid (confidence, probabilities, verdict)
├── lib/
│   ├── analyzeAudio.ts     # Client inference API caller & normalizer
│   └── historyDb.ts        # IndexedDB storage provider
├── samples/                # Sample audio files for testing and evaluation
├── next.config.mjs         # Next.js production & rewrite configuration
├── package.json            # Frontend dependencies & scripts
├── requirements.txt        # Python backend dependencies
├── .env.example            # Environment variable template
└── Dockerfile.backend      # Container configuration for backend deployment
```

---

## Python Version & Requirements

- **Recommended Python Version:** `Python 3.11` (specifically `3.11.x` provides rock-solid PyTorch binary support and avoids Windows 11 Smart App Control execution blocks).
- **Core Dependencies:**
  - `torch`
  - `librosa`
  - `soundfile`
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `fastapi`
  - `uvicorn`
  - `python-multipart`

---

## Environment Variables

Create `.env.local` for local development by copying `.env.example`:

```bash
cp .env.example .env.local
```

### Configuration Options

| Variable | Default (Local) | Purpose |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Target URL for the Python ML inference API |
| `HOST` | `0.0.0.0` | Host address for FastAPI service binding |
| `PORT` | `8000` | Port for FastAPI service binding |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed CORS origins for browser security |
| `BACKEND_INTERNAL_URL` | `http://127.0.0.1:8000` | Internal URL for Next.js server-side rewrites |

---

## Local Development Setup

### 1. Setup Python Backend

```bash
# Create virtual environment with Python 3.11
uv venv .venv --python 3.11
# Or: python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health at: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

### 2. Setup Next.js Frontend

```bash
# Install frontend dependencies
npm install

# Start Next.js development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Production Deployment

### Backend Deployment (Container or PaaS)

Using Docker:
```bash
docker build -f Dockerfile.backend -t verifyvoice-backend .
docker run -p 8000:8000 -e PORT=8000 -e ALLOWED_ORIGINS="https://yourfrontend.com" verifyvoice-backend
```

Using Uvicorn directly (PaaS / VM):
```bash
uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
```

### Frontend Deployment (Vercel, Node, or Cloudflare)

1. Set `NEXT_PUBLIC_API_URL` to your production backend URL (e.g. `https://api.yourdomain.com`).
2. Build and start:
```bash
npm run build
npm run start
```

---

## API Reference

### 1. Health Check
- **Endpoint:** `GET /health` or `GET /api/health`
- **Response:**
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "device": "cpu",
    "model_path": "Model/CNN_4Layer_64_128_256_512_BN_Dropout_best.pth",
    "service": "VerifyVoice Audio Deepfake Detection"
  }
  ```

### 2. Audio Deepfake Prediction
- **Endpoint:** `POST /api/predict`
- **Content-Type:** `multipart/form-data`
- **Fields:** `audio` or `file` (binary audio file: WAV, MP3, FLAC, OGG, M4A)
- **Response:**
  ```json
  {
    "success": true,
    "prediction": "FAKE",
    "probability": 0.8819,
    "confidence": 88.19,
    "fake_probability": 88.19,
    "real_probability": 11.81,
    "chunks_analyzed": 3,
    "gradcam_url": "/api/gradcam/gradcam_13ab7085eae7.png"
  }
  ```

### 3. Grad-CAM Visualization Image
- **Endpoint:** `GET /api/gradcam/{filename}`
- **Response:** `image/png` containing the acoustic Mel-spectrogram with overlaid activation heatmap.

---

## Known Limitations

- **Classifier Scope:** This system is trained on specific deepfake voice datasets (acoustic synthesis, voice cloning, vocoder generation). While it exhibits strong discrimination against common text-to-speech (TTS) and voice conversion models, it is not guaranteed to detect novel, unseen zero-shot architectures.
- **Audio Quality:** Heavy background noise, aggressive lossy MP3 compression (< 64 kbps), or heavy reverberation can impact acoustic feature extraction and lower classification confidence.
- **Decision Support:** Outputs represent statistical classification probabilities based on time-frequency speech analysis and should be utilized as decision support, not absolute cryptographic proof.
