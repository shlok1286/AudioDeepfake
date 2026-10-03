import io
import math
import os
import sys
import types
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure numba doesn't fail on Windows Smart App Control if unsigned _devicearray is blocked
if 'numba' not in sys.modules:
    try:
        import numba
    except (ImportError, OSError):
        numba_mock = types.ModuleType('numba')
        numba_mock.jit = lambda *args, **kwargs: (lambda f: f)
        numba_mock.njit = lambda *args, **kwargs: (lambda f: f)
        numba_mock.vectorize = lambda *args, **kwargs: (lambda f: f)
        numba_mock.guvectorize = lambda *args, **kwargs: (lambda f: f)
        numba_mock.stencil = lambda *args, **kwargs: (lambda f: f)
        numba_mock.cfunc = lambda *args, **kwargs: (lambda f: f)
        numba_mock.prange = range
        sys.modules['numba'] = numba_mock

import librosa
import numpy as np
import soundfile as sf

from .model import get_model, no_grad, _TORCH_AVAILABLE
from .gradcam import generate_gradcam_overlay

# Target audio specifications expected by the trained CNN
TARGET_SR = 16000
CHUNK_DURATION = 2.0  # seconds
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION)  # 32,000 samples
CHUNK_STEP = int(CHUNK_SAMPLES * 0.5)  # 50% overlap = 16,000 samples
N_MELS = 128
TARGET_FRAMES = 63
HOP_LENGTH = 512
N_FFT = 2048

SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}


def validate_audio_file(filename: str, file_size: int, max_size_mb: int = 50):
    """
    Validates file extension and size.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if file_size > max_size_mb * 1024 * 1024:
        raise ValueError(f"File exceeds maximum allowed size of {max_size_mb}MB.")


def load_audio_from_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Decodes audio bytes into a 16 kHz mono floating-point numpy array.
    Uses soundfile or librosa fallback with safe temp file handling.
    """
    try:
        buffer = io.BytesIO(file_bytes)
        audio_data, sr = sf.read(buffer, always_2d=False, dtype='float32')

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sr != TARGET_SR:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)

        if len(audio_data) > 0 and not np.isnan(audio_data).any():
            return audio_data
    except Exception:
        pass

    safe_ext = Path(filename).suffix.lower() if Path(filename).suffix else ".wav"
    if safe_ext not in SUPPORTED_EXTENSIONS:
        safe_ext = ".wav"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=safe_ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        audio_data, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
        if len(audio_data) == 0 or np.isnan(audio_data).any():
            raise ValueError("Decoded audio is empty or corrupted.")
        return audio_data
    except Exception:
        raise ValueError("Unable to process this audio file. Please try another recording.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def slice_into_chunks(audio_data: np.ndarray) -> List[np.ndarray]:
    """
    Splits audio into 2-second chunks (32,000 samples) with 50% overlap.
    If the audio is shorter than 2 seconds, pads with zeros to 2 seconds.
    """
    total_samples = len(audio_data)

    if total_samples <= CHUNK_SAMPLES:
        padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        padded[:total_samples] = audio_data
        return [padded]

    chunks = []
    start = 0
    while start + CHUNK_SAMPLES <= total_samples:
        chunk = audio_data[start : start + CHUNK_SAMPLES]
        chunks.append(chunk)
        start += CHUNK_STEP

    if start < total_samples and (total_samples - start) > (TARGET_SR * 0.4):
        tail = audio_data[start:]
        tail_padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        tail_padded[:len(tail)] = tail
        chunks.append(tail_padded)

    return chunks


def compute_mel_spectrogram(chunk: np.ndarray) -> Tuple[Any, np.ndarray]:
    """
    Converts a 2-second 16 kHz chunk into a normalized Mel spectrogram of shape [1, 1, 128, 63].
    Uses librosa.feature.melspectrogram, power_to_db, and (mel_db - mean) / (std + 1e-8) standardization.
    """
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=TARGET_SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    current_frames = mel_db.shape[1]
    if current_frames < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - current_frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    elif current_frames > TARGET_FRAMES:
        mel_db = mel_db[:, :TARGET_FRAMES]

    raw_mel_db = mel_db.copy()

    mean = mel_db.mean()
    std = mel_db.std()
    mel_norm = (mel_db - mean) / (std + 1e-8)

    # Return torch tensor if PyTorch is active, else numpy array [1, 1, 128, 63]
    if _TORCH_AVAILABLE:
        import torch
        tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        tensor = mel_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)

    return tensor, raw_mel_db


def run_inference_pipeline(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Complete inference pipeline:
    1. Validate uploaded audio
    2. Load and resample to 16 kHz mono
    3. Split into 2-sec chunks with 50% overlap
    4. Compute Mel spectrograms [1, 1, 128, 63]
    5. Run PyTorch / ONNX CNN inference on all chunks
    6. Aggregate predictions using average fake probability
    7. Generate Grad-CAM on final 512-ch Conv2D layer
    8. Return JSON payload matching specifications
    """
    validate_audio_file(filename, len(file_bytes))
    audio = load_audio_from_bytes(file_bytes, filename)

    if len(audio) == 0:
        raise ValueError("Audio recording is empty or corrupt. Please try another file.")

    chunks = slice_into_chunks(audio)
    model, device, _ = get_model()

    chunk_probs = []
    chunk_tensors = []
    raw_specs = []

    for chunk in chunks:
        tensor, raw_mel = compute_mel_spectrogram(chunk)
        chunk_tensors.append(tensor)
        raw_specs.append(raw_mel)

        with no_grad():
            if hasattr(tensor, 'to'):
                input_data = tensor.to(device)
            else:
                input_data = tensor

            logit = model(input_data)
            if hasattr(logit, 'item'):
                logit_val = logit.item()
            elif isinstance(logit, np.ndarray):
                logit_val = float(logit[0, 0])
            else:
                logit_val = float(logit)

            fake_prob = float(1.0 / (1.0 + math.exp(-logit_val)))
            chunk_probs.append(fake_prob)

    avg_fake_prob = float(np.mean(chunk_probs))
    avg_real_prob = float(1.0 - avg_fake_prob)

    # 0.5 Threshold classification
    prediction = "FAKE" if avg_fake_prob >= 0.5 else "REAL"

    confidence_val = (avg_fake_prob if prediction == "FAKE" else avg_real_prob) * 100.0
    confidence_val = round(confidence_val, 2)
    fake_prob_percent = round(avg_fake_prob * 100.0, 2)
    real_prob_percent = round(avg_real_prob * 100.0, 2)

    # Select representative chunk for Grad-CAM
    if prediction == "FAKE":
        best_chunk_idx = int(np.argmax(chunk_probs))
    else:
        best_chunk_idx = int(np.argmin(chunk_probs))

    best_tensor = chunk_tensors[best_chunk_idx]
    best_raw_mel = raw_specs[best_chunk_idx]

    gradcam_filename = generate_gradcam_overlay(
        model=model,
        mel_tensor=best_tensor,
        raw_mel_db=best_raw_mel,
        device=device,
        target_layer_idx=15
    )

    return {
        "success": True,
        "prediction": prediction,
        "probability": round(avg_fake_prob, 4),
        "confidence": confidence_val,
        "fake_probability": fake_prob_percent,
        "real_probability": real_prob_percent,
        "chunks_analyzed": len(chunks),
        "gradcam_url": f"/api/gradcam/{gradcam_filename}",
    }
