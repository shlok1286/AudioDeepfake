import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Ensure the backend directory is in Python module search path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir.parent) not in sys.path:
    sys.path.insert(0, str(backend_dir.parent))

from backend.model import get_model
from backend.pipeline import run_inference_pipeline, validate_audio_file
from backend.gradcam import GRADCAM_DIR

app = FastAPI(
    title="VerifyVoice ML Backend",
    description="Deepfake Audio Detection API with 4-Layer CNN & Grad-CAM",
    version="1.0.0"
)

# Enable CORS for local Next.js frontend dev server and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import traceback


@app.on_event("startup")
async def startup_event():
    """
    Pre-loads model checkpoint on backend startup.
    Ensures model is only initialized once and ready for inference.
    """
    try:
        model, device, path = get_model()
        print("=" * 65)
        print(" VerifyVoice ML Inference Backend Running")
        print(f"[*] Inference URL:  http://127.0.0.1:8000")
        print(f"[*] Loaded Model:   {path}")
        print(f"[*] Compute Device: {device}")
        print(f"[*] Health Check:   http://127.0.0.1:8000/health")
        print(f"[*] Predict Route:  POST http://127.0.0.1:8000/api/predict")
        print("=" * 65)
    except Exception as e:
        traceback.print_exc()
        print(f"[!] Error loading model during startup: {e}", file=sys.stderr)


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend and model status."""
    try:
        model, device, path = get_model()
        model_loaded = model is not None
        device_str = str(device)
        model_path_str = str(path)
    except Exception as e:
        model_loaded = False
        device_str = "error"
        model_path_str = str(e)

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device_str,
        "model_path": model_path_str,
        "service": "VerifyVoice Audio Deepfake Detection"
    }


@app.post("/predict")
@app.post("/api/predict")
async def predict_audio(
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Receives uploaded audio file as multipart/form-data.
    Returns deepfake prediction, confidence %, chunk count, and Grad-CAM explanation URL.
    """
    upload = audio or file
    if not upload:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "No audio file provided. Please upload an audio recording in 'audio' field.",
                "detail": "No audio file provided. Please upload an audio recording in 'audio' field."
            }
        )

    filename = upload.filename or "recording.wav"

    try:
        file_bytes = await upload.read()
        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        # Execute full preprocessing, inference, and Grad-CAM pipeline
        result = run_inference_pipeline(file_bytes=file_bytes, filename=filename)
        return JSONResponse(status_code=200, content=result)

    except ValueError as ve:
        # Sanitized 400 user-facing error message
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(ve),
                "detail": str(ve)
            }
        )
    except Exception as e:
        # Log complete internal detail and traceback for debugging
        print(f"[Inference Error]: {e}", file=sys.stderr)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Inference failed: {str(e)}",
                "detail": f"Inference failed: {str(e)}"
            }
        )


@app.get("/api/gradcam/{filename}")
async def get_gradcam_image(filename: str):
    """
    Serves generated Grad-CAM spectrogram heatmap visualization images.
    Secured against directory traversal.
    """
    # Sanitize filename
    safe_name = Path(filename).name
    target_file = GRADCAM_DIR / safe_name

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="Grad-CAM visualization not found or expired.")

    return FileResponse(
        path=str(target_file),
        media_type="image/png",
        filename=safe_name
    )


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
