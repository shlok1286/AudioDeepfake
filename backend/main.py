import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("verifyvoice-backend")

# Ensure the backend directory is in Python module search path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir.parent) not in sys.path:
    sys.path.insert(0, str(backend_dir.parent))

from backend.model import get_model
from backend.pipeline import run_inference_pipeline, validate_audio_file
from backend.gradcam import GRADCAM_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-loads model checkpoint on backend startup.
    Ensures model is only initialized once and ready for inference.
    """
    try:
        model, device, path = get_model()
        host = os.environ.get("HOST", "0.0.0.0")
        port = os.environ.get("PORT", "8000")
        logger.info("=" * 65)
        logger.info(" VerifyVoice ML Inference Backend Running")
        logger.info(f"[*] Binding:        http://{host}:{port}")
        logger.info(f"[*] Loaded Model:   {path}")
        logger.info(f"[*] Compute Device: {device}")
        logger.info(f"[*] Health Check:   /health")
        logger.info(f"[*] Predict Route:  POST /api/predict")
        logger.info("=" * 65)
    except Exception as e:
        logger.error(f"[!] Critical error initializing model on startup: {e}", exc_info=True)
    yield


app = FastAPI(
    title="VerifyVoice ML Backend",
    description="Production Deepfake Audio Detection API with 4-Layer CNN & Grad-CAM",
    version="1.0.0",
    lifespan=lifespan
)

# Parse allowed CORS origins from environment
def get_cors_origins() -> List[str]:
    raw_origins = os.environ.get("ALLOWED_ORIGINS") or os.environ.get("FRONTEND_URL")
    if raw_origins:
        return [o.strip() for o in raw_origins.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

origins = get_cors_origins()
logger.info(f"Configuring CORS for origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True if "*" not in origins else False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


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
        # Sanitized user-facing input validation error
        logger.warning(f"Validation error for '{filename}': {ve}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(ve),
                "detail": str(ve)
            }
        )
    except Exception as e:
        # Log complete internal detail on server, return safe user error
        logger.error(f"Inference pipeline failure on '{filename}': {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Audio analysis could not be completed. Please ensure the recording is valid speech audio.",
                "detail": "Inference server encountered an internal processing error."
            }
        )


@app.get("/api/gradcam/{filename}")
async def get_gradcam_image(filename: str):
    """
    Serves generated Grad-CAM spectrogram heatmap visualization images.
    Secured against directory traversal.
    """
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
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
