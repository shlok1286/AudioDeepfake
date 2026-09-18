import os
import glob
from pathlib import Path
import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    """
    4-Layer CNN for Deepfake Audio Detection.
    Architecture:
      - 4 Conv Blocks: 64 -> 128 -> 256 -> 512
        Each block: Conv2D(3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) -> Dropout2d(0.3)
      - Global Average Pooling: AdaptiveAvgPool2d((1, 1))
      - Classifier:
        Flatten -> Linear(512 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 64
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 4: 256 -> 512 (Layer 15 is the target for Grad-CAM)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Adaptive pooling to ensure 512-dim output regardless of input duration
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits


def find_model_checkpoint(project_root: Path = None) -> Path:
    """
    Automatically inspects the model/ and Model/ directories to find the trained .pth file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    search_dirs = [
        project_root / "Model",
        project_root / "model",
        project_root,
    ]

    for s_dir in search_dirs:
        if s_dir.exists() and s_dir.is_dir():
            pth_files = list(s_dir.glob("*.pth")) + list(s_dir.glob("*.pt"))
            if pth_files:
                # Prefer best model if named so
                best = [p for p in pth_files if "best" in p.name.lower()]
                return best[0] if best else pth_files[0]

    raise FileNotFoundError(
        f"No .pth checkpoint found in project model directories: {[str(d) for d in search_dirs]}"
    )


# Global singleton instance loaded once
_MODEL_INSTANCE = None
_DEVICE = None
_MODEL_PATH = None


def get_model(device: torch.device = None):
    """
    Loads and returns the cached deepfake detection model in eval mode.
    Ensures model is only loaded into memory once.
    """
    global _MODEL_INSTANCE, _DEVICE, _MODEL_PATH

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _MODEL_INSTANCE is None:
        _MODEL_PATH = find_model_checkpoint()
        print(f"[Model Loader] Loading trained checkpoint from: {_MODEL_PATH}")
        print(f"[Model Loader] Using compute device: {device}")

        model = AudioCNN()
        state_dict = torch.load(_MODEL_PATH, map_location=device, weights_only=False)

        # In case the checkpoint was saved as a dict with 'state_dict' or 'model_state_dict'
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False

        _MODEL_INSTANCE = model
        _DEVICE = device
        print(f"[Model Loader] Model loaded successfully into eval mode.")

    return _MODEL_INSTANCE, _DEVICE, _MODEL_PATH
