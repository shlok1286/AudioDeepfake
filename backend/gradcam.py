import os
import time
import uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRADCAM_DIR = Path(__file__).resolve().parent / "temp_gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_gradcams(max_age_seconds: int = 900):
    """
    Deletes temporary Grad-CAM images older than max_age_seconds (default 15 minutes).
    """
    now = time.time()
    for file_path in GRADCAM_DIR.glob("*.png"):
        try:
            if now - file_path.stat().st_mtime > max_age_seconds:
                file_path.unlink()
        except Exception:
            pass


def generate_gradcam_overlay(
    model: nn.Module,
    mel_tensor: torch.Tensor,
    raw_mel_db: np.ndarray,
    device: torch.device,
    target_layer_idx: int = 15,
) -> str:
    """
    Generates a Grad-CAM heatmap specifically for the 512-channel Conv2D layer (features[15]).
    Overlays the activation heatmap onto the Mel spectrogram and saves as an image file.

    Returns the filename of the saved image (e.g., 'gradcam_abc123.png').
    """
    cleanup_old_gradcams()

    # Enable gradients for the input chunk and target layer
    input_tensor = mel_tensor.clone().detach().to(device)
    input_tensor.requires_grad = True

    # Target the 512-channel Conv2D layer (layer 15 in features)
    target_conv = model.features[target_layer_idx]
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd_handle = target_conv.register_forward_hook(fwd_hook)
    bwd_handle = target_conv.register_full_backward_hook(bwd_hook)

    try:
        # Forward pass
        logit = model(input_tensor)

        # Backpropagate logit to obtain gradients at features[15]
        model.zero_grad()
        logit.backward()

        act = activations[0].detach()  # [1, 512, H_feat, W_feat]
        grad = gradients[0].detach()   # [1, 512, H_feat, W_feat]

        # Global average pooling of gradients per channel (weights alpha_k)
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, 512, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * act, dim=1, keepdim=True)    # [1, 1, H_feat, W_feat]

        # Apply ReLU to keep only features that positively contributed
        cam = torch.relu(cam)

        # Interpolate to the original Mel spectrogram resolution (128, 87)
        h, w = mel_tensor.shape[2], mel_tensor.shape[3]
        cam = nn.functional.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        denom = (cam_np.max() - cam_np.min())
        if denom > 1e-8:
            cam_np = (cam_np - cam_np.min()) / denom
        else:
            cam_np = np.zeros_like(cam_np)

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    # Render Mel Spectrogram + Grad-CAM Heatmap overlay
    # Aesthetics: dark themed, crisp typography, clean borders
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')

    # Base spectrogram
    spec_min, spec_max = raw_mel_db.min(), raw_mel_db.max()
    norm_spec = (raw_mel_db - spec_min) / (spec_max - spec_min + 1e-8)
    ax.imshow(norm_spec, aspect='auto', origin='lower', cmap='magma', alpha=0.65)

    # Grad-CAM heatmap overlay (inferno / jet colormap with transparency)
    im = ax.imshow(cam_np, aspect='auto', origin='lower', cmap='jet', alpha=0.45)

    # Colorbar styling
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.tick_params(labelsize=8, colors='#94A3B8')
    cbar.outline.set_edgecolor('#1E293B')
    cbar.set_label('Feature Attribution Intensity', fontsize=8, color='#94A3B8', labelpad=6)

    ax.set_title('Acoustic Spectrogram Attribution (Grad-CAM Layer 15: Conv2D 512-ch)', fontsize=9.5, color='#F8FAFC', pad=8, fontweight='bold')
    ax.set_xlabel('Temporal Frames (2-Second Window)', fontsize=8.5, color='#94A3B8')
    ax.set_ylabel('Mel Frequency Bins (128)', fontsize=8.5, color='#94A3B8')
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#64748B')

    for spine in ax.spines.values():
        spine.set_color('#1E293B')

    plt.tight_layout()

    filename = f"gradcam_{uuid.uuid4().hex[:12]}.png"
    save_path = GRADCAM_DIR / filename
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close(fig)

    return filename
