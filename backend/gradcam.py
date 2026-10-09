import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False

from scipy.ndimage import zoom

GRADCAM_DIR = Path(__file__).resolve().parent / "temp_gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

def _render_with_pil(cam_np: np.ndarray, raw_mel_db: np.ndarray, save_path: Path):
    try:
        from PIL import Image
        spec_min, spec_max = raw_mel_db.min(), raw_mel_db.max()
        norm_spec = (raw_mel_db - spec_min) / (spec_max - spec_min + 1e-8)
        norm_spec = np.clip(norm_spec * 255, 0, 255).astype(np.uint8)

        cam_u8 = np.clip(cam_np * 255, 0, 255).astype(np.uint8)
        r = np.clip(norm_spec * 0.9 + cam_u8 * 1.2, 0, 255).astype(np.uint8)
        g = np.clip(norm_spec * 0.3 + cam_u8 * 0.5, 0, 255).astype(np.uint8)
        b = np.clip(norm_spec * 0.4 + (255 - cam_u8) * 0.3, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.flipud(rgb)
        img = Image.fromarray(rgb)
        img = img.resize((800, 350), Image.Resampling.BILINEAR)
        img.save(save_path, format="PNG")
    except Exception as e:
        # Fallback to 1x1 black image if PIL also has issue
        from PIL import Image
        img = Image.new('RGB', (800, 350), color=(11, 15, 25))
        img.save(save_path, format="PNG")


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
    model: Any,
    mel_tensor: Any,
    raw_mel_db: np.ndarray,
    device: Any,
    target_layer_idx: int = 15,
) -> str:
    """
    Generates a Grad-CAM heatmap specifically for the 512-channel Conv2D layer (features[15]).
    Overlays the activation heatmap onto the Mel spectrogram and saves as an image file.

    Returns the filename of the saved image (e.g., 'gradcam_abc123.png').
    """
    cleanup_old_gradcams()

    cam_np = None

    # Check if model has PyTorch hook capability
    if hasattr(model, 'features') and hasattr(model, 'zero_grad'):
        try:
            import torch
            import torch.nn as nn
            input_tensor = mel_tensor.clone().detach().to(device)
            input_tensor.requires_grad = True

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
                logit = model(input_tensor)
                model.zero_grad()
                logit.backward()

                act = activations[0].detach()
                grad = gradients[0].detach()
                weights = torch.mean(grad, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * act, dim=1, keepdim=True)
                cam = torch.relu(cam)

                h, w = raw_mel_db.shape[0], raw_mel_db.shape[1]
                cam = nn.functional.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
                cam_np = cam.squeeze().cpu().numpy()

                denom = (cam_np.max() - cam_np.min())
                if denom > 1e-8:
                    cam_np = (cam_np - cam_np.min()) / denom
                else:
                    cam_np = np.zeros_like(cam_np)
            finally:
                fwd_handle.remove()
                bwd_handle.remove()
        except Exception as e:
            cam_np = None

    # Analytical Grad-CAM for ONNX / NumPy engine
    if cam_np is None:
        try:
            if hasattr(model, 'forward_with_features'):
                logits, l15 = model.forward_with_features(mel_tensor)
            else:
                l15 = None

            if l15 is not None and hasattr(model, 'state_dict'):
                state_dict = model.state_dict
                w1 = state_dict['classifier.1.weight']
                b1 = state_dict['classifier.1.bias']
                w2 = state_dict['classifier.4.weight']
                b2 = state_dict['classifier.4.bias']
                bn_gamma = state_dict['features.16.weight']
                bn_var = state_dict['features.16.running_var']
                bn_mean = state_dict['features.16.running_mean']
                bn_beta = state_dict['features.16.bias']

                # Forward through BN16 -> ReLU -> MaxPool -> GAP -> Classifier
                y = (l15[0] - bn_mean[:, None, None]) / np.sqrt(bn_var[:, None, None] + 1e-5) * bn_gamma[:, None, None] + bn_beta[:, None, None]
                r = np.maximum(0, y)

                h_pool, w_pool = r.shape[1] // 2, r.shape[2] // 2
                mp = np.zeros((512, h_pool, w_pool), dtype=np.float32)
                for i in range(h_pool):
                    for j in range(w_pool):
                        mp[:, i, j] = np.max(r[:, 2*i:2*i+2, 2*j:2*j+2], axis=(1, 2))
                p = np.mean(mp, axis=(1, 2))
                z1_pre = np.dot(w1, p) + b1
                dz1 = w2[0] * (z1_pre > 0)
                dp = np.dot(dz1, w1)
                alpha = dp * (bn_gamma / np.sqrt(bn_var + 1e-5))

                cam = np.zeros((l15.shape[2], l15.shape[3]), dtype=np.float32)
                for c in range(512):
                    cam += alpha[c] * l15[0, c]
                cam = np.maximum(0, cam)

                h_target, w_target = raw_mel_db.shape[0], raw_mel_db.shape[1]
                cam_resized = zoom(cam, (h_target / cam.shape[0], w_target / cam.shape[1]), order=1)
                if cam_resized.shape != (h_target, w_target):
                    cam_resized = cam_resized[:h_target, :w_target]

                denom = cam_resized.max() - cam_resized.min()
                if denom > 1e-8:
                    cam_np = (cam_resized - cam_resized.min()) / denom
                else:
                    cam_np = np.zeros_like(cam_resized)
        except Exception as e:
            cam_np = None

    if cam_np is None:
        cam_np = np.zeros_like(raw_mel_db, dtype=np.float32)

    filename = f"gradcam_{uuid.uuid4().hex[:12]}.png"
    save_path = GRADCAM_DIR / filename

    saved = False
    if _MATPLOTLIB_AVAILABLE:
        try:
            # Render Mel Spectrogram + Grad-CAM Heatmap overlay
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
            fig.patch.set_facecolor('#0B0F19')
            ax.set_facecolor('#0B0F19')

            # Base spectrogram
            spec_min, spec_max = raw_mel_db.min(), raw_mel_db.max()
            norm_spec = (raw_mel_db - spec_min) / (spec_max - spec_min + 1e-8)
            ax.imshow(norm_spec, aspect='auto', origin='lower', cmap='magma', alpha=0.65)

            # Grad-CAM heatmap overlay
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
            plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
            plt.close(fig)
            saved = True
        except Exception:
            saved = False

    if not saved:
        _render_with_pil(cam_np, raw_mel_db, save_path)

    return filename
