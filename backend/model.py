import os
import io
import sys
import glob
import zipfile
import pickle
from pathlib import Path
from typing import Tuple, Any, Optional

import numpy as np

# Try importing torch if allowed by Windows Application Control
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    class AudioCNN(nn.Module):
        """
        4-Layer CNN for Deepfake Audio Detection.
        Architecture:
          - 4 Conv Blocks: 64 -> 128 -> 256 -> 512
            Each block: Conv2D(3x3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) -> Dropout2d(0.2)
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
                best = [p for p in pth_files if "best" in p.name.lower()]
                return best[0] if best else pth_files[0]

    raise FileNotFoundError(
        f"No .pth checkpoint found in project model directories: {[str(d) for d in search_dirs]}"
    )


def extract_state_dict_from_pth(pth_path: Path):
    """
    Extracts numpy state dict directly from .pth zip archive without importing torch._C.
    """
    z = zipfile.ZipFile(str(pth_path))
    data_pkl_matches = [n for n in z.namelist() if n.endswith('data.pkl')]
    if not data_pkl_matches:
        raise ValueError(f"Invalid PyTorch checkpoint format in {pth_path}")

    prefix = data_pkl_matches[0].rsplit('data.pkl', 1)[0]
    data_pkl = z.read(prefix + 'data.pkl')
    storages = {}

    class DummyTorchClass:
        def __init__(self, name):
            self.__name__ = name

    class TorchUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                def rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                    dtype, key, device, numel = storage
                    raw_bytes = storages[key]
                    itemsize = 8 if dtype == 'LongStorage' else 4
                    np_dtype = np.float32 if dtype == 'FloatStorage' else (np.int64 if dtype == 'LongStorage' else np.float32)
                    arr = np.frombuffer(raw_bytes, dtype=np_dtype, count=numel, offset=storage_offset * itemsize)
                    arr = np.lib.stride_tricks.as_strided(arr, shape=size, strides=[s * arr.itemsize for s in stride])
                    return arr.copy()
                return rebuild
            if module.startswith('torch'):
                return DummyTorchClass(name)
            return super().find_class(module, name)

        def persistent_load(self, pid):
            tag, storage_type, key, device, numel = pid
            storage_name = getattr(storage_type, '__name__', str(storage_type))
            if key not in storages:
                storages[key] = z.read(f'{prefix}data/{key}')
            return (storage_name, key, device, numel)

    unpickler = TorchUnpickler(io.BytesIO(data_pkl))
    state_dict = unpickler.load()
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    return state_dict


def ensure_onnx_model(pth_path: Path) -> Path:
    """
    Compiles an ONNX model from the .pth weights with exact 4-layer CNN architecture.
    """
    onnx_path = pth_path.parent / (pth_path.stem + ".onnx")
    if onnx_path.exists() and onnx_path.stat().st_size > 1000000:
        return onnx_path

    import onnx
    from onnx import helper, TensorProto

    sd = extract_state_dict_from_pth(pth_path)

    inits = []
    for k, v in sd.items():
        if v.ndim == 0:
            continue
        safe_name = k.replace('.', '_')
        inits.append(helper.make_tensor(safe_name, TensorProto.FLOAT, list(v.shape), v.astype(np.float32).flatten().tolist()))

    nodes = [
        # Block 1
        helper.make_node('Conv', ['input', 'features_0_weight', 'features_0_bias'], ['b1_conv'], pads=[1, 1, 1, 1]),
        helper.make_node('BatchNormalization', ['b1_conv', 'features_1_weight', 'features_1_bias', 'features_1_running_mean', 'features_1_running_var'], ['b1_bn'], epsilon=1e-5),
        helper.make_node('Relu', ['b1_bn'], ['b1_relu']),
        helper.make_node('MaxPool', ['b1_relu'], ['b1_pool'], kernel_shape=[2, 2], strides=[2, 2]),

        # Block 2
        helper.make_node('Conv', ['b1_pool', 'features_5_weight', 'features_5_bias'], ['b2_conv'], pads=[1, 1, 1, 1]),
        helper.make_node('BatchNormalization', ['b2_conv', 'features_6_weight', 'features_6_bias', 'features_6_running_mean', 'features_6_running_var'], ['b2_bn'], epsilon=1e-5),
        helper.make_node('Relu', ['b2_bn'], ['b2_relu']),
        helper.make_node('MaxPool', ['b2_relu'], ['b2_pool'], kernel_shape=[2, 2], strides=[2, 2]),

        # Block 3
        helper.make_node('Conv', ['b2_pool', 'features_10_weight', 'features_10_bias'], ['b3_conv'], pads=[1, 1, 1, 1]),
        helper.make_node('BatchNormalization', ['b3_conv', 'features_11_weight', 'features_11_bias', 'features_11_running_mean', 'features_11_running_var'], ['b3_bn'], epsilon=1e-5),
        helper.make_node('Relu', ['b3_bn'], ['b3_relu']),
        helper.make_node('MaxPool', ['b3_relu'], ['b3_pool'], kernel_shape=[2, 2], strides=[2, 2]),

        # Block 4 (Layer 15 Conv is target for Grad-CAM)
        helper.make_node('Conv', ['b3_pool', 'features_15_weight', 'features_15_bias'], ['layer15_act'], pads=[1, 1, 1, 1]),
        helper.make_node('BatchNormalization', ['layer15_act', 'features_16_weight', 'features_16_bias', 'features_16_running_mean', 'features_16_running_var'], ['b4_bn'], epsilon=1e-5),
        helper.make_node('Relu', ['b4_bn'], ['b4_relu']),
        helper.make_node('MaxPool', ['b4_relu'], ['b4_pool'], kernel_shape=[2, 2], strides=[2, 2]),

        # Global Average Pooling & Flatten
        helper.make_node('GlobalAveragePool', ['b4_pool'], ['gap']),
        helper.make_node('Flatten', ['gap'], ['flat'], axis=1),

        # Classifier
        helper.make_node('Gemm', ['flat', 'classifier_1_weight', 'classifier_1_bias'], ['fc1'], transB=1, alpha=1.0, beta=1.0),
        helper.make_node('Relu', ['fc1'], ['fc1_relu']),
        helper.make_node('Gemm', ['fc1_relu', 'classifier_4_weight', 'classifier_4_bias'], ['logits'], transB=1, alpha=1.0, beta=1.0),
    ]

    inputs = [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 128, 63])]
    outputs = [
        helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info('layer15_act', TensorProto.FLOAT, [1, 512, 16, 7]),
    ]

    graph = helper.make_graph(nodes, 'AudioCNN', inputs, outputs, inits)
    model_def = helper.make_model(graph, producer_name='VerifyVoice')
    model_def.ir_version = 10
    del model_def.opset_import[:]
    model_def.opset_import.append(helper.make_opsetid('', 20))
    onnx.checker.check_model(model_def)
    onnx.save(model_def, str(onnx_path))
    return onnx_path


class AudioCNNORT:
    """
    ONNX Runtime production model wrapper that mirrors the PyTorch AudioCNN interface.
    """
    def __init__(self, onnx_path: Path, pth_path: Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        self.onnx_path = onnx_path
        self.pth_path = pth_path
        self.state_dict = extract_state_dict_from_pth(pth_path)

    def eval(self):
        return self

    def __call__(self, x):
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        logits, _ = self.session.run(None, {'input': x})
        return logits

    def forward_with_features(self, x):
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        logits, l15 = self.session.run(None, {'input': x})
        return logits, l15


class NoGradContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def no_grad():
    if _TORCH_AVAILABLE:
        return torch.no_grad()
    return NoGradContext()


_MODEL_INSTANCE = None
_DEVICE = None
_MODEL_PATH = None


def get_model(device=None):
    """
    Loads and returns the cached deepfake detection model in eval mode.
    Ensures model is only loaded into memory once.
    """
    global _MODEL_INSTANCE, _DEVICE, _MODEL_PATH

    if _MODEL_INSTANCE is None:
        _MODEL_PATH = find_model_checkpoint()
        print(f"[Model Loader] Loading trained checkpoint from: {_MODEL_PATH}")

        if _TORCH_AVAILABLE:
            try:
                if device is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"[Model Loader] Using PyTorch compute device: {device}")
                model = AudioCNN()
                state_dict = torch.load(_MODEL_PATH, map_location=device, weights_only=False)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict, strict=True)
                model.to(device)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                _MODEL_INSTANCE = model
                _DEVICE = device
                print(f"[Model Loader] PyTorch AudioCNN loaded successfully into eval mode.")
                return _MODEL_INSTANCE, _DEVICE, _MODEL_PATH
            except Exception as e:
                print(f"[Model Loader] PyTorch execution unavailable: {e}. Falling back to ONNX Runtime engine.")

        # Robust ONNX Runtime fallback
        onnx_path = ensure_onnx_model(_MODEL_PATH)
        _MODEL_INSTANCE = AudioCNNORT(onnx_path, _MODEL_PATH)
        _DEVICE = "cpu"
        print(f"[Model Loader] ONNX Runtime AudioCNN loaded successfully from: {onnx_path}")

    return _MODEL_INSTANCE, _DEVICE, _MODEL_PATH
