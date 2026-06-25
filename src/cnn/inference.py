"""
Inference wrapper for ForgeryDetector.

Used by web_app.py to get a CNN genuineness score.
If the weights file does not exist yet, predict_genuine() returns None
and the caller falls back to the ELA heuristic score.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent / "weights" / "forgery_cnn.pt"
IMG_SIZE = (224, 224)

_PREPROCESS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None


def model_ready() -> bool:
    """True when trained weights exist on disk."""
    return WEIGHTS_PATH.exists()


def _load():
    global _model
    if _model is not None:
        return _model
    if not WEIGHTS_PATH.exists():
        return None
    from .model import ForgeryDetector
    m = ForgeryDetector(pretrained=False)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    m.eval()
    _model = m
    return _model


@torch.no_grad()
def predict_genuine(ela_image: Image.Image) -> float | None:
    """
    Score an ELA image with the trained CNN.

    Returns:
        Float in [0, 1] — 1.0 = confident Genuine, 0.0 = confident Forged.
        None if the model has not been trained yet (caller uses ELA fallback).
    """
    model = _load()
    if model is None:
        return None
    tensor = _PREPROCESS(ela_image.convert("RGB")).unsqueeze(0)  # (1, 3, H, W)
    return float(model(tensor)[0])
