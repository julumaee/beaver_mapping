"""Prithvi-100M encoder wrapper, band adapter, BeaverCNN classifier, and inference helpers."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    from huggingface_hub import hf_hub_download
    _DL_AVAILABLE = True
except ImportError:
    _DL_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 if DEVICE.type == "cuda" else 1

_PRITHVI_REPO = "ibm-nasa-geospatial/Prithvi-100M"
_PRITHVI_FILE = "Prithvi_100M.pt"
_INPUT_SIZE = 224

# MML CIR band order: NIR=0, Red=1, Green=2
# HLS band order:     Blue=0, Green=1, Red=2, NIR=3, SWIR1=4, SWIR2=5
# Mapping: HLS[1]=Green <- MML[2], HLS[2]=Red <- MML[1], HLS[3]=NIR <- MML[0]
# HLS[0]=Blue, HLS[4]=SWIR1, HLS[5]=SWIR2 are zero-filled.
_MML_TO_HLS_SLOTS = [(1, 2), (2, 1), (3, 0)]  # (hls_idx, mml_idx)


class BandAdapter(nn.Module):
    """Map a 3-band MML CIR tensor to a 6-band HLS tensor (zero-fill missing channels)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        out = torch.zeros(B, 6, H, W, dtype=x.dtype, device=x.device)
        for hls_idx, mml_idx in _MML_TO_HLS_SLOTS:
            out[:, hls_idx] = x[:, mml_idx]
        return out


class BeaverCNN(nn.Module):
    """Frozen Prithvi-100M ViT encoder + trainable 2-class MLP head."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.band_adapter = BandAdapter()
        self.encoder = _build_encoder()
        embed_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        _freeze(self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) float32, normalised
        x = self.band_adapter(x)       # (B, 6, H, W)
        features = self.encoder(x)     # (B, embed_dim)
        return self.head(features)     # (B, num_classes)


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def _build_encoder() -> nn.Module:
    if not _DL_AVAILABLE:
        raise ImportError("torch, timm, and huggingface_hub are required for CNN inference")
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        in_chans=6,
        num_classes=0,
        global_pool="avg",
    )
    _try_load_prithvi_weights(model)
    return model


def _try_load_prithvi_weights(model: nn.Module) -> None:
    """Download Prithvi-100M and load all compatible encoder weights."""
    try:
        weights_path = hf_hub_download(repo_id=_PRITHVI_REPO, filename=_PRITHVI_FILE)
        state = torch.load(weights_path, map_location="cpu")
        if "model" in state:
            state = state["model"]

        # Strip optional 'encoder.' prefix from Prithvi keys
        stripped = {k.removeprefix("encoder."): v for k, v in state.items()}

        # Prithvi patch_embed weight has an extra temporal dimension [D, C, t, H, W].
        # Squeeze t=1 to match the standard timm ViT shape [D, C, H, W].
        if "patch_embed.proj.weight" in stripped:
            w = stripped["patch_embed.proj.weight"]
            if w.ndim == 5 and w.shape[2] == 1:
                stripped["patch_embed.proj.weight"] = w.squeeze(2)

        missing, unexpected = model.load_state_dict(stripped, strict=False)
        n_loaded = len(stripped) - len(missing)
        print(f"Prithvi weights: {n_loaded}/{len(stripped)} keys loaded "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")
    except Exception as exc:
        print(f"Warning: could not load Prithvi weights ({exc}). "
              "Encoder will be randomly initialised.")


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_cnn(model: BeaverCNN, model_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def load_cnn(model_path: str) -> BeaverCNN:
    model = BeaverCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_cnn(
    model: BeaverCNN,
    chip: np.ndarray,
    norm_stats: dict | None,
) -> tuple[int, float]:
    """Single-chip inference. Returns (label, confidence)."""
    tensor = _chip_to_tensor(chip, norm_stats).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    label = int(probs.argmax().item())
    return label, float(probs[label].item())


def predict_cnn_batch(
    model: BeaverCNN,
    chips: list[np.ndarray],
    norm_stats: dict | None,
) -> list[tuple[int, float]]:
    """Batch inference. Returns [(label, confidence), ...]."""
    tensors = torch.stack([_chip_to_tensor(c, norm_stats) for c in chips]).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensors), dim=1)
    results = []
    for row in probs:
        label = int(row.argmax().item())
        results.append((label, float(row[label].item())))
    return results


def _chip_to_tensor(chip: np.ndarray, norm_stats: dict | None) -> torch.Tensor:
    """(3, H, W) uint8 -> (3, 224, 224) float32, optionally normalised."""
    t = torch.from_numpy(chip.astype(np.float32)) / 255.0
    t = F.interpolate(
        t.unsqueeze(0), size=(_INPUT_SIZE, _INPUT_SIZE),
        mode="bilinear", align_corners=False,
    )[0]
    if norm_stats is not None:
        mean = torch.tensor(norm_stats["mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(norm_stats["std"], dtype=torch.float32).view(3, 1, 1)
        t = (t - mean) / (std + 1e-6)
    return t
