"""Prithvi-EO-1.0-100M encoder wrapper, band adapter, BeaverCNN classifier, and inference helpers.

Uses the official PrithviMAE implementation downloaded from HuggingFace, giving exact
weight loading with no architecture mismatches. The frozen ViT-Base encoder (embed_dim=768,
depth=12) feeds a trainable MLP head. Only the head weights are saved to disk; the encoder
is re-fetched from HF cache on load.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 if DEVICE.type == "cuda" else 1

_PRITHVI_REPO    = "ibm-nasa-geospatial/prithvi-eo-1.0-100M"
_PRITHVI_WEIGHTS = "Prithvi_EO_V1_100M.pt"
_PRITHVI_SOURCE  = "prithvi_mae.py"
_EMBED_DIM       = 768   # ViT-Base config of the 100M checkpoint
_INPUT_SIZE      = 224

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
    """Frozen Prithvi-EO ViT-Base encoder + trainable 2-class MLP head."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.band_adapter = BandAdapter()
        self.encoder = _build_prithvi_encoder()
        self.head = nn.Sequential(
            nn.LayerNorm(_EMBED_DIM),
            nn.Linear(_EMBED_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        _freeze(self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) float32, normalised
        x = self.band_adapter(x)          # (B, 6, H, W)
        x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (B, 6, 3, H, W) — repeat to match T=3 checkpoint
        features_list = self.encoder.forward_features(x)
        # forward_features returns a list of (B, num_patches+1, embed_dim) tensors.
        # Index 0 of the sequence is the CLS token from the last block.
        cls = features_list[-1][:, 0]     # (B, embed_dim)
        return self.head(cls)             # (B, num_classes)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_prithvi_encoder() -> nn.Module:
    if not _HF_AVAILABLE:
        raise ImportError("huggingface_hub is required — pip install huggingface_hub")

    # Allow placing weights manually in data/models/ to avoid HF download issues
    local_weights = Path("data/models") / _PRITHVI_WEIGHTS
    local_src     = Path("data/models") / _PRITHVI_SOURCE

    src_path     = str(local_src)     if local_src.exists()     else hf_hub_download(repo_id=_PRITHVI_REPO, filename=_PRITHVI_SOURCE)
    weights_path = str(local_weights) if local_weights.exists() else hf_hub_download(repo_id=_PRITHVI_REPO, filename=_PRITHVI_WEIGHTS)

    # Import PrithviMAE from the downloaded source file
    src_dir = str(Path(src_path).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from prithvi_mae import PrithviMAE  # noqa: PLC0415

    # Instantiate encoder-only model with the 100M ViT-Base config (T=1 for single images).
    # Sin-cos 3D positional embeddings are computed dynamically so T=1 is compatible
    # with weights trained on T=3.
    mae = PrithviMAE(
        img_size=224,
        patch_size=(1, 16, 16),
        num_frames=3,   # match checkpoint; forward() repeats the single frame 3×
        in_chans=6,
        embed_dim=_EMBED_DIM,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        encoder_only=True,
    )

    state = torch.load(weights_path, map_location="cpu")
    if "model" in state:
        state = state["model"]

    missing, unexpected = mae.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(missing)
    print(f"Prithvi-EO weights: {n_loaded}/{len(state)} keys loaded "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")

    return mae.encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


# ---------------------------------------------------------------------------
# Persistence — save only the trainable head to keep the .pth file small.
# The encoder is always re-fetched from the HF cache on load.
# ---------------------------------------------------------------------------

def save_cnn(model: BeaverCNN, model_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.head.state_dict(), model_path)


def load_cnn(model_path: str) -> BeaverCNN:
    model = BeaverCNN()
    model.head.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
        std  = torch.tensor(norm_stats["std"],  dtype=torch.float32).view(3, 1, 1)
        t = (t - mean) / (std + 1e-6)
    return t
