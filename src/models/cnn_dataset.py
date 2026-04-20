"""Dataset, data augmentation, and normalization-stats computation for CNN training."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_INPUT_SIZE = 224


class ChipDataset(Dataset):
    """Loads chips listed in a manifest CSV as normalised float32 tensors."""

    def __init__(
        self,
        manifest_path: str,
        norm_stats: dict | None = None,
        augment: bool = False,
    ) -> None:
        with open(manifest_path) as f:
            self.rows = list(csv.DictReader(f))
        self.norm_stats = norm_stats
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        chip = np.load(row["path"])
        label = int(row["label"])
        tensor = _to_tensor(chip, self.norm_stats)
        if self.augment:
            tensor = _augment(tensor)
        return tensor, label


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment(t: torch.Tensor) -> torch.Tensor:
    """Random horizontal/vertical flip, 0/90/180/270° rotation, brightness jitter."""
    if random.random() > 0.5:
        t = t.flip(dims=[2])  # horizontal flip
    if random.random() > 0.5:
        t = t.flip(dims=[1])  # vertical flip
    k = random.randint(0, 3)
    if k:
        t = torch.rot90(t, k, dims=[1, 2])
    # Per-band brightness jitter in [0.8, 1.2]
    factors = torch.empty(t.shape[0], 1, 1).uniform_(0.8, 1.2)
    t = t * factors
    return t


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def compute_norm_stats(manifest_path: str, save_path: str) -> dict:
    """
    Compute per-band mean and std over all chips in the manifest (pixel-wise,
    bands scaled to [0, 1]) and save to save_path as JSON.
    """
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    n_bands = 3
    n_pixels = 0
    band_sum = np.zeros(n_bands, dtype=np.float64)
    band_sum_sq = np.zeros(n_bands, dtype=np.float64)

    for row in rows:
        chip = np.load(row["path"]).astype(np.float64) / 255.0  # (3, H, W)
        px = chip.shape[1] * chip.shape[2]
        n_pixels += px
        for b in range(n_bands):
            band_sum[b] += chip[b].sum()
            band_sum_sq[b] += (chip[b] ** 2).sum()

    mean = (band_sum / n_pixels).tolist()
    var = band_sum_sq / n_pixels - (band_sum / n_pixels) ** 2
    std = np.sqrt(np.maximum(var, 0.0)).tolist()

    stats = {"mean": mean, "std": std}
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Norm stats saved to {save_path}")
    print(f"  mean: {[f'{m:.4f}' for m in mean]}")
    print(f"  std:  {[f'{s:.4f}' for s in std]}")
    return stats


# ---------------------------------------------------------------------------
# Tensor helper (shared with cnn_handler)
# ---------------------------------------------------------------------------

def _to_tensor(chip: np.ndarray, norm_stats: dict | None) -> torch.Tensor:
    """(3, H, W) uint8 -> (3, 224, 224) float32, optionally normalised."""
    import torch.nn.functional as F
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
