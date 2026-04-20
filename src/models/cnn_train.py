"""AdamW training loop for BeaverCNN: freezes Prithvi encoder, trains head only."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from models.cnn_handler import BeaverCNN, DEVICE, BATCH_SIZE, save_cnn
from models.cnn_dataset import ChipDataset, compute_norm_stats


def train_cnn(
    manifest_path: str,
    model_path: str,
    norm_stats_path: str,
    epochs: int = 30,
    lr: float = 1e-3,
    val_split: float = 0.2,
) -> BeaverCNN:
    """
    Compute norm stats, build datasets, train the classification head with AdamW
    and CosineAnnealingLR, save the best checkpoint, return the trained model.
    """
    norm_stats = compute_norm_stats(manifest_path, norm_stats_path)

    # Build a non-augmented dataset for validation and index splitting
    base_ds = ChipDataset(manifest_path, norm_stats=norm_stats, augment=False)
    n_val = max(1, int(len(base_ds) * val_split))
    n_train = len(base_ds) - n_val
    train_split, val_split_ds = random_split(
        base_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Augmented copy used only for the training subset
    aug_ds = ChipDataset(manifest_path, norm_stats=norm_stats, augment=True)
    train_loader = DataLoader(
        Subset(aug_ds, train_split.indices),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(base_ds, val_split_ds.indices),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = BeaverCNN().to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for chips, labels in train_loader:
            chips, labels = chips.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(chips), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for chips, labels in val_loader:
                chips, labels = chips.to(DEVICE), labels.to(DEVICE)
                preds = model(chips).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_acc = correct / total if total > 0 else 0.0
        scheduler.step()
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss / n_train:.4f}  val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_cnn(model, model_path)

    print(f"\nBest val_acc={best_val_acc:.3f} — model saved to {model_path}")
    return model
