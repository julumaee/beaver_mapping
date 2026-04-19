"""Tests for src/classifier.py (Task 4.2)."""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from classifier import train, predict, save_model, load_model, build_feature_matrix


def _make_chips_and_manifest(tmp_path: Path, n=10) -> str:
    """Write n positive and n negative synthetic chips, return manifest path."""
    rng = np.random.default_rng(0)
    rows = []

    for i in range(n):
        # Dam (label=1): high NIR, low red/green
        chip = np.zeros((3, 512, 512), dtype=np.uint8)
        chip[0] = rng.integers(150, 220, (512, 512))
        chip[1] = rng.integers(40, 80, (512, 512))
        chip[2] = rng.integers(40, 80, (512, 512))
        p = tmp_path / f"dam_{i}.npy"
        np.save(str(p), chip)
        rows.append({"path": str(p), "label": 1, "feature_type": "dam", "x": 0, "y": 0})

    for i in range(n):
        # Flood (label=2): high green, low NIR
        chip = np.zeros((3, 512, 512), dtype=np.uint8)
        chip[0] = rng.integers(30, 60, (512, 512))   # NIR low
        chip[1] = rng.integers(40, 80, (512, 512))
        chip[2] = rng.integers(150, 200, (512, 512)) # Green high
        p = tmp_path / f"flood_{i}.npy"
        np.save(str(p), chip)
        rows.append({"path": str(p), "label": 2, "feature_type": "wet_forest", "x": 0, "y": 0})

    for i in range(n):
        # Negative (label=0): balanced bands
        chip = rng.integers(60, 100, (3, 512, 512), dtype=np.uint8)
        p = tmp_path / f"neg_{i}.npy"
        np.save(str(p), chip)
        rows.append({"path": str(p), "label": 0, "feature_type": "negative", "x": 0, "y": 0})

    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "feature_type", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)

    return str(manifest)


class TestBuildFeatureMatrix:
    def test_shape(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path, n=3)
        X, y = build_feature_matrix(manifest)
        assert X.shape == (9, 36)  # 3 classes × 3 samples each
        assert y.shape == (9,)

    def test_labels_correct(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path, n=4)
        _, y = build_feature_matrix(manifest)
        assert list(y[:4]) == [1, 1, 1, 1]   # dam
        assert list(y[4:8]) == [2, 2, 2, 2]  # flood
        assert list(y[8:]) == [0, 0, 0, 0]   # negative


class TestTrainAndPredict:
    def test_predict_returns_label_and_confidence(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        clf = train(manifest, model_path)

        rng = np.random.default_rng(7)
        chip = rng.integers(60, 100, (3, 512, 512), dtype=np.uint8)
        label, confidence = predict(clf, chip)
        assert label in (0, 1, 2)
        assert 0.0 <= confidence <= 1.0

    def test_model_saved_and_loadable(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        clf = train(manifest, model_path)

        loaded = load_model(model_path)
        rng = np.random.default_rng(3)
        chip = rng.integers(60, 100, (3, 512, 512), dtype=np.uint8)
        assert predict(clf, chip) == predict(loaded, chip)

    def test_positive_chip_classified_positive(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path, n=20)
        model_path = str(tmp_path / "model.pkl")
        clf = train(manifest, model_path)

        # Very strong positive signal
        chip = np.zeros((3, 512, 512), dtype=np.uint8)
        chip[0] = 200  # NIR very high
        chip[1] = 40   # Red low
        chip[2] = 40   # Green low
        label, _ = predict(clf, chip)
        assert label == 1
