"""Tests for src/classifier.py."""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.random_forest import (
    build_feature_matrix,
    check_feature_length,
    load_model,
    load_model_metadata,
    make_classifier,
    predict,
    save_model,
    train,
)
from spectral import FEATURE_VECTOR_LENGTH


def _make_chips_and_manifest(tmp_path: Path, n=10) -> str:
    """Write n positive and n negative synthetic chips, return manifest path."""
    rng = np.random.default_rng(0)
    rows = []

    for i in range(n):
        # Flood (label=1): high green, low NIR
        chip = np.zeros((3, 512, 512), dtype=np.uint8)
        chip[0] = rng.integers(30, 60, (512, 512))    # NIR low
        chip[1] = rng.integers(40, 80, (512, 512))
        chip[2] = rng.integers(150, 200, (512, 512))  # Green high
        p = tmp_path / f"flood_{i}.npy"
        np.save(str(p), chip)
        rows.append({"path": str(p), "label": 1, "feature_type": "wet_forest", "x": 0, "y": 0})

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
        assert X.shape == (6, FEATURE_VECTOR_LENGTH)  # 2 classes × 3 samples each
        assert y.shape == (6,)

    def test_labels_correct(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path, n=4)
        _, y = build_feature_matrix(manifest)
        assert list(y[:4]) == [1, 1, 1, 1]   # flood
        assert list(y[4:]) == [0, 0, 0, 0]   # negative


class TestTrainAndPredict:
    def test_predict_returns_label_and_confidence(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        clf = train(manifest, model_path)

        rng = np.random.default_rng(7)
        chip = rng.integers(60, 100, (3, 512, 512), dtype=np.uint8)
        label, confidence = predict(clf, chip)
        assert label in (0, 1)
        assert 0.0 <= confidence <= 1.0

    def test_model_saved_and_loadable(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        clf = train(manifest, model_path)

        loaded = load_model(model_path)
        rng = np.random.default_rng(3)
        chip = rng.integers(60, 100, (3, 512, 512), dtype=np.uint8)
        assert predict(clf, chip) == predict(loaded, chip)

    def test_flood_chip_classified_positive(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path, n=20)
        model_path = str(tmp_path / "model.pkl")
        clf = train(manifest, model_path)

        # Very strong flood signal: low NIR, high green
        chip = np.zeros((3, 512, 512), dtype=np.uint8)
        chip[0] = 40   # NIR low
        chip[1] = 60   # Red mid
        chip[2] = 190  # Green high
        label, _ = predict(clf, chip)
        assert label == 1


class TestMakeClassifierConfigs:
    """R3.1/R3.2 — make_classifier must build a working, fittable estimator
    for every supported config type, and stay backwards compatible with the
    original config=None + **overrides call style."""

    def _fit_predict(self, clf) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 10)).astype(np.float32)
        y = (rng.random(60) > 0.5).astype(int)
        groups = np.repeat(np.arange(20), 3)
        clf.fit(X, y)
        proba = clf.predict_proba(X[:5])
        assert proba.shape == (5, 2)
        assert clf.n_features_in_ == 10
        return groups

    def test_default_backwards_compatible(self):
        clf = make_classifier(random_state=1)
        self._fit_predict(clf)
        assert type(clf).__name__ == "RandomForestClassifier"

    def test_rf_config(self):
        clf = make_classifier(
            config={"type": "rf", "params": {"min_samples_leaf": 3, "max_features": 0.3,
                                              "n_estimators": 20}},
            random_state=1,
        )
        self._fit_predict(clf)
        assert type(clf).__name__ == "RandomForestClassifier"
        assert clf.min_samples_leaf == 3

    def test_extra_trees_config(self):
        clf = make_classifier(
            config={"type": "extra_trees", "params": {"n_estimators": 20, "min_samples_leaf": 5}},
            random_state=1,
        )
        self._fit_predict(clf)
        assert type(clf).__name__ == "ExtraTreesClassifier"

    def test_hgb_config(self):
        clf = make_classifier(
            config={"type": "hgb", "params": {"learning_rate": 0.1, "max_leaf_nodes": 15}},
            random_state=1,
        )
        self._fit_predict(clf)
        assert type(clf).__name__ == "HistGradientBoostingClassifier"

    def test_rf_calibrated_config(self):
        clf = make_classifier(
            config={"type": "rf_calibrated", "params": {"n_estimators": 20, "method": "isotonic"}},
            random_state=1,
        )
        groups = self._fit_predict(clf)
        assert type(clf).__name__ == "CalibratedClassifierCV"
        _ = groups  # calibration also works when groups is supplied at construction time
        calibrated_with_groups = make_classifier(
            config={"type": "rf_calibrated", "params": {"n_estimators": 20}},
            groups=groups, random_state=1,
        )
        self._fit_predict(calibrated_with_groups)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            make_classifier(config={"type": "not_a_real_classifier", "params": {}})


class TestModelMetadataSidecar:
    """R3.5 — train() writes a JSON sidecar with the documented schema."""

    def test_sidecar_written_with_required_keys(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        train(manifest, model_path)

        sidecar_path = tmp_path / "model.json"
        assert sidecar_path.exists()
        with open(sidecar_path) as f:
            meta = json.load(f)

        for key in (
            "format_version", "created", "classifier", "feature_length",
            "label_map", "chips_by_type", "n_chips", "recommended_threshold", "cv",
        ):
            assert key in meta

        assert meta["format_version"] == 1
        assert meta["feature_length"] == FEATURE_VECTOR_LENGTH
        assert meta["n_chips"] == 20
        assert meta["chips_by_type"] == {"wet_forest": 10, "negative": 10}
        assert meta["label_map"] == {"wet_forest": 1, "negative": 0}
        assert meta["classifier"]["type"] == "RandomForestClassifier"
        # No CV was run (train() called directly, without cv_results) — both
        # CV-derived fields must be null, not missing or fabricated.
        assert meta["cv"] is None
        assert meta["recommended_threshold"] is None

    def test_load_model_metadata_roundtrip(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        train(manifest, model_path)

        meta = load_model_metadata(model_path)
        assert meta is not None
        assert meta["n_chips"] == 20

    def test_load_model_metadata_missing_returns_none(self, tmp_path):
        assert load_model_metadata(str(tmp_path / "nonexistent.pkl")) is None

    def test_cv_results_embedded_when_supplied(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        fake_cv = {
            "roc_auc": 0.9, "pr_auc": 0.85, "recommended_threshold": 0.33,
            "high_recall_threshold": 0.2,
            "metrics_at_recommended": {"precision": 0.9, "recall": 0.8},
            "point_metrics_at_recommended": {"precision": 0.9, "recall": 0.8},
            "per_type": {"wet_forest": {"recall": 0.8}},
        }
        train(manifest, model_path, cv_results=fake_cv)

        meta = load_model_metadata(model_path)
        assert meta["recommended_threshold"] == 0.33
        assert meta["cv"]["roc_auc"] == 0.9
        assert meta["cv"]["pr_auc"] == 0.85
        assert meta["cv"]["per_type"] == {"wet_forest": {"recall": 0.8}}


class TestCheckFeatureLength:
    """R3.5 — detect must fail clearly if a model was trained with a
    different feature-vector length than the current extract_features()."""

    def test_matching_length_passes(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        clf = train(manifest, model_path)
        check_feature_length(clf)  # must not raise

    def test_mismatched_length_raises_clear_error(self, tmp_path):
        manifest = _make_chips_and_manifest(tmp_path)
        model_path = str(tmp_path / "model.pkl")
        clf = train(manifest, model_path)
        clf.n_features_in_ = clf.n_features_in_ + 1  # simulate an older/newer feature set

        with pytest.raises(ValueError, match="retrain"):
            check_feature_length(clf)

    def test_unfitted_or_missing_attr_is_a_noop(self):
        class _NoAttr:
            pass
        check_feature_length(_NoAttr())  # no n_features_in_ -> nothing to check, no raise
