"""Tests for src/models/evaluate.py — pooled out-of-fold spatial CV."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.evaluate import evaluate_rf_spatial, evaluate_rf_per_class, _spatial_clusters


# 6 well-separated cluster origins, alternating positive/negative, 2 label
# points per cluster, 2 chips per point (original + "augmented" duplicate
# sharing x, y) — small enough to run fast, spatially separated enough that
# clustering and pooled CV both have something real to do.
_CLUSTER_ORIGINS = [(0, 0), (10_000, 0), (20_000, 0), (0, 10_000), (10_000, 10_000), (20_000, 10_000)]


def _make_manifest(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    rows = []

    for ci, (ox, oy) in enumerate(_CLUSTER_ORIGINS):
        label = 1 if ci % 2 == 0 else 0
        ftype = "flood" if label == 1 else "negative"
        for p in range(2):  # two label points per cluster, well within any sane cluster radius
            x, y = ox + p * 5, oy + p * 5
            for aug in range(2):  # original + "augmented" chip sharing x, y
                if label == 1:
                    chip = np.zeros((3, 64, 64), dtype=np.uint8)
                    chip[0] = rng.integers(30, 60, (64, 64))   # NIR low
                    chip[1] = rng.integers(40, 80, (64, 64))
                    chip[2] = rng.integers(150, 200, (64, 64))  # Green high -> wet signal
                else:
                    chip = rng.integers(60, 100, (3, 64, 64), dtype=np.uint8)
                fname = f"c{ci}_p{p}_a{aug}.npy"
                fpath = tmp_path / fname
                np.save(fpath, chip)
                rows.append({"path": str(fpath), "label": label, "feature_type": ftype, "x": x, "y": y})

    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "feature_type", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)
    return str(manifest)


class TestSpatialClusters:
    def test_separated_points_form_separate_clusters(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        clusters = _spatial_clusters(rows, radius=500.0)
        assert len(set(clusters)) == len(_CLUSTER_ORIGINS)

    def test_shared_points_in_same_cluster(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        clusters = _spatial_clusters(rows, radius=500.0)
        # rows come in groups of 4 (2 points x 2 aug chips) per cluster origin
        for i in range(0, len(rows), 4):
            assert len(set(clusters[i:i + 4])) == 1


class TestEvaluateRFSpatial:
    def test_oof_csv_columns_and_row_count(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        with open(manifest) as f:
            manifest_rows = list(csv.DictReader(f))

        result = evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)

        oof_path = Path(result["oof_csv"])
        assert oof_path.exists()
        with open(oof_path) as f:
            oof_rows = list(csv.DictReader(f))
        assert set(oof_rows[0].keys()) == {
            "path", "feature_type", "label", "x", "y", "cluster", "fold", "prob",
        }
        # Pooled OOF covers every chip exactly once.
        assert len(oof_rows) == len(manifest_rows)

    def test_metrics_keys(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        result = evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)
        for key in (
            "roc_auc", "pr_auc", "recommended_threshold",
            "metrics_at_0.5", "metrics_at_recommended",
            "point_metrics_at_0.5", "point_metrics_at_recommended",
            "confusion_matrix_at_recommended", "per_type", "n_points", "n_clusters",
        ):
            assert key in result
        for m_key in ("metrics_at_0.5", "metrics_at_recommended"):
            for sub in ("accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"):
                assert sub in result[m_key]

    def test_pooled_recall_not_dragged_to_zero_by_single_class_groups(self, tmp_path):
        # Every spatial cluster here is single-class (all-positive or all-negative), the exact
        # situation that made naive per-fold LOCO averaging report meaningless 0 recall/precision
        # for single-class folds. Pooling OOF probabilities before scoring should recover the
        # real (strong, by construction) signal instead.
        manifest = _make_manifest(tmp_path)
        result = evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)
        assert result["metrics_at_recommended"]["recall"] > 0.0
        assert result["roc_auc"] > 0.5

    def test_point_level_deduplicates_augmented_chips(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        result = evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)
        # 6 clusters x 2 points = 12 distinct label points, vs 24 chips (2 aug per point).
        assert result["n_points"] == 12
        assert result["n_chips"] == 24

    def test_per_type_contains_actual_type_names(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        result = evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)
        assert "flood" in result["per_type"]
        assert "negative" in result["per_type"]
        assert "wet_forest" not in result["per_type"]
        assert "beaver_flood" not in result["per_type"]
        # Positive type reports recall, negative type reports specificity.
        assert "recall" in result["per_type"]["flood"]
        assert "specificity" in result["per_type"]["negative"]

    def test_feature_cache_reused(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)
        cache_path = Path(manifest).parent / "features.npy"
        assert cache_path.exists()
        mtime1 = cache_path.stat().st_mtime

        evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)
        mtime2 = cache_path.stat().st_mtime
        assert mtime1 == mtime2  # not recomputed

    def test_no_cache_forces_recompute(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)
        cache_path = Path(manifest).parent / "features.npy"
        mtime1_ns = cache_path.stat().st_mtime_ns

        import time
        time.sleep(1.1)  # some filesystems only offer 1s mtime resolution
        evaluate_rf_spatial(manifest, cluster_radius=500.0, n_splits=3, random_seed=0, use_cache=False)
        mtime2_ns = cache_path.stat().st_mtime_ns
        assert mtime2_ns > mtime1_ns


class TestEvaluateRFPerClass:
    def test_wrapper_returns_per_type_dict(self, tmp_path):
        manifest = _make_manifest(tmp_path)
        result = evaluate_rf_per_class(manifest, cluster_radius=500.0, n_splits=3, random_seed=0)
        assert "flood" in result and "negative" in result
