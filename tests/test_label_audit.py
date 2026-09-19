"""Tests for src/label_audit.py (R1.5 / G2.2)."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import label_audit


def _write_oof(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "oof.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "feature_type", "label", "x", "y", "cluster", "fold", "prob"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return str(path)


def _row(path, ftype, label, x, y, prob, cluster=0, fold=0):
    return {
        "path": path, "feature_type": ftype, "label": label,
        "x": x, "y": y, "cluster": cluster, "fold": fold, "prob": prob,
    }


class TestAuditLabels:
    def test_no_suspicious_points_returns_empty(self, tmp_path):
        rows = [
            _row("a.npy", "flood", 1, 100.0, 200.0, 0.9),
            _row("b.npy", "negative", 0, 300.0, 400.0, 0.05),
        ]
        oof = _write_oof(tmp_path, rows)
        assert label_audit.audit_labels(oof) == []

    def test_low_confidence_positive_is_flagged(self, tmp_path):
        rows = [_row("a.npy", "dead_forest", 1, 100.0, 200.0, 0.1)]
        oof = _write_oof(tmp_path, rows)
        result = label_audit.audit_labels(oof, low=0.2, high=0.8)
        assert len(result) == 1
        r = result[0]
        assert r["feature_type"] == "dead_forest"
        assert r["label"] == 1
        assert r["mean_prob"] == 0.1
        assert r["x"] == 100.0 and r["y"] == 200.0
        assert r["n_chips"] == 1

    def test_high_confidence_hand_labelled_negative_is_flagged(self, tmp_path):
        rows = [_row("a.npy", "negative", 0, 100.0, 200.0, 0.95)]
        oof = _write_oof(tmp_path, rows)
        result = label_audit.audit_labels(oof, low=0.2, high=0.8)
        assert len(result) == 1
        assert result[0]["feature_type"] == "negative"
        assert result[0]["label"] == 0

    def test_auto_negative_excluded_by_default(self, tmp_path):
        rows = [_row("a.npy", "auto_negative", 0, 100.0, 200.0, 0.95)]
        oof = _write_oof(tmp_path, rows)
        assert label_audit.audit_labels(oof, low=0.2, high=0.8) == []

    def test_auto_negative_included_when_requested(self, tmp_path):
        rows = [_row("a.npy", "auto_negative", 0, 100.0, 200.0, 0.95)]
        oof = _write_oof(tmp_path, rows)
        result = label_audit.audit_labels(oof, low=0.2, high=0.8, include_auto_negatives=True)
        assert len(result) == 1
        assert result[0]["feature_type"] == "auto_negative"

    def test_augmented_chips_averaged_into_one_point(self, tmp_path):
        # Same (x, y, feature_type) — like a base chip + augmented offsets.
        rows = [
            _row("a.npy", "flood", 1, 100.0, 200.0, 0.5),
            _row("a_aug0.npy", "flood", 1, 100.0, 200.0, 0.1),
            _row("a_aug1.npy", "flood", 1, 100.0, 200.0, 0.05),
        ]
        oof = _write_oof(tmp_path, rows)
        result = label_audit.audit_labels(oof, low=0.3, high=0.8)
        assert len(result) == 1
        r = result[0]
        assert r["n_chips"] == 3
        assert abs(r["mean_prob"] - (0.5 + 0.1 + 0.05) / 3) < 1e-9

    def test_sorted_most_suspicious_first(self, tmp_path):
        rows = [
            _row("a.npy", "flood", 1, 1.0, 1.0, 0.19),   # score 0.01
            _row("b.npy", "flood", 1, 2.0, 2.0, 0.0),    # score 0.2
            _row("c.npy", "negative", 0, 3.0, 3.0, 0.99),  # score 0.19
        ]
        oof = _write_oof(tmp_path, rows)
        result = label_audit.audit_labels(oof, low=0.2, high=0.8)
        assert [r["x"] for r in result] == [2.0, 3.0, 1.0]

    def test_different_points_at_same_xy_kept_separate_by_type(self, tmp_path):
        rows = [
            _row("a.npy", "flood", 1, 100.0, 200.0, 0.1),
            _row("b.npy", "negative", 0, 100.0, 200.0, 0.9),
        ]
        oof = _write_oof(tmp_path, rows)
        result = label_audit.audit_labels(oof, low=0.2, high=0.8)
        assert len(result) == 2
        types = {r["feature_type"] for r in result}
        assert types == {"flood", "negative"}
