"""Tests for src/cli.py — detect threshold resolution and classifier-config
loading for train --classifier-config (R3.5 / R3.1)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli import _load_classifier_config, _resolve_detect_threshold


class TestResolveDetectThreshold:
    """Precedence: explicit --threshold > model metadata sidecar
    recommended_threshold > 0.5."""

    def test_explicit_always_wins(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        (tmp_path / "model.json").write_text(json.dumps({"recommended_threshold": 0.2}))
        assert _resolve_detect_threshold(0.9, "rf", str(model_path)) == 0.9

    def test_falls_back_to_sidecar_recommended_threshold(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        (tmp_path / "model.json").write_text(json.dumps({"recommended_threshold": 0.33}))
        assert _resolve_detect_threshold(None, "rf", str(model_path)) == 0.33

    def test_falls_back_to_sidecar_for_both_method(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        (tmp_path / "model.json").write_text(json.dumps({"recommended_threshold": 0.4}))
        assert _resolve_detect_threshold(None, "both", str(model_path)) == 0.4

    def test_falls_back_to_default_when_no_sidecar(self, tmp_path):
        model_path = tmp_path / "model.pkl"  # no .json sidecar written
        assert _resolve_detect_threshold(None, "rf", str(model_path)) == 0.5

    def test_falls_back_to_default_when_sidecar_threshold_is_null(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        (tmp_path / "model.json").write_text(json.dumps({"recommended_threshold": None}))
        assert _resolve_detect_threshold(None, "rf", str(model_path)) == 0.5

    def test_cnn_method_ignores_rf_sidecar(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        (tmp_path / "model.json").write_text(json.dumps({"recommended_threshold": 0.33}))
        assert _resolve_detect_threshold(None, "cnn", str(model_path)) == 0.5

    def test_no_rf_model_path_falls_back_to_default(self):
        assert _resolve_detect_threshold(None, "rf", None) == 0.5


class TestLoadClassifierConfig:
    def test_raw_config(self, tmp_path):
        cfg = {"type": "hgb", "params": {"learning_rate": 0.1}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))
        assert _load_classifier_config(str(path)) == cfg

    def test_tune_output_uses_best_config(self, tmp_path):
        cfg = {"type": "extra_trees", "params": {"n_estimators": 300}}
        tuning = {"results": [], "best": {"name": "extra_trees_leaf1", "config": cfg}}
        path = tmp_path / "tuning.json"
        path.write_text(json.dumps(tuning))
        assert _load_classifier_config(str(path)) == cfg

    def test_unrecognized_file_exits(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"nonsense": True}))
        with pytest.raises(SystemExit):
            _load_classifier_config(str(path))
