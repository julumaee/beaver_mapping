"""Tests for the G1.1 (path derivation), G1.4 (settings autosave/migration)
and R1.5/G2.2 (audit wiring) helpers in src/app.py.

These are pure functions — no Gradio launch needed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import app


class TestDerivePaths:
    def test_default_project_dir(self):
        # An empty project dir falls back to the detected data directory, which
        # is the repo's own data/ or — from a worktree — the main checkout's.
        root = Path(app.DEFAULT_PROJECT_DIR)
        assert root.name == "data"
        paths = app.derive_paths("")
        assert paths["project_dir"] == str(root)
        assert paths["models_dir"] == str(root / "models")
        assert paths["rf_model"] == str(root / "models/model.pkl")
        assert paths["rf_sidecar"] == str(root / "models/model.json")
        assert paths["chips_dir"] == str(root / "chips")
        assert paths["manifest"] == str(root / "chips/manifest.csv")
        assert paths["oof_csv"] == str(root / "chips/oof.csv")
        assert paths["cnn_model"] == str(root / "models/beaver_cnn_v1.pth")
        assert paths["norm_stats"] == str(root / "models/norm_stats.json")
        assert paths["output_dir"] == str(root / "output")

    def test_custom_project_dir(self):
        paths = app.derive_paths("/tmp/myproj")
        assert paths["project_dir"] == str(Path("/tmp/myproj"))
        assert paths["rf_model"] == str(Path("/tmp/myproj/models/model.pkl"))

    def test_rf_model_override_changes_only_rf_model_and_sidecar(self):
        paths = app.derive_paths("/tmp/myproj", "/tmp/other/custom_rf.pkl")
        assert paths["rf_model"] == str(Path("/tmp/other/custom_rf.pkl"))
        assert paths["rf_sidecar"] == str(Path("/tmp/other/custom_rf.json"))
        # Everything else is unaffected by the override.
        assert paths["chips_dir"] == str(Path("/tmp/myproj/chips"))
        assert paths["cnn_model"] == str(Path("/tmp/myproj/models/beaver_cnn_v1.pth"))

    def test_blank_override_falls_back_to_default(self):
        a = app.derive_paths("/tmp/p", "")
        b = app.derive_paths("/tmp/p", "   ")
        c = app.derive_paths("/tmp/p")
        assert a["rf_model"] == b["rf_model"] == c["rf_model"]


class TestMakeOutputPath:
    def test_contains_method_and_timestamp_pattern(self):
        out = app.make_output_path("/tmp/proj", "rf")
        assert out.startswith(str(Path("/tmp/proj/output/detections_rf_")))
        assert out.endswith(".kml")


class TestListOutputKmls:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert app.list_output_kmls(str(tmp_path / "nope")) == []

    def test_newest_first(self, tmp_path):
        import os
        import time
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        old = out_dir / "detections_rf_old.kml"
        new = out_dir / "detections_rf_new.kml"
        old.write_text("<kml/>")
        time.sleep(0.01)
        new.write_text("<kml/>")
        result = app.list_output_kmls(str(tmp_path))
        assert result[0] == str(new)
        assert result[1] == str(old)

    def test_ignores_non_kml_files(self, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "notes.txt").write_text("hi")
        assert app.list_output_kmls(str(tmp_path)) == []


class TestListModelFiles:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert app.list_model_files(str(tmp_path / "nope"), ".pkl") == []

    def test_filters_by_suffix(self, tmp_path):
        (tmp_path / "a.pkl").write_text("x")
        (tmp_path / "b.pth").write_text("x")
        (tmp_path / "c.pkl").write_text("x")
        result = app.list_model_files(str(tmp_path), ".pkl")
        assert len(result) == 2
        assert all(f.endswith(".pkl") for f in result)


class TestReadModelSidecar:
    def test_missing_sidecar_returns_none(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        model_path.write_text("x")
        assert app.read_model_sidecar(str(model_path)) is None

    def test_reads_valid_sidecar(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        model_path.write_text("x")
        sidecar = tmp_path / "model.json"
        sidecar.write_text(json.dumps({"created": "2026-01-01", "n_chips": 42}))
        meta = app.read_model_sidecar(str(model_path))
        assert meta["created"] == "2026-01-01"
        assert meta["n_chips"] == 42

    def test_malformed_sidecar_returns_none(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        model_path.write_text("x")
        (tmp_path / "model.json").write_text("{not json")
        assert app.read_model_sidecar(str(model_path)) is None

    def test_blank_path_returns_none(self):
        assert app.read_model_sidecar("") is None


class TestBuildStatusLine:
    def test_no_model_no_sidecar(self, tmp_path):
        line = app.build_status_line(str(tmp_path / "model.pkl"))
        assert "not trained yet" in line

    def test_model_without_sidecar(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        model_path.write_text("x")
        line = app.build_status_line(str(model_path))
        assert "no metadata sidecar" in line

    def test_full_metadata_renders_all_parts(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        model_path.write_text("x")
        (tmp_path / "model.json").write_text(json.dumps({
            "created": "2026-01-01", "n_chips": 100,
            "cv": {"pr_auc": 0.876}, "recommended_threshold": 0.33,
        }))
        line = app.build_status_line(str(model_path))
        assert "2026-01-01" in line
        assert "100 chips" in line
        assert "0.88" in line or "0.876" in line
        assert "0.33" in line

    def test_stale_model_warns_when_labels_newer(self, tmp_path):
        import time
        model_path = tmp_path / "model.pkl"
        model_path.write_text("x")
        (tmp_path / "model.json").write_text(json.dumps({"created": "2026-01-01"}))
        time.sleep(0.01)
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "new_labels.kml").write_text("<kml/>")
        line = app.build_status_line(str(model_path), str(labels_dir))
        assert "changed since" in line

    def test_fresh_model_no_warning(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "labels.kml").write_text("<kml/>")
        import time
        time.sleep(0.01)
        model_path = tmp_path / "model.pkl"
        model_path.write_text("x")
        (tmp_path / "model.json").write_text(json.dumps({"created": "2026-01-01"}))
        line = app.build_status_line(str(model_path), str(labels_dir))
        assert "changed since" not in line


class TestMigrateSettings:
    def test_empty_settings_pass_through(self):
        assert app.migrate_settings({}) == {}

    def test_already_migrated_passes_through_unchanged(self):
        settings = {"project_dir": "data", "imagery_dir": "data/imagery"}
        assert app.migrate_settings(settings) == settings

    def test_unrecognised_settings_pass_through(self):
        settings = {"some_future_key": 1}
        assert app.migrate_settings(settings) == settings

    def test_legacy_settings_migrated(self):
        old = {
            "rf_imagery": "data/imagery",
            "rf_labels": "data/labels",
            "rf_hydro": "data/hydrography",
            "det_rf_model": "data/models/model.pkl",
            "det_threshold": 0.6,
            "ev_radius": 400,
            "ev_per_class": True,
            "cnn_epochs": 20,
            "cnn_lr": 0.01,
        }
        new = app.migrate_settings(old)
        assert new["project_dir"] == "data"
        assert new["imagery_dir"] == "data/imagery"
        assert new["labels_dir"] == "data/labels"
        assert new["hydro_dir"] == "data/hydrography"
        # det_rf_model matches the derived default, so no override needed.
        assert new["rf_model_override"] == ""
        assert new["det_threshold"] == 0.6
        assert new["ev_radius"] == 400
        assert new["ev_per_class"] is True
        assert new["cnn_epochs"] == 20
        assert new["cnn_lr"] == 0.01

    def test_legacy_settings_with_custom_model_path_kept_as_override(self):
        old = {
            "rf_imagery": "data/imagery",
            "rf_labels": "data/labels",
            "det_rf_model": "data/models/custom_v2.pkl",
        }
        new = app.migrate_settings(old)
        assert new["rf_model_override"] == "data/models/custom_v2.pkl"

    def test_real_world_settings_json_migrates_without_error(self):
        # Mirrors the shape of the actual pre-G1.1 data/settings.json on disk.
        old = {
            "rf_imagery": "data/imagery", "rf_labels": "data/labels",
            "rf_model": "data/models.pkl", "rf_hydro": "data/hydrography",
            "rf_chips": "", "rf_flood_samples": 100,
            "cnn_imagery": "", "cnn_labels": "", "cnn_model": "",
            "cnn_norm_stats": "", "cnn_hydro": "", "cnn_epochs": 30, "cnn_lr": 0.001,
            "det_imagery": "data/imagery/R4431G.jp2",
            "det_output": "data/output/detections_R4431G.kml",
            "det_rf_model": "data/models/model.pkl", "det_cnn_model": "",
            "det_norm_stats": "", "det_hydro": "data/hydrography",
            "det_method": "rf", "det_threshold": 0.5,
            "ev_manifest": "", "ev_rf_model": "", "ev_radius": 500, "ev_per_class": False,
            "cmp_manifest": "", "cmp_rf_model": "", "cmp_cnn_model": "",
            "cmp_norm_stats": "", "cmp_test_frac": 0.2,
            "diag_imagery": "", "diag_rf_model": "",
            "ov_imagery": "data/imagery", "ov_labels": "data/labels",
            "ov_models_dir": "data/models", "ov_chips": "data/chips",
            "map_kml": "data/output/detections.kml", "map_labels": "data/labels",
            "map_hydro": "data/hydrography",
        }
        new = app.migrate_settings(old)
        assert new["project_dir"] == "data"
        assert new["imagery_dir"] == "data/imagery"
        assert new["labels_dir"] == "data/labels"
        assert new["hydro_dir"] == "data/hydrography"
        assert new["train_flood_samples"] == 100
        assert new["map_kml"] == "data/output/detections.kml"


class TestSettingsRoundTrip:
    def test_save_and_load_via_env_var(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        monkeypatch.setenv("CASTOR_SETTINGS", str(settings_file))
        app._save_settings({"project_dir": "data", "imagery_dir": "data/imagery"})
        loaded = app._load_settings()
        assert loaded["project_dir"] == "data"
        assert loaded["imagery_dir"] == "data/imagery"

    def test_handle_autosave_writes_all_keys(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        monkeypatch.setenv("CASTOR_SETTINGS", str(settings_file))
        values = list(range(len(app._SETTINGS_KEYS)))
        status = app.handle_autosave(*values)
        assert "saved" in status.lower()
        with open(settings_file) as f:
            saved = json.load(f)
        assert saved[app._SETTINGS_KEYS[0]] == 0
        assert saved[app._SETTINGS_KEYS[-1]] == len(app._SETTINGS_KEYS) - 1


class TestAuditTableAndLayer:
    def test_handle_audit_table_missing_oof_returns_empty(self, tmp_path):
        assert app.handle_audit_table(str(tmp_path / "noproj"), "") == []

    def test_handle_audit_table_reads_oof_csv(self, tmp_path):
        import csv
        chips_dir = tmp_path / "chips"
        chips_dir.mkdir()
        oof_path = chips_dir / "oof.csv"
        with open(oof_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "feature_type", "label", "x", "y", "cluster", "fold", "prob"])
            # EPSG:3067 point roughly in Finland.
            writer.writerow(["a.npy", "dead_forest", 1, 450000, 7200000, 0, 0, 0.05])
        table = app.handle_audit_table(str(tmp_path), "", low=0.2, high=0.8)
        assert len(table) == 1
        row = table[0]
        assert row[0] == "dead_forest"
        assert row[1] == "positive"
        # lat around 64-65N, lon around 24-26E for this EPSG:3067 point.
        assert 55 < row[2] < 70
        assert 15 < row[3] < 35
