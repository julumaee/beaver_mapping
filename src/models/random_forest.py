"""Random Forest (and alternative tree-ensemble) classifiers for beaver
signature detection.

make_classifier() is the single source of truth for classifier construction —
used by train(), by the spatial cross-validation in models/evaluate.py, and
by the `tune` CLI command — so folds, tuning candidates, and the final saved
model are always built the same way for a given config.
"""

import csv
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from spectral import extract_features

# Single source of truth for RF hyperparameters — used by both train() and the
# spatial cross-validation in models/evaluate.py so folds and the final saved
# model are always trained with identical settings.
#
# n_estimators=300 / min_samples_leaf=3 replaced the original 100/1 default
# after `tune` (R3.1/R3.2) on chips_new (1745 chips) showed a reproducible
# improvement under identical pooled spatial CV: PR-AUC 0.908 -> 0.912,
# ROC-AUC 0.931 -> 0.932, hard-negative specificity 0.889 -> 0.909, with
# positive recall unchanged at 0.801 (recommended-threshold, point-level).
_DEFAULT_RF_PARAMS: dict = dict(
    n_estimators=300,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-2,
)

_DEFAULT_EXTRA_TREES_PARAMS: dict = dict(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-2,
)

_DEFAULT_HGB_PARAMS: dict = dict(
    class_weight="balanced",
    random_state=42,
)


class _GroupAwareCV:
    """Minimal cross-validation splitter for CalibratedClassifierCV that
    performs a StratifiedGroupKFold split using `groups` fixed at
    construction time.

    CalibratedClassifierCV.fit(X, y) calls `cv.split(X, y)` without forwarding
    a `groups` argument unless sklearn's metadata routing is explicitly
    enabled project-wide, which is more machinery than this needs. Since the
    `groups` array only needs to align row-for-row with the X the caller is
    about to fit on (guaranteed here — it's the same X passed from the outer
    CV fold), capturing it at construction time and ignoring the `groups`
    argument in `split()` gets the same result far more simply.
    """

    def __init__(self, groups, n_splits: int = 3, random_state: int = 42) -> None:
        self.groups = np.asarray(groups)
        self.random_state = random_state
        n_groups = len(set(self.groups.tolist()))
        self.n_splits = max(2, min(n_splits, n_groups))

    def split(self, X, y=None, groups=None):
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        yield from splitter.split(X, y, groups=self.groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def make_classifier(config: dict | None = None, groups=None, **overrides):
    """Build a classifier.

    config=None (default) preserves the original behaviour: a
    RandomForestClassifier built from _DEFAULT_RF_PARAMS with **overrides
    applied on top (e.g. random_state=fold_seed) — this is what every
    pre-existing call site relies on.

    config, when given, is a dict {"type": "rf" | "extra_trees" | "hgb" |
    "rf_calibrated", "params": {...}}. config["params"] is merged with
    **overrides (overrides win) on top of that type's defaults.

    groups, if given, is only used by type "rf_calibrated": it grounds the
    inner calibration cv in the same spatial clusters as the outer CV fold
    (see _GroupAwareCV), so the calibration split doesn't leak nearby chips
    across train/test the way a naive StratifiedKFold would. Ignored by
    every other type.
    """
    if config is None:
        params = {**_DEFAULT_RF_PARAMS, **overrides}
        return RandomForestClassifier(**params)

    clf_type = config.get("type", "rf")
    cfg_params = {**config.get("params", {}), **overrides}

    if clf_type == "rf":
        params = {**_DEFAULT_RF_PARAMS, **cfg_params}
        return RandomForestClassifier(**params)

    if clf_type == "extra_trees":
        params = {**_DEFAULT_EXTRA_TREES_PARAMS, **cfg_params}
        return ExtraTreesClassifier(**params)

    if clf_type == "hgb":
        params = {**_DEFAULT_HGB_PARAMS, **cfg_params}
        return HistGradientBoostingClassifier(**params)

    if clf_type == "rf_calibrated":
        from sklearn.calibration import CalibratedClassifierCV

        cfg_params = dict(cfg_params)  # don't mutate caller's dict
        method = cfg_params.pop("method", "isotonic")
        calibration_splits = int(cfg_params.pop("calibration_splits", 3))
        base_params = {**_DEFAULT_RF_PARAMS, **cfg_params}
        base = RandomForestClassifier(**base_params)

        if groups is not None:
            cv = _GroupAwareCV(
                groups, n_splits=calibration_splits,
                random_state=base_params.get("random_state", 42),
            )
        else:
            cv = calibration_splits
        return CalibratedClassifierCV(
            estimator=base, method=method, cv=cv, n_jobs=base_params.get("n_jobs", -2)
        )

    raise ValueError(f"Unknown classifier type: {clf_type!r}")


# ---------------------------------------------------------------------------
# Feature matrix construction, with optional on-disk caching
# ---------------------------------------------------------------------------

def feature_cache_paths(cache_base: Path) -> tuple[Path, Path]:
    return cache_base / "features.npy", cache_base / "features.key.json"


def load_or_compute_features(
    rows: list[dict],
    manifest_path: str,
    cache_base: Path,
    use_cache: bool = True,
) -> np.ndarray:
    """Compute the (N, feature_len) feature matrix, or reuse a cached one keyed
    by the manifest's mtime/size and row count. Feature extraction (GLCM in
    particular) dominates runtime, so this cache matters a lot when training
    and evaluating from the same manifest in one run (see cli.cmd_train).

    Shared by models.evaluate (spatial CV / tuning) and train() below so a
    `castor train` run with CV enabled computes features exactly once.
    """
    feat_path, key_path = feature_cache_paths(Path(cache_base))
    stat = Path(manifest_path).stat()
    key = {"manifest_mtime": stat.st_mtime, "manifest_size": stat.st_size, "n_rows": len(rows)}

    if use_cache and feat_path.exists() and key_path.exists():
        try:
            with open(key_path) as f:
                cached_key = json.load(f)
            if (cached_key.get("manifest_mtime") == key["manifest_mtime"]
                    and cached_key.get("manifest_size") == key["manifest_size"]
                    and cached_key.get("n_rows") == key["n_rows"]):
                X = np.load(feat_path)
                if X.shape[0] == len(rows) and X.shape[1] == cached_key.get("feature_len"):
                    print(f"Using cached features: {feat_path} ({X.shape})")
                    return X
        except Exception:
            pass  # fall through to recompute

    print("Computing features ..." if use_cache else "Computing features (--no-cache) ...")
    X = np.array([extract_features(np.load(r["path"])) for r in rows], dtype=np.float32)
    print(f"  Done. Feature matrix: {X.shape}")

    try:
        Path(cache_base).mkdir(parents=True, exist_ok=True)
        np.save(feat_path, X)
        key["feature_len"] = int(X.shape[1])
        with open(key_path, "w") as f:
            json.dump(key, f)
    except Exception as exc:
        print(f"  (could not write feature cache: {exc})")

    return X


def build_feature_matrix(manifest_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load chips listed in a manifest CSV and return (X, y) arrays. No
    caching — for that, use load_or_compute_features directly (train() does)."""
    X, y = [], []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            chip = np.load(row["path"])
            X.append(extract_features(chip))
            y.append(int(row["label"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Train / save / load
# ---------------------------------------------------------------------------

def train(
    manifest_path: str,
    model_path: str,
    n_estimators: int = 300,
    random_state: int = 42,
    n_jobs: int = -2,
    classifier_config: dict | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
    cv_results: dict | None = None,
):
    """Train a classifier on chips in manifest_path and save it to model_path,
    plus a JSON metadata sidecar at model_path with its suffix replaced by
    .json (e.g. model.pkl -> model.json — see write_model_metadata).

    classifier_config, if given, is a {"type", "params"} dict as accepted by
    make_classifier — typically the "best" block from a `tune` output file
    (cli.cmd_train reads --classifier-config and passes it through). Without
    it, a plain RandomForestClassifier is built from n_estimators/random_state
    /n_jobs, matching the original default behaviour.

    cache_dir/use_cache let the caller reuse a feature cache already computed
    by an earlier CV run on the same manifest (cli.cmd_train does this so
    features aren't extracted twice in one `train` invocation).

    cv_results, if given, is embedded in the metadata sidecar's "cv" field
    (and its recommended_threshold used as the sidecar's top-level
    recommended_threshold) — cli.cmd_train supplies the spatial CV result it
    ran right after training.
    """
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    manifest_dir = Path(manifest_path).resolve().parent
    cache_base = Path(cache_dir).resolve() if cache_dir else manifest_dir
    X = load_or_compute_features(rows, manifest_path, cache_base, use_cache)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)

    if classifier_config is not None:
        clf = make_classifier(config=classifier_config, random_state=random_state)
    else:
        clf = make_classifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    clf.fit(X, y)
    save_model(clf, model_path)
    write_model_metadata(clf, model_path, rows, feature_length=X.shape[1], cv_results=cv_results)
    return clf


def predict(clf, chip: np.ndarray) -> tuple[int, float]:
    """Return (label, confidence) for a single chip."""
    feats = extract_features(chip).reshape(1, -1)
    label = int(clf.predict(feats)[0])
    confidence = float(clf.predict_proba(feats)[0][label])
    return label, confidence


def save_model(clf, model_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def check_feature_length(clf) -> None:
    """Raise ValueError with a clear message if clf was trained with a
    different feature-vector length than spectral.extract_features currently
    produces — e.g. the feature set grew/shrank since this model was trained.
    """
    n_features = getattr(clf, "n_features_in_", None)
    if n_features is None:
        return  # nothing to check (unfitted or an estimator that doesn't expose this)

    current_len = int(extract_features(np.zeros((3, 64, 64), dtype=np.uint8)).shape[0])
    if int(n_features) != current_len:
        raise ValueError(
            f"Model was trained with {n_features} features but the current "
            f"feature extractor produces {current_len} — "
            "model was trained with an older feature set — retrain."
        )


# ---------------------------------------------------------------------------
# R3.5 — model metadata sidecar
# ---------------------------------------------------------------------------

def _sanitize_json(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    return str(value)


def _classifier_summary(clf) -> dict:
    try:
        params = _sanitize_json(clf.get_params())
    except Exception:
        params = {}
    return {"type": type(clf).__name__, "params": params}


def write_model_metadata(
    clf,
    model_path: str,
    rows: list[dict],
    feature_length: int,
    cv_results: dict | None = None,
) -> dict:
    """Write the JSON metadata sidecar for a trained model. See module
    docstring / CLAUDE.md task R3.5 for the exact schema other agents rely on.
    """
    chips_by_type = dict(Counter(r["feature_type"] for r in rows))
    label_map: dict[str, int] = {}
    for r in rows:
        label_map[r["feature_type"]] = int(r["label"])

    recommended_threshold = None
    cv_summary = None
    if cv_results is not None:
        recommended_threshold = cv_results.get("recommended_threshold")
        cv_summary = {
            "roc_auc": cv_results.get("roc_auc"),
            "pr_auc": cv_results.get("pr_auc"),
            "recommended_threshold": cv_results.get("recommended_threshold"),
            "high_recall_threshold": cv_results.get("high_recall_threshold"),
            "metrics_at_recommended": cv_results.get("metrics_at_recommended"),
            "point_metrics_at_recommended": cv_results.get("point_metrics_at_recommended"),
            "per_type": cv_results.get("per_type"),
        }

    meta = {
        "format_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "classifier": _classifier_summary(clf),
        "feature_length": int(feature_length),
        "label_map": label_map,
        "chips_by_type": chips_by_type,
        "n_chips": len(rows),
        "recommended_threshold": recommended_threshold,
        "cv": cv_summary,
    }

    sidecar_path = Path(model_path).with_suffix(".json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta


def load_model_metadata(model_path: str) -> dict | None:
    """Load the JSON metadata sidecar for model_path, or None if it doesn't
    exist / can't be parsed (e.g. a model saved before R3.5)."""
    sidecar_path = Path(model_path).with_suffix(".json")
    if not sidecar_path.exists():
        return None
    try:
        with open(sidecar_path) as f:
            return json.load(f)
    except Exception:
        return None
