"""Random Forest classifier for beaver signature detection."""

import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from spectral import extract_features

# Single source of truth for RF hyperparameters — used by both train() and the
# spatial cross-validation in models/evaluate.py so folds and the final saved
# model are always trained with identical settings.
_DEFAULT_RF_PARAMS: dict = dict(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-2,
)


def make_classifier(**overrides) -> RandomForestClassifier:
    """Build a RandomForestClassifier with the project's default hyperparameters,
    optionally overridden (e.g. random_state=fold_seed)."""
    params = {**_DEFAULT_RF_PARAMS, **overrides}
    return RandomForestClassifier(**params)


def build_feature_matrix(manifest_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load chips listed in a manifest CSV and return (X, y) arrays."""
    X, y = [], []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            chip = np.load(row["path"])
            X.append(extract_features(chip))
            y.append(int(row["label"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train(
    manifest_path: str,
    model_path: str,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -2,
) -> RandomForestClassifier:
    """Train a Random Forest on chips in manifest_path and save to model_path."""
    X, y = build_feature_matrix(manifest_path)
    clf = make_classifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    clf.fit(X, y)
    save_model(clf, model_path)
    return clf


def predict(clf: RandomForestClassifier, chip: np.ndarray) -> tuple[int, float]:
    """Return (label, confidence) for a single chip."""
    feats = extract_features(chip).reshape(1, -1)
    label = int(clf.predict(feats)[0])
    confidence = float(clf.predict_proba(feats)[0][label])
    return label, confidence


def save_model(clf: RandomForestClassifier, model_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)


def load_model(model_path: str) -> RandomForestClassifier:
    with open(model_path, "rb") as f:
        return pickle.load(f)
