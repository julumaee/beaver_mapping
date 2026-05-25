"""Evaluate RF vs CNN on a held-out fraction of the training manifest."""

import csv
import json
from collections import defaultdict

import numpy as np


def evaluate_models(
    manifest_path: str,
    rf_model_path: str,
    cnn_model_path: str,
    norm_stats_path: str,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """
    Hold out test_fraction of manifest rows, run both models, print a comparison
    table, and return {'rf': metrics, 'cnn': metrics, 'n_test': int}.

    Note: uses a random split of the training manifest — results are indicative
    rather than truly out-of-sample. Supply a separate test manifest for rigour.
    """
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(rows))
    n_test = max(1, int(len(rows) * test_fraction))
    test_rows = [rows[i] for i in indices[:n_test]]

    y_true = [min(int(r["label"]), 1) for r in test_rows]
    chips = [np.load(r["path"]) for r in test_rows]

    # RF
    from models.random_forest import load_model as load_rf, predict as rf_predict
    rf_clf = load_rf(rf_model_path)
    rf_preds = [rf_predict(rf_clf, c)[0] for c in chips]

    # CNN
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    from models.cnn_handler import load_cnn, predict_cnn_batch, BATCH_SIZE
    cnn_model = load_cnn(cnn_model_path)
    cnn_preds: list[int] = []
    for i in range(0, len(chips), BATCH_SIZE):
        batch_results = predict_cnn_batch(cnn_model, chips[i : i + BATCH_SIZE], norm_stats)
        cnn_preds.extend(label for label, _ in batch_results)

    rf_metrics = _metrics(y_true, rf_preds)
    cnn_metrics = _metrics(y_true, cnn_preds)
    _print_table(rf_metrics, cnn_metrics, n_test)

    return {"rf": rf_metrics, "cnn": cnn_metrics, "n_test": n_test}


def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    accuracy  = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _print_table(rf: dict, cnn: dict, n_test: int) -> None:
    print(f"\n{'Model':<8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7}")
    print("-" * 46)
    for name, m in [("RF", rf), ("CNN", cnn)]:
        print(f"{name:<8} {m['accuracy']:>9.3f} {m['precision']:>10.3f} "
              f"{m['recall']:>8.3f} {m['f1']:>7.3f}")
    print(f"\n(evaluated on {n_test} held-out chips)")


# ---------------------------------------------------------------------------
# RF-only spatial cross-validation
# ---------------------------------------------------------------------------

def evaluate_rf_spatial(
    manifest_path: str,
    rf_model_path: str,
    cluster_radius: float = 500.0,
    random_seed: int = 42,
) -> dict:
    """
    Evaluate the RF classifier with spatial leave-one-cluster-out CV.

    Label points within cluster_radius metres of each other are grouped
    into the same cluster. Each fold holds out one cluster's chips as the
    test set and trains on the rest. This avoids the spatial autocorrelation
    leak that a random split introduces (nearby chips share landscape context).

    Returns averaged metrics across folds plus per-fold details.
    """
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    clusters = _spatial_clusters(rows, cluster_radius)
    n_clusters = max(clusters) + 1
    print(f"Spatial CV: {len(rows)} chips in {n_clusters} clusters "
          f"(radius={cluster_radius:.0f} m)")

    fold_metrics: list[dict] = []

    for held_out in range(n_clusters):
        test_rows  = [r for r, c in zip(rows, clusters) if c == held_out]
        train_rows = [r for r, c in zip(rows, clusters) if c != held_out]

        if not test_rows or not train_rows:
            continue

        # Build train feature matrix and fit a fresh RF
        from models.random_forest import load_model as load_rf
        import pickle
        from sklearn.ensemble import RandomForestClassifier
        from spectral import extract_features

        X_train = np.array([extract_features(np.load(r["path"])) for r in train_rows], dtype=np.float32)
        y_train = np.array([min(int(r["label"]), 1) for r in train_rows], dtype=np.int32)

        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                     random_state=random_seed, n_jobs=-2)
        clf.fit(X_train, y_train)

        X_test = np.array([extract_features(np.load(r["path"])) for r in test_rows], dtype=np.float32)
        y_true = [min(int(r["label"]), 1) for r in test_rows]
        y_pred = clf.predict(X_test).tolist()

        fold_metrics.append(_metrics(y_true, y_pred))

    avg = _average_metrics(fold_metrics)
    _print_spatial_cv_table(avg, fold_metrics)
    return {"avg": avg, "folds": fold_metrics, "n_clusters": n_clusters}


def _spatial_clusters(rows: list[dict], radius: float) -> list[int]:
    """
    Greedy single-linkage clustering of label points by Euclidean distance.
    Returns a cluster-id list parallel to rows.
    """
    coords = [(float(r["x"]), float(r["y"])) for r in rows]
    cluster_ids = list(range(len(coords)))

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            if (dx * dx + dy * dy) ** 0.5 < radius:
                # Merge j's cluster into i's cluster
                old_id = cluster_ids[j]
                new_id = cluster_ids[i]
                if old_id != new_id:
                    cluster_ids = [new_id if c == old_id else c for c in cluster_ids]

    # Re-index cluster ids to 0..N-1
    unique = {c: idx for idx, c in enumerate(sorted(set(cluster_ids)))}
    return [unique[c] for c in cluster_ids]


def _average_metrics(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], float)]
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


def _print_spatial_cv_table(avg: dict, folds: list[dict]) -> None:
    print(f"\n{'Metric':<12} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 40)
    for key in ("accuracy", "precision", "recall", "f1"):
        vals = [m[key] for m in folds]
        print(f"{key:<12} {avg[key]:>8.3f} {min(vals):>8.3f} {max(vals):>8.3f}")
    print(f"\n(spatial LOCO-CV over {len(folds)} folds)")
