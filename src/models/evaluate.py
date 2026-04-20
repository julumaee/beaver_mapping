"""Evaluate RF vs CNN on a held-out fraction of the training manifest."""

import csv
import json

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

    y_true = [int(r["label"]) for r in test_rows]
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
