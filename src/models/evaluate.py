"""RF evaluation: pooled out-of-fold spatial cross-validation, and an
in-sample-by-necessity RF vs CNN comparison.

Spatial CV (evaluate_rf_spatial) is the primary tool for judging RF quality:
label points are grouped into spatial clusters, split into folds with
StratifiedGroupKFold (so no cluster's chips appear in both train and test for
any fold), and every chip's out-of-fold probability is pooled before any
metric is computed. This avoids two failure modes of naive per-fold
leave-one-cluster-out averaging: (1) folds with only one class produce
degenerate precision/recall (0 or undefined) that corrupt a simple average,
and (2) small folds are noisy on their own but fine once pooled.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _spatial_clusters(rows: list[dict], radius: float) -> list[int]:
    """
    Single-linkage clustering of label points by Euclidean distance, using a
    radius-neighbours graph + connected components instead of an O(n^2) pairwise
    scan with list rebuilds. Chips sharing the same (x, y) (e.g. augmented
    copies of one label point) always land in the same cluster.

    Returns a cluster-id list parallel to rows.
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import radius_neighbors_graph

    coords = [(float(r["x"]), float(r["y"])) for r in rows]
    uniq_coords = sorted(set(coords))
    n_uniq = len(uniq_coords)

    if n_uniq == 1:
        point_cluster = np.zeros(1, dtype=int)
    else:
        pts = np.array(uniq_coords)
        graph = radius_neighbors_graph(pts, radius=radius, mode="connectivity", include_self=True)
        # Symmetrize: radius_neighbors_graph is not guaranteed symmetric at the boundary.
        graph = graph.maximum(graph.T)
        _, point_cluster = connected_components(graph, directed=False)

    cluster_of_coord = {uniq_coords[i]: int(point_cluster[i]) for i in range(n_uniq)}
    return [cluster_of_coord[c] for c in coords]


# Feature-matrix caching lives in models.random_forest (train() needs it too,
# to avoid recomputing features that cli.cmd_train's pre-training CV already
# cached) — re-exported here under its old name so this module's existing
# call sites are unchanged.
from models.random_forest import load_or_compute_features as _load_or_compute_features


def _recommended_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.8,
) -> tuple[float, float | None, float | None]:
    """Return (max-F1 threshold, high-recall threshold, precision at that
    threshold). The high-recall threshold is the one giving the best
    precision among thresholds with recall >= min_recall, or None if no
    threshold achieves that recall."""
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, None, None

    # precision_recall_curve appends a final (precision=1, recall=0) point with
    # no corresponding threshold — drop it so indices line up with `thresholds`.
    p, r = precisions[:-1], recalls[:-1]
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    best_idx = int(np.argmax(f1))
    best_threshold = float(thresholds[best_idx])

    hr_threshold = hr_precision = None
    mask = r >= min_recall
    if mask.any():
        candidates = np.where(mask)[0]
        best_hr = candidates[np.argmax(p[candidates])]
        hr_threshold = float(thresholds[best_hr])
        hr_precision = float(p[best_hr])

    return best_threshold, hr_threshold, hr_precision


# ---------------------------------------------------------------------------
# R0.1 — pooled out-of-fold spatial cross-validation
#
# The fold-fitting loop (_run_cv_folds) and the metrics computation
# (_summarize_cv) are factored out so `tune` (below) can run the identical CV
# procedure for several classifier candidates against one shared spatial
# clustering and one shared cached feature matrix, instead of duplicating
# this logic.
# ---------------------------------------------------------------------------

def _run_cv_folds(
    X_all: np.ndarray,
    y_all: np.ndarray,
    clusters_arr: np.ndarray,
    classifier_factory,
    n_splits: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Core pooled out-of-fold spatial CV loop. Splits by StratifiedGroupKFold
    over clusters_arr (capped at the number of distinct clusters), fits a
    fresh classifier per fold via
    classifier_factory(random_state=random_seed, groups=<fold's train-set
    clusters>), and predicts probabilities for the held-out chips.

    Returns (oof_prob, fold_of, n_splits_eff) — oof_prob and fold_of are
    parallel to X_all/y_all (NaN / -1 for any chip never held out, which
    shouldn't happen with StratifiedGroupKFold but is guarded against by the
    caller).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_clusters = int(clusters_arr.max()) + 1
    n_splits_eff = min(n_splits, n_clusters)
    if n_splits_eff < n_splits:
        print(f"  (capping n_splits to {n_splits_eff}; only {n_clusters} spatial clusters available)")

    splitter = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_seed)

    oof_prob = np.full(len(y_all), np.nan, dtype=np.float64)
    fold_of  = np.full(len(y_all), -1, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_all, y_all, groups=clusters_arr)):
        print(f"  Fold {fold_idx + 1}/{n_splits_eff}: "
              f"train={len(train_idx)}, test={len(test_idx)} chips", end=" ... ")
        clf = classifier_factory(random_state=random_seed, groups=clusters_arr[train_idx])
        clf.fit(X_all[train_idx], y_all[train_idx])
        classes = list(clf.classes_)
        if 1 in classes:
            proba = clf.predict_proba(X_all[test_idx])[:, classes.index(1)]
        else:
            proba = np.zeros(len(test_idx))  # degenerate: fold's train set had no positives
        oof_prob[test_idx] = proba
        fold_of[test_idx] = fold_idx
        print("done")

    return oof_prob, fold_of, n_splits_eff


def _summarize_cv(
    rows: list[dict],
    y_all: np.ndarray,
    oof_prob: np.ndarray,
    fold_of: np.ndarray,
    clusters_arr: np.ndarray,
    cluster_radius: float,
    n_clusters: int,
    n_splits_eff: int,
    oof_out: Path | None,
    per_class: bool = True,
    write_oof: bool = True,
    print_summary: bool = True,
) -> dict:
    """Turn pooled out-of-fold probabilities into the full metrics dict
    returned by evaluate_rf_spatial / used per-candidate by tune."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    valid_mask = ~np.isnan(oof_prob)
    if not valid_mask.all():
        print(f"  WARNING: {int((~valid_mask).sum())} chips were never held out; excluded from pooled metrics.")

    if write_oof and oof_out is not None:
        _write_oof_csv(oof_out, rows, y_all, oof_prob, clusters_arr, fold_of, valid_mask)

    y_true = y_all[valid_mask]
    y_prob = oof_prob[valid_mask]
    two_classes = len(set(y_true.tolist())) > 1

    roc_auc = float(roc_auc_score(y_true, y_prob)) if two_classes else float("nan")
    pr_auc  = float(average_precision_score(y_true, y_prob)) if two_classes else float("nan")
    recommended_threshold, hr_threshold, hr_precision = (
        _recommended_thresholds(y_true, y_prob) if two_classes else (0.5, None, None)
    )

    metrics_05  = _metrics(y_true.tolist(), (y_prob >= 0.5).astype(int).tolist())
    metrics_rec = _metrics(y_true.tolist(), (y_prob >= recommended_threshold).astype(int).tolist())

    # Point-level: aggregate chips sharing (x, y, feature_type) by mean probability,
    # since augmented chips otherwise inflate chip-level counts for the same point.
    point_probs: dict[tuple, list[float]] = defaultdict(list)
    point_label: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        if not valid_mask[i]:
            continue
        key = (r["x"], r["y"], r["feature_type"])
        point_probs[key].append(float(oof_prob[i]))
        point_label[key] = int(y_all[i])

    point_keys = list(point_probs.keys())
    point_prob_arr = np.array([float(np.mean(point_probs[k])) for k in point_keys])
    point_true_arr = np.array([point_label[k] for k in point_keys])

    point_metrics_05  = _metrics(point_true_arr.tolist(), (point_prob_arr >= 0.5).astype(int).tolist())
    point_metrics_rec = _metrics(point_true_arr.tolist(),
                                  (point_prob_arr >= recommended_threshold).astype(int).tolist())

    per_type = _per_type_breakdown(rows, y_all, oof_prob, valid_mask, recommended_threshold)

    result = {
        "n_chips": int(valid_mask.sum()),
        "n_points": len(point_keys),
        "n_clusters": n_clusters,
        "n_splits": n_splits_eff,
        "cluster_radius": cluster_radius,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recommended_threshold": recommended_threshold,
        "high_recall_threshold": hr_threshold,
        "high_recall_precision": hr_precision,
        "metrics_at_0.5": metrics_05,
        "metrics_at_recommended": metrics_rec,
        "point_metrics_at_0.5": point_metrics_05,
        "point_metrics_at_recommended": point_metrics_rec,
        "confusion_matrix_at_recommended": {k: metrics_rec[k] for k in ("tp", "fp", "fn", "tn")},
        "per_type": per_type,
        "oof_csv": str(oof_out) if (write_oof and oof_out is not None) else None,
    }
    if print_summary:
        _print_summary(result, print_per_type=per_class)
    return result


def _prepare_cv_inputs(
    manifest_path: str,
    cluster_radius: float,
    cache_dir: str | None,
    use_cache: bool,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, int, Path]:
    """Shared setup for evaluate_rf_spatial and tune_rf: load the manifest,
    build spatial clusters, and compute/reuse the cached feature matrix.
    Returns (rows, X_all, y_all, clusters_arr, n_clusters, cache_base)."""
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in manifest {manifest_path}")

    manifest_dir = Path(manifest_path).resolve().parent
    cache_base = Path(cache_dir).resolve() if cache_dir else manifest_dir

    clusters = _spatial_clusters(rows, cluster_radius)
    clusters_arr = np.array(clusters, dtype=int)
    n_clusters = int(clusters_arr.max()) + 1
    print(f"{len(rows)} chips in {n_clusters} clusters (radius={cluster_radius:.0f} m)")
    if n_clusters < 2:
        raise ValueError(
            f"Need at least 2 spatial clusters for cross-validation (found {n_clusters}); "
            "reduce --cluster-radius or add more spatially separated labels."
        )

    X_all = _load_or_compute_features(rows, manifest_path, cache_base, use_cache)
    y_all = np.array([min(int(r["label"]), 1) for r in rows], dtype=np.int32)

    return rows, X_all, y_all, clusters_arr, n_clusters, cache_base


def evaluate_rf_spatial(
    manifest_path: str,
    rf_model_path: str | None = None,
    cluster_radius: float = 500.0,
    n_splits: int = 5,
    random_seed: int = 42,
    oof_path: str | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
    per_class: bool = True,
    classifier_factory=None,
) -> dict:
    """
    Evaluate a classifier with pooled out-of-fold spatial cross-validation.

    Label points within cluster_radius metres of each other are grouped into
    the same spatial cluster (so nearby chips never straddle train/test).
    Clusters are split into folds with StratifiedGroupKFold (n_splits, capped
    at the number of clusters) so each fold has a mix of both classes where
    possible. A fresh classifier is trained per fold and predicts
    probabilities for its held-out chips; those probabilities are pooled
    across all folds before any metric is computed, which avoids the
    degenerate 0/1 precision-recall folds produce when a fold's test set
    happens to contain only one class.

    classifier_factory, if given, is a callable(random_state=..., groups=...)
    -> unfitted estimator, used to build the classifier trained per fold —
    `tune` uses this to run the exact same CV procedure for several candidate
    classifier configs. Defaults to models.random_forest.make_classifier
    (config=None), i.e. the project's default RandomForestClassifier.

    rf_model_path is accepted for interface/logging continuity (it is not
    used to fit the CV folds).

    Returns a dict with ROC-AUC, PR-AUC, recommended/high-recall thresholds,
    chip- and point-level metrics at 0.5 and the recommended threshold, a
    per-feature-type breakdown, and the path to the written OOF CSV.
    """
    from models.random_forest import make_classifier

    if rf_model_path:
        print(f"Model (unused for CV fitting, kept for reference): {rf_model_path}")

    rows, X_all, y_all, clusters_arr, n_clusters, cache_base = _prepare_cv_inputs(
        manifest_path, cluster_radius, cache_dir, use_cache
    )
    oof_out = Path(oof_path).resolve() if oof_path else cache_base / "oof.csv"

    if classifier_factory is None:
        classifier_factory = lambda **kw: make_classifier(**kw)

    oof_prob, fold_of, n_splits_eff = _run_cv_folds(
        X_all, y_all, clusters_arr, classifier_factory, n_splits, random_seed
    )

    return _summarize_cv(
        rows, y_all, oof_prob, fold_of, clusters_arr, cluster_radius, n_clusters, n_splits_eff,
        oof_out, per_class=per_class, write_oof=True, print_summary=True,
    )


def _write_oof_csv(
    path: Path,
    rows: list[dict],
    y_all: np.ndarray,
    oof_prob: np.ndarray,
    clusters: np.ndarray,
    fold_of: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "feature_type", "label", "x", "y", "cluster", "fold", "prob"])
        for i, r in enumerate(rows):
            if not valid_mask[i]:
                continue
            writer.writerow([
                r["path"], r["feature_type"], int(y_all[i]), r["x"], r["y"],
                int(clusters[i]), int(fold_of[i]), f"{oof_prob[i]:.6f}",
            ])
    print(f"OOF predictions written to {path}")


def _per_type_breakdown(
    rows: list[dict],
    y_all: np.ndarray,
    oof_prob: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
) -> dict[str, dict]:
    """Per-feature-type breakdown from pooled OOF predictions. Never hardcodes
    type names — walks whatever feature_type values are actually present."""
    idx_by_type: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if valid_mask[i]:
            idx_by_type[r["feature_type"]].append(i)

    results: dict[str, dict] = {}
    for ftype, idx in idx_by_type.items():
        idx_arr = np.array(idx)
        y_t = y_all[idx_arr]
        probs = oof_prob[idx_arr]
        preds = (probs >= threshold).astype(int)
        pos_mask = y_t == 1
        neg_mask = y_t == 0
        n_points = len({(rows[i]["x"], rows[i]["y"]) for i in idx})

        entry: dict = {
            "n_chips": len(idx),
            "n_points": n_points,
            "mean_prob": float(probs.mean()),
        }
        if pos_mask.any():
            entry["recall"] = float(preds[pos_mask].mean())
        if neg_mask.any():
            entry["specificity"] = float(1.0 - preds[neg_mask].mean())
        results[ftype] = entry

    return results


def _print_summary(result: dict, print_per_type: bool = True) -> None:
    print(f"\n{'=' * 66}")
    print(f"Spatial CV summary  ({result['n_chips']} chips / {result['n_points']} points, "
          f"{result['n_clusters']} clusters, {result['n_splits']}-fold, "
          f"radius={result['cluster_radius']:.0f} m)")
    print("=" * 66)
    print(f"ROC-AUC: {result['roc_auc']:.3f}    PR-AUC (avg. precision): {result['pr_auc']:.3f}")
    print(f"Recommended threshold (max F1): {result['recommended_threshold']:.3f}")
    if result["high_recall_threshold"] is not None:
        print(f"High-recall threshold (recall>=0.8, best precision): "
              f"{result['high_recall_threshold']:.3f}  (precision={result['high_recall_precision']:.3f})")
    else:
        print("High-recall threshold (recall>=0.8): not achievable at any threshold")

    print(f"\n{'Level':<8} {'Threshold':>10} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7}")
    print("-" * 56)
    rows = [
        ("chip",  "0.5",         result["metrics_at_0.5"]),
        ("chip",  "recommended", result["metrics_at_recommended"]),
        ("point", "0.5",         result["point_metrics_at_0.5"]),
        ("point", "recommended", result["point_metrics_at_recommended"]),
    ]
    for level, thr, m in rows:
        print(f"{level:<8} {thr:>10} {m['accuracy']:>9.3f} {m['precision']:>10.3f} "
              f"{m['recall']:>8.3f} {m['f1']:>7.3f}")

    m = result["metrics_at_recommended"]
    print(f"\nConfusion matrix at recommended threshold (chip-level):")
    print(f"  tp={m['tp']}  fp={m['fp']}  fn={m['fn']}  tn={m['tn']}")

    if print_per_type:
        _print_per_type_table(result["per_type"])


def _print_per_type_table(per_type: dict[str, dict]) -> None:
    print(f"\n{'Feature type':<18} {'N chips':>8} {'N points':>9} {'Recall':>8} {'Specificity':>12} {'MeanProb':>9}")
    print("-" * 68)
    for ftype, e in sorted(per_type.items()):
        recall = f"{e['recall']:.3f}" if "recall" in e else "—"
        spec = f"{e['specificity']:.3f}" if "specificity" in e else "—"
        print(f"{ftype:<18} {e['n_chips']:>8} {e['n_points']:>9} {recall:>8} {spec:>12} {e['mean_prob']:>9.3f}")


def evaluate_rf_per_class(
    manifest_path: str,
    rf_model_path: str | None = None,
    cluster_radius: float = 500.0,
    **kwargs,
) -> dict:
    """Backwards-compatible wrapper: runs the full pooled spatial CV and
    returns just the per-feature-type breakdown (see evaluate_rf_spatial)."""
    result = evaluate_rf_spatial(
        manifest_path, rf_model_path=rf_model_path, cluster_radius=cluster_radius, **kwargs
    )
    return result["per_type"]


# ---------------------------------------------------------------------------
# R3.1/R3.2/R3.4 — tune: compare classifier candidates on identical CV
# ---------------------------------------------------------------------------

def default_tune_candidates() -> list[dict]:
    """The default `tune` grid — a small, deliberately cheap set of
    alternatives to the plain default RandomForestClassifier:

    - RF at higher min_samples_leaf / a smaller max_features fraction
      (more regularisation, in case the default RF is overfitting to
      near-duplicate chips within a cluster)
    - ExtraTreesClassifier (fully random split thresholds — usually more
      resistant to overfitting than RF at the cost of some bias)
    - HistGradientBoostingClassifier (a different bias/variance tradeoff,
      handles the mixed feature scales natively)
    - RF + isotonic probability calibration (same decision boundary, better
      calibrated probabilities — useful since detection thresholds are
      chosen directly off predict_proba)

    Each entry is {"name": str, "config": {"type", "params"}} as accepted by
    models.random_forest.make_classifier.
    """
    candidates: list[dict] = []

    for msl in (1, 3, 5, 10):
        for mf in ("sqrt", 0.3):
            candidates.append({
                "name": f"rf_leaf{msl}_mf{mf}",
                "config": {"type": "rf", "params": {
                    "n_estimators": 300, "min_samples_leaf": msl, "max_features": mf,
                }},
            })

    for msl in (1, 5):
        candidates.append({
            "name": f"extra_trees_leaf{msl}",
            "config": {"type": "extra_trees", "params": {
                "n_estimators": 300, "min_samples_leaf": msl,
            }},
        })

    for lr, mln in ((0.05, 31), (0.1, 31), (0.1, 63)):
        candidates.append({
            "name": f"hgb_lr{lr}_leaves{mln}",
            "config": {"type": "hgb", "params": {
                "learning_rate": lr, "max_leaf_nodes": mln,
            }},
        })

    candidates.append({
        "name": "rf_calibrated_isotonic",
        "config": {"type": "rf_calibrated", "params": {
            "n_estimators": 300, "method": "isotonic",
        }},
    })

    return candidates


def tune_rf(
    manifest_path: str,
    cache_dir: str | None = None,
    out_path: str | None = None,
    cluster_radius: float = 500.0,
    n_splits: int = 5,
    random_seed: int = 42,
    candidates: list[dict] | None = None,
) -> dict:
    """
    Evaluate a grid of classifier candidates with the same pooled
    out-of-fold spatial CV used by evaluate_rf_spatial. The spatial
    clustering and the feature matrix are computed once and shared across
    every candidate (via _prepare_cv_inputs) — only the fold-fitting cost
    (_run_cv_folds) is repeated per candidate.

    candidates overrides the default grid (default_tune_candidates()) — this
    is also the `tune` CLI command's test hook, letting tests run a couple of
    cheap candidates on tiny synthetic manifests instead of the full grid.

    Writes {"results": [ranked by PR-AUC], "best": <top result>} to out_path
    (default: <cache_base>/tuning.json) and returns the same dict (plus
    "out_path").
    """
    import time

    from models.random_forest import make_classifier

    rows, X_all, y_all, clusters_arr, n_clusters, cache_base = _prepare_cv_inputs(
        manifest_path, cluster_radius, cache_dir, use_cache=True
    )

    if candidates is None:
        candidates = default_tune_candidates()

    results: list[dict] = []
    for cand in candidates:
        name, config = cand["name"], cand["config"]
        print(f"\n--- {name} ---")
        t0 = time.time()

        def _factory(config=config, **kw):
            return make_classifier(config=config, **kw)

        oof_prob, fold_of, n_splits_eff = _run_cv_folds(
            X_all, y_all, clusters_arr, _factory, n_splits, random_seed,
        )
        fit_time = time.time() - t0

        summary = _summarize_cv(
            rows, y_all, oof_prob, fold_of, clusters_arr, cluster_radius,
            n_clusters, n_splits_eff, oof_out=None,
            per_class=True, write_oof=False, print_summary=False,
        )
        per_type = summary["per_type"]
        # "hard_negatives" is chips_new's naming; fall back to "negative" for
        # older/synthetic manifests that don't distinguish hard negatives.
        hard_neg = per_type.get("hard_negatives") or per_type.get("negative") or {}

        entry = {
            "name": name,
            "config": config,
            "roc_auc": summary["roc_auc"],
            "pr_auc": summary["pr_auc"],
            "point_f1_at_recommended": summary["point_metrics_at_recommended"]["f1"],
            "positive_recall_at_recommended": summary["metrics_at_recommended"]["recall"],
            "hard_negative_specificity": hard_neg.get("specificity"),
            "recommended_threshold": summary["recommended_threshold"],
            "fit_time_sec": round(fit_time, 2),
        }
        results.append(entry)
        spec_str = (f"{entry['hard_negative_specificity']:.3f}"
                    if entry["hard_negative_specificity"] is not None else "n/a")
        print(f"  pr_auc={entry['pr_auc']:.3f}  roc_auc={entry['roc_auc']:.3f}  "
              f"point_f1={entry['point_f1_at_recommended']:.3f}  "
              f"recall={entry['positive_recall_at_recommended']:.3f}  "
              f"hard_neg_spec={spec_str}  time={entry['fit_time_sec']:.1f}s")

    def _sort_key(r: dict) -> float:
        v = r["pr_auc"]
        return v if v == v else -1.0  # NaN (single-class pooled OOF) sorts last

    results_sorted = sorted(results, key=_sort_key, reverse=True)
    best = results_sorted[0] if results_sorted else None

    out_file = Path(out_path).resolve() if out_path else cache_base / "tuning.json"
    out = {
        "manifest": str(Path(manifest_path).resolve()),
        "cluster_radius": cluster_radius,
        "n_clusters": n_clusters,
        "n_candidates": len(results),
        "results": results_sorted,
        "best": best,
        "out_path": str(out_file),
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=str)

    _print_tune_table(results_sorted)
    if best:
        print(f"\nBest by PR-AUC: {best['name']}  (results written to {out_file})")

    return out


def _print_tune_table(results: list[dict]) -> None:
    print(f"\n{'=' * 100}")
    print(f"Tuning results ({len(results)} candidates, ranked by PR-AUC)")
    print("=" * 100)
    header = (f"{'Name':<26} {'PR-AUC':>8} {'ROC-AUC':>8} {'PointF1':>8} "
              f"{'Recall':>8} {'HardNegSpec':>12} {'Fit(s)':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        spec = (f"{r['hard_negative_specificity']:.3f}"
                if r["hard_negative_specificity"] is not None else "n/a")
        print(f"{r['name']:<26} {r['pr_auc']:>8.3f} {r['roc_auc']:>8.3f} "
              f"{r['point_f1_at_recommended']:>8.3f} {r['positive_recall_at_recommended']:>8.3f} "
              f"{spec:>12} {r['fit_time_sec']:>8.1f}")


# ---------------------------------------------------------------------------
# R0.3 — RF vs CNN comparison
# ---------------------------------------------------------------------------

def evaluate_models(
    manifest_path: str,
    rf_model_path: str,
    cnn_model_path: str,
    norm_stats_path: str,
    test_manifest_path: str | None = None,
    test_fraction: float = 0.2,
    cluster_radius: float = 500.0,
    random_seed: int = 42,
) -> dict:
    """
    Compare the saved RF and CNN models.

    If test_manifest_path is given (a manifest built from separate,
    held-out tiles), that is a genuine out-of-sample comparison. Without it,
    this falls back to holding out a fraction of manifest_path by spatial
    cluster (not random rows, to avoid augmented copies of one point
    straddling the split) — but since both saved models were already trained
    on every row in manifest_path, the result is still in-sample and only
    indicative. A prominent warning is printed in that case.
    """
    in_sample = test_manifest_path is None

    if test_manifest_path:
        with open(test_manifest_path) as f:
            test_rows = list(csv.DictReader(f))
        print(f"Evaluating on separate test manifest: {test_manifest_path} ({len(test_rows)} chips)")
    else:
        print("=" * 70)
        print("WARNING: IN-SAMPLE EVALUATION")
        print("These chips were used to train the saved RF/CNN models. The numbers")
        print("below are optimistic and do not reflect real-world performance.")
        print("Supply --test-manifest with chips from held-out tiles for a true")
        print("out-of-sample comparison.")
        print("=" * 70)

        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        clusters = _spatial_clusters(rows, cluster_radius)
        n_clusters = max(clusters) + 1
        rows_by_cluster: dict[int, list[dict]] = defaultdict(list)
        for r, c in zip(rows, clusters):
            rows_by_cluster[c].append(r)

        rng = np.random.default_rng(random_seed)
        cluster_order = rng.permutation(n_clusters)
        target_n = max(1, int(len(rows) * test_fraction))

        chosen_clusters: list[int] = []
        n_selected = 0
        for c in cluster_order:
            if n_selected >= target_n:
                break
            chosen_clusters.append(int(c))
            n_selected += len(rows_by_cluster[int(c)])
        test_rows = [r for c in chosen_clusters for r in rows_by_cluster[c]]
        print(f"Held out {len(test_rows)} chips from {len(chosen_clusters)} of {n_clusters} "
              f"spatial clusters (target {target_n}; still in-sample — see warning above).")

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
    _print_table(rf_metrics, cnn_metrics, len(test_rows), in_sample)

    return {"rf": rf_metrics, "cnn": cnn_metrics, "n_test": len(test_rows), "in_sample": in_sample}


def _print_table(rf: dict, cnn: dict, n_test: int, in_sample: bool) -> None:
    print(f"\n{'Model':<8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7}")
    print("-" * 46)
    for name, m in [("RF", rf), ("CNN", cnn)]:
        print(f"{name:<8} {m['accuracy']:>9.3f} {m['precision']:>10.3f} "
              f"{m['recall']:>8.3f} {m['f1']:>7.3f}")
    tag = " — IN-SAMPLE, optimistic" if in_sample else " (held-out test manifest)"
    print(f"\n(evaluated on {n_test} chips{tag})")
