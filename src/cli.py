"""Batch-processing CLI for CastorDetector."""

import argparse
import sys
import tempfile
from pathlib import Path


class _NullContext:
    """Context manager that yields a fixed path without creating/deleting it."""
    def __init__(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        self._path = path

    def __enter__(self) -> str:
        return self._path

    def __exit__(self, *_) -> None:
        pass


def _find_files(path: str, suffix: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    return sorted(str(f) for f in p.rglob(f"*{suffix}"))


def _load_mask(hydro_path: str | None, bbox=None):
    if hydro_path is None:
        return None
    from masking import build_stream_mask, BUFFER_METERS
    if bbox is not None:
        # Expand bbox by buffer so streams just outside the tile edge are included.
        minx, miny, maxx, maxy = bbox
        bbox = (minx - BUFFER_METERS, miny - BUFFER_METERS,
                maxx + BUFFER_METERS, maxy + BUFFER_METERS)
        print(f"Building stream mask from {hydro_path} (tile bbox only) ...")
    else:
        print(f"Building stream mask from {hydro_path} ...")
    return build_stream_mask(hydro_path, bbox=bbox)


def _tile_bbox(jp2_path: str):
    """Return (minx, miny, maxx, maxy) bounds of a JP2 file in its native CRS."""
    import rasterio
    with rasterio.open(jp2_path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def _union_bbox(jp2_paths: list[str]):
    """Return the union bounding box of all JP2 files."""
    import rasterio
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for p in jp2_paths:
        with rasterio.open(p) as src:
            b = src.bounds
        minx, miny = min(minx, b.left), min(miny, b.bottom)
        maxx, maxy = max(maxx, b.right), max(maxy, b.top)
    return (minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# train (RF)
# ---------------------------------------------------------------------------

def _load_classifier_config(path: str) -> dict:
    """Load a classifier config for train --classifier-config. Accepts either
    a raw {"type", "params"} dict, or a `tune` output file (its "best" entry's
    "config" is used)."""
    import json as _json

    with open(path) as f:
        data = _json.load(f)

    if isinstance(data, dict):
        if "type" in data and "params" in data:
            return data
        best = data.get("best")
        if isinstance(best, dict) and "config" in best:
            return best["config"]

    sys.exit(
        f"--classifier-config {path}: could not find a classifier config "
        "(expected a `tune` output file with a 'best' entry, or a raw "
        "{'type', 'params'} config)."
    )


def cmd_train(args: argparse.Namespace) -> None:
    from training_data import build_training_dataset
    from models.random_forest import make_classifier, train

    jp2_files = _find_files(args.imagery, ".jp2")
    kml_files = _find_files(args.labels, ".kml") + _find_files(args.labels, ".kmz")

    if not jp2_files:
        sys.exit(f"No .jp2 files found in {args.imagery}")
    if not kml_files:
        sys.exit(f"No KML/KMZ files found in {args.labels}")

    chip_dir_ctx = (
        tempfile.TemporaryDirectory()
        if args.chip_dir is None
        else _NullContext(args.chip_dir)
    )
    # GUI callers (src/app.py) build a hand-crafted Namespace that may not
    # carry newly-added attributes — fall back to the CLI default.
    neg_ratio = getattr(args, "neg_ratio", 1.0)
    run_cv = not getattr(args, "no_cv", False)
    classifier_config_path = getattr(args, "classifier_config", None)
    classifier_config = _load_classifier_config(classifier_config_path) if classifier_config_path else None

    with chip_dir_ctx as chip_dir:
        print("Extracting training chips ...")
        manifest = build_training_dataset(
            jp2_paths=jp2_files,
            kml_paths=kml_files,
            out_dir=chip_dir,
            hydro_path=args.hydro,
            hydro_flood_samples=args.flood_samples,
            hydro_negatives=not args.no_hydro_negatives,
            augment_positives=args.augment_positives,
            neg_ratio=neg_ratio,
        )

        import csv
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        flood_chips = [r for r in rows if int(r["label"]) == 1]
        neg_chips   = [r for r in rows if int(r["label"]) == 0]
        by_type: dict[str, int] = {}
        for r in flood_chips:
            by_type[r["feature_type"]] = by_type.get(r["feature_type"], 0) + 1
        neg_by_type: dict[str, int] = {}
        for r in neg_chips:
            neg_by_type[r["feature_type"]] = neg_by_type.get(r["feature_type"], 0) + 1

        print(f"  Flood chips    : {len(flood_chips)}")
        for ftype, count in sorted(by_type.items()):
            print(f"    {ftype}: {count}")
        print(f"  Negative chips : {len(neg_chips)}")
        for ftype, count in sorted(neg_by_type.items()):
            print(f"    {ftype}: {count}")

        if not flood_chips:
            sys.exit(
                "\nERROR: No positive chips were extracted.\n"
                "Check that your imagery tiles cover the labeled feature locations."
            )
        if len(flood_chips) < 20:
            print(f"\nWARNING: Only {len(flood_chips)} positive chips — model will be "
                  "unreliable. Add more imagery tiles that cover your labeled features.")

        # Spatial CV runs (by default) before the final fit, sharing chip_dir as the
        # feature cache — train() below reuses the same cache, so features aren't
        # extracted twice in one `train` invocation. Works with a temp chip_dir too:
        # the cache lives for the lifetime of the `with chip_dir_ctx` block, which
        # covers both this CV run and the train() call.
        cv_results = None
        if run_cv:
            from models.evaluate import evaluate_rf_spatial
            print("Running spatial CV before saving the final model ...")
            classifier_factory = (
                (lambda **kw: make_classifier(config=classifier_config, **kw))
                if classifier_config is not None else None
            )
            try:
                cv_results = evaluate_rf_spatial(
                    manifest, cache_dir=chip_dir, use_cache=True, per_class=True,
                    classifier_factory=classifier_factory,
                )
            except ValueError as exc:
                print(f"  Skipping CV: {exc}")
                cv_results = None

        print("Training Random Forest ..." if classifier_config is None
              else f"Training classifier (config: {classifier_config.get('type')}) ...")
        train(
            manifest, args.model,
            classifier_config=classifier_config,
            cache_dir=chip_dir, use_cache=True,
            cv_results=cv_results,
        )

    print(f"Model saved to {args.model}")
    print(f"Model metadata saved to {Path(args.model).with_suffix('.json')}")
    if args.chip_dir is not None:
        manifest_path = Path(args.chip_dir) / "manifest.csv"
        print(f"Chips and manifest saved to {args.chip_dir}/")
        print(f"  Run evaluate-rf with: --manifest {manifest_path}")


# ---------------------------------------------------------------------------
# cnn-train
# ---------------------------------------------------------------------------

def cmd_cnn_train(args: argparse.Namespace) -> None:
    from training_data import build_training_dataset
    from models.cnn_train import train_cnn

    jp2_files = _find_files(args.imagery, ".jp2")
    kml_files = _find_files(args.labels, ".kml") + _find_files(args.labels, ".kmz")

    if not jp2_files:
        sys.exit(f"No .jp2 files found in {args.imagery}")
    if not kml_files:
        sys.exit(f"No KML/KMZ files found in {args.labels}")

    with tempfile.TemporaryDirectory() as chip_dir:
        print("Extracting training chips ...")
        manifest = build_training_dataset(
            jp2_paths=jp2_files,
            kml_paths=kml_files,
            out_dir=chip_dir,
            hydro_path=args.hydro,
        )
        print(f"Training CNN (epochs={args.epochs}, lr={args.lr}) ...")
        train_cnn(
            manifest_path=manifest,
            model_path=args.model,
            norm_stats_path=args.norm_stats,
            epochs=args.epochs,
            lr=args.lr,
        )

    print(f"CNN model saved to {args.model}")
    print(f"Norm stats saved to {args.norm_stats}")


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

def _resolve_detect_threshold(
    explicit_threshold: float | None,
    method: str,
    rf_model_path: str | None,
) -> float:
    """Resolve the confidence threshold for `detect`: an explicit
    --threshold always wins (this is how the GUI, which always passes one,
    picks its threshold). Otherwise fall back to the RF model's .json
    metadata sidecar recommended_threshold (R3.5), else 0.5. Prints which
    source was used."""
    if explicit_threshold is not None:
        threshold = float(explicit_threshold)
        print(f"Using explicit confidence threshold: {threshold:.3f}")
        return threshold

    if method in ("rf", "both") and rf_model_path:
        from models.random_forest import load_model_metadata
        meta = load_model_metadata(rf_model_path)
        rec = meta.get("recommended_threshold") if meta else None
        if rec is not None:
            threshold = float(rec)
            print(f"Using recommended threshold from model metadata: {threshold:.3f}")
            return threshold

    print("Using default confidence threshold: 0.500")
    return 0.5


def cmd_detect(args: argparse.Namespace) -> None:
    from polygonizer import detect_rois_rf_segmentation, detect_rois_cnn
    from export import export_kml

    jp2_files = _find_files(args.imagery, ".jp2")
    if not jp2_files:
        sys.exit(f"No .jp2 files found in {args.imagery}")

    method = args.method

    # Load models as needed
    rf_clf = cnn_model = norm_stats = None

    if method in ("rf", "both"):
        from models.random_forest import check_feature_length, load_model as load_rf
        print(f"Loading RF model from {args.rf_model} ...")
        rf_clf = load_rf(args.rf_model)
        try:
            check_feature_length(rf_clf)
        except ValueError as exc:
            sys.exit(str(exc))

    if method in ("cnn", "both"):
        import json
        from models.cnn_handler import load_cnn
        cnn_model = load_cnn(args.cnn_model)
        with open(args.norm_stats) as f:
            norm_stats = json.load(f)

    # Threshold resolution: an explicit --threshold always wins (this is how the
    # GUI, which always passes one, picks the threshold). Otherwise fall back to
    # the RF model's metadata sidecar (R3.5) recommended_threshold, else 0.5.
    threshold = _resolve_detect_threshold(getattr(args, "threshold", None), method, args.rf_model)

    # GUI callers (src/app.py) build a hand-crafted Namespace that may not
    # carry newly-added attributes — fall back to the CLI defaults.
    from polygonizer import MIN_AREA_M2
    min_area = getattr(args, "min_area", MIN_AREA_M2)
    seed_threshold = getattr(args, "seed_threshold", None)
    smooth = not getattr(args, "no_smooth", False)

    all_rois: list[tuple] = []

    for jp2_path in jp2_files:
        print(f"Processing {jp2_path} ...")
        # Build mask scoped to this tile — avoids loading millions of features globally.
        stream_mask = _load_mask(args.hydro, bbox=_tile_bbox(jp2_path) if args.hydro else None)

        if method == "rf":
            rois = detect_rois_rf_segmentation(
                jp2_path, rf_clf, stream_mask, threshold,
                min_area_m2=min_area, seed_threshold=seed_threshold, smooth=smooth,
            )
            print(f"  RF detections: {len(rois)}")
            all_rois.extend(rois)

        elif method == "cnn":
            rois = detect_rois_cnn(
                jp2_path, cnn_model, norm_stats, stream_mask, threshold,
                min_area_m2=min_area,
            )
            print(f"  CNN detections: {len(rois)}")
            all_rois.extend(rois)

        elif method == "both":
            rf_rois  = detect_rois_rf_segmentation(
                jp2_path, rf_clf, stream_mask, threshold,
                min_area_m2=min_area, seed_threshold=seed_threshold, smooth=smooth,
            )
            cnn_rois = detect_rois_cnn(
                jp2_path, cnn_model, norm_stats, stream_mask, threshold,
                min_area_m2=min_area,
            )
            print(f"  RF detections: {len(rf_rois)}  CNN detections: {len(cnn_rois)}")
            combined = _merge_multi_model(rf_rois, cnn_rois)
            print(f"  After merge — rf:{sum(1 for r in combined if r[3]=='rf')}  "
                  f"cnn:{sum(1 for r in combined if r[3]=='cnn')}  "
                  f"both:{sum(1 for r in combined if r[3]=='both')}")
            all_rois.extend(combined)

    print(f"Exporting {len(all_rois)} detection(s) to {args.output} ...")
    export_kml(all_rois, args.output)
    print("Done.")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    from models.evaluate import evaluate_models
    evaluate_models(
        manifest_path=args.manifest,
        rf_model_path=args.rf_model,
        cnn_model_path=args.cnn_model,
        norm_stats_path=args.norm_stats,
        test_manifest_path=args.test_manifest,
        test_fraction=args.test_fraction,
    )


def cmd_evaluate_rf(args: argparse.Namespace) -> None:
    from models.evaluate import evaluate_rf_spatial
    evaluate_rf_spatial(
        manifest_path=args.manifest,
        rf_model_path=args.rf_model,
        cluster_radius=args.cluster_radius,
        n_splits=args.n_splits,
        oof_path=args.oof_path,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        per_class=args.per_class,
    )


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------

def cmd_tune(args: argparse.Namespace) -> None:
    from models.evaluate import tune_rf
    tune_rf(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        out_path=args.out,
        cluster_radius=args.cluster_radius,
        n_splits=args.n_splits,
    )


# ---------------------------------------------------------------------------
# Agreement merge logic
# ---------------------------------------------------------------------------

def _merge_multi_model(
    rf_rois: list[tuple],
    cnn_rois: list[tuple],
) -> list[tuple]:
    """
    Tag ROIs as 'rf', 'cnn', or 'both' based on spatial overlap.
    Agreeing pairs are merged into their union polygon, taking the higher confidence.
    """
    tagged: list[tuple] = []
    matched_cnn: set[int] = set()

    for rf_poly, rf_conf, rf_area in rf_rois:
        matched = False
        for j, (cnn_poly, cnn_conf, cnn_area) in enumerate(cnn_rois):
            if rf_poly.intersects(cnn_poly):
                merged_poly = rf_poly.union(cnn_poly)
                conf = max(rf_conf, cnn_conf)
                tagged.append((merged_poly, conf, merged_poly.area, "both"))
                matched_cnn.add(j)
                matched = True
                break
        if not matched:
            tagged.append((rf_poly, rf_conf, rf_area, "rf"))

    for j, (cnn_poly, cnn_conf, cnn_area) in enumerate(cnn_rois):
        if j not in matched_cnn:
            tagged.append((cnn_poly, cnn_conf, cnn_area, "cnn"))

    return tagged


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="castor",
        description="CastorDetector — detect beaver activity in MML aerial imagery",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train (RF) --
    p_train = sub.add_parser("train", help="Train the Random Forest classifier")
    p_train.add_argument("--imagery", required=True, help="Directory of .jp2 files")
    p_train.add_argument("--labels",  required=True, help="KML/KMZ file or directory")
    p_train.add_argument("--model",   required=True, help="Output RF model path (.pkl)")
    p_train.add_argument("--hydro",     default=None, help="Hydrography directory or file (optional)")
    p_train.add_argument("--chip-dir",  default=None, dest="chip_dir",
                         help="Persist extracted chips to this directory (enables evaluate-rf later). "
                              "Default: temp directory deleted after training.")
    p_train.add_argument("--augment-positives", type=int, default=6, dest="augment_positives",
                         help="Extra offset chips per positive label point (default 6). "
                              "Set to 0 to disable augmentation.")
    p_train.add_argument("--flood-samples", type=int, default=0, dest="flood_samples",
                         help="Auto-extract this many flood chips from the tulvaalue layer "
                              "in --hydro data (default 0). Requires --hydro.")
    p_train.add_argument("--no-hydro-negatives", action="store_true", dest="no_hydro_negatives",
                         help="Sample auto-negatives from the full imagery extent instead of "
                              "the stream corridor. Useful when --hydro is only needed for "
                              "--flood-samples and you do not want stream-mask filtering.")
    p_train.add_argument("--neg-ratio", type=float, default=1.0, dest="neg_ratio",
                         help="Auto-negative chip count relative to the number of positive "
                              "chips after augmentation (default 1.0). Ignored if the "
                              "positive:negative balance is overridden elsewhere.")
    p_train.add_argument("--classifier-config", default=None, dest="classifier_config",
                         help="Path to a classifier config JSON — either a `tune` output file "
                              "(its best candidate is used) or a raw {'type','params'} config "
                              "as accepted by models.random_forest.make_classifier. Default: "
                              "the project's default RandomForestClassifier.")
    p_train.add_argument("--no-cv", action="store_true", dest="no_cv",
                         help="Skip the spatial CV run performed after training by default. "
                              "CV results (including the recommended threshold) are stored in "
                              "the model's .json metadata sidecar when CV runs.")

    # -- cnn-train --
    p_cnn = sub.add_parser("cnn-train", help="Train the CNN classifier (Prithvi head)")
    p_cnn.add_argument("--imagery",    required=True, help="Directory of .jp2 files")
    p_cnn.add_argument("--labels",     required=True, help="KML/KMZ file or directory")
    p_cnn.add_argument("--model",      required=True, help="Output CNN weights path (.pth)")
    p_cnn.add_argument("--norm-stats", required=True, dest="norm_stats",
                       help="Output norm stats path (data/models/norm_stats.json)")
    p_cnn.add_argument("--hydro",      default=None,  help="Hydrography directory or file (optional)")
    p_cnn.add_argument("--epochs",     type=int,   default=30, help="Training epochs (default 30)")
    p_cnn.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default 1e-3)")

    # -- detect --
    p_detect = sub.add_parser("detect", help="Run detection on .jp2 files")
    p_detect.add_argument("--imagery",   required=True, help="Directory of .jp2 files")
    p_detect.add_argument("--output",    required=True, help="Output KML path")
    p_detect.add_argument("--method",    default="rf", choices=["rf", "cnn", "both"],
                          help="Detection method: rf, cnn, or both (default: rf)")
    p_detect.add_argument("--rf-model",  default=None, dest="rf_model",
                          help="RF model path (.pkl) — required for --method rf or both")
    p_detect.add_argument("--cnn-model", default=None, dest="cnn_model",
                          help="CNN weights path (.pth) — required for --method cnn or both")
    p_detect.add_argument("--norm-stats", default=None, dest="norm_stats",
                          help="Norm stats JSON — required for --method cnn or both")
    p_detect.add_argument("--hydro",     default=None, help="Hydrography directory or file (optional)")
    p_detect.add_argument("--threshold", type=float, default=None,
                          help="Confidence threshold. Default: the RF model's recommended "
                               "threshold from its .json metadata sidecar (R3.5) if present, "
                               "else 0.5.")
    p_detect.add_argument("--min-area", type=float, default=2048.0, dest="min_area",
                          help="Minimum detection area in m^2 (default 2048 — "
                               "roughly 2 RF patches; a single 64px patch is 1024 m^2)")
    p_detect.add_argument("--seed-threshold", type=float, default=None, dest="seed_threshold",
                          help="RF only: hysteresis seed threshold — a region is kept only if "
                               "it contains a cell at or above this confidence (default: "
                               "min(--threshold + 0.15, 0.95))")
    p_detect.add_argument("--no-smooth", action="store_true", dest="no_smooth",
                          help="RF only: disable 3x3 NaN-aware smoothing of the probability "
                               "map before hysteresis thresholding")

    # -- evaluate --
    p_eval = sub.add_parser("evaluate", help="Compare RF vs CNN on a held-out test set")
    p_eval.add_argument("--manifest",      required=True, help="Training manifest CSV")
    p_eval.add_argument("--rf-model",      required=True, dest="rf_model",  help="RF model path (.pkl)")
    p_eval.add_argument("--cnn-model",     required=True, dest="cnn_model", help="CNN weights path (.pth)")
    p_eval.add_argument("--norm-stats",    required=True, dest="norm_stats", help="Norm stats JSON")
    p_eval.add_argument("--test-manifest", default=None, dest="test_manifest",
                        help="Separate manifest CSV built from held-out tiles, for a true "
                             "out-of-sample comparison. Without this, evaluation falls back to "
                             "a spatially-held-out slice of --manifest and is still in-sample "
                             "(both models were trained on it) — a warning is printed.")
    p_eval.add_argument("--test-fraction", type=float, default=0.2, dest="test_fraction",
                        help="Fraction of --manifest to hold out when --test-manifest is not "
                             "given (default 0.2)")

    # -- evaluate-rf --
    p_eval_rf = sub.add_parser(
        "evaluate-rf",
        help="Evaluate RF with pooled out-of-fold spatial cross-validation",
    )
    p_eval_rf.add_argument("--manifest",       required=True, help="Training manifest CSV")
    p_eval_rf.add_argument("--rf-model",       default=None, dest="rf_model",
                           help="RF model path (.pkl) — optional, kept for reference in the "
                                "output; CV folds are always trained fresh with "
                                "models.random_forest.make_classifier's defaults")
    p_eval_rf.add_argument("--cluster-radius", type=float, default=500.0, dest="cluster_radius",
                           help="Group label points within this radius (metres) into one spatial "
                                "cluster (default 500)")
    p_eval_rf.add_argument("--n-splits", type=int, default=5, dest="n_splits",
                           help="Number of CV folds, capped at the number of spatial clusters "
                                "(default 5)")
    p_eval_rf.add_argument("--oof-path", default=None, dest="oof_path",
                           help="Where to write out-of-fold predictions CSV "
                                "(default: <manifest_dir>/oof.csv)")
    p_eval_rf.add_argument("--cache-dir", default=None, dest="cache_dir",
                           help="Where to read/write the feature cache and default OOF CSV "
                                "(default: the manifest's own directory)")
    p_eval_rf.add_argument("--no-cache", action="store_true", dest="no_cache",
                           help="Force recomputation of the feature matrix instead of reusing "
                                "a cached one")
    p_eval_rf.add_argument("--per-class", action="store_true", dest="per_class",
                           help="Print the per-feature-type breakdown table (always computed "
                                "and returned; this only toggles printing it)")

    # -- tune --
    p_tune = sub.add_parser(
        "tune",
        help="Compare classifier candidates (RF variants, ExtraTrees, HistGradientBoosting, "
             "calibrated RF) with the same pooled spatial CV and write a ranked report",
    )
    p_tune.add_argument("--manifest", required=True, help="Training manifest CSV")
    p_tune.add_argument("--cache-dir", default=None, dest="cache_dir",
                        help="Where to read/write the shared feature cache and the default "
                             "tuning.json output (default: the manifest's own directory)")
    p_tune.add_argument("--out", default=None,
                        help="Output JSON path (default: <cache-dir or manifest dir>/tuning.json)")
    p_tune.add_argument("--cluster-radius", type=float, default=500.0, dest="cluster_radius",
                        help="Group label points within this radius (metres) into one spatial "
                             "cluster (default 500)")
    p_tune.add_argument("--n-splits", type=int, default=5, dest="n_splits",
                        help="Number of CV folds per candidate, capped at the number of spatial "
                             "clusters (default 5)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate train flags
    if args.command == "train":
        if getattr(args, "flood_samples", 0) > 0 and not args.hydro:
            parser.error("--flood-samples requires --hydro")

    # Validate model paths for detect command
    if args.command == "detect":
        if args.method in ("rf", "both") and not args.rf_model:
            parser.error("--rf-model is required when --method is 'rf' or 'both'")
        if args.method in ("cnn", "both") and not args.cnn_model:
            parser.error("--cnn-model is required when --method is 'cnn' or 'both'")
        if args.method in ("cnn", "both") and not args.norm_stats:
            parser.error("--norm-stats is required when --method is 'cnn' or 'both'")

    dispatch = {
        "train":       cmd_train,
        "cnn-train":   cmd_cnn_train,
        "detect":      cmd_detect,
        "evaluate":    cmd_evaluate,
        "evaluate-rf": cmd_evaluate_rf,
        "tune":        cmd_tune,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
