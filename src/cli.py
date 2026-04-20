"""Batch-processing CLI for CastorDetector."""

import argparse
import sys
import tempfile
from pathlib import Path


def _find_files(path: str, suffix: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    return sorted(str(f) for f in p.rglob(f"*{suffix}"))


def _load_mask(hydro_path: str | None):
    if hydro_path is None:
        return None
    from masking import build_stream_mask
    return build_stream_mask(hydro_path)


# ---------------------------------------------------------------------------
# train (RF)
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from training_data import build_training_dataset
    from models.random_forest import train

    jp2_files = _find_files(args.imagery, ".jp2")
    kml_files = _find_files(args.labels, ".kml") + _find_files(args.labels, ".kmz")

    if not jp2_files:
        sys.exit(f"No .jp2 files found in {args.imagery}")
    if not kml_files:
        sys.exit(f"No KML/KMZ files found in {args.labels}")

    stream_mask = _load_mask(args.hydro)

    with tempfile.TemporaryDirectory() as chip_dir:
        print("Extracting training chips ...")
        manifest = build_training_dataset(
            jp2_paths=jp2_files,
            kml_paths=kml_files,
            stream_mask=stream_mask,
            out_dir=chip_dir,
        )

        import csv
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        flood_chips = [r for r in rows if int(r["label"]) == 1]
        neg_chips   = [r for r in rows if int(r["label"]) == 0]
        by_type: dict[str, int] = {}
        for r in flood_chips:
            by_type[r["feature_type"]] = by_type.get(r["feature_type"], 0) + 1

        print(f"  Flood chips    : {len(flood_chips)}")
        for ftype, count in sorted(by_type.items()):
            print(f"    {ftype}: {count}")
        print(f"  Negative chips : {len(neg_chips)}")

        if not flood_chips:
            sys.exit(
                "\nERROR: No positive chips were extracted.\n"
                "Check that your imagery tiles cover the labeled feature locations."
            )
        if len(flood_chips) < 20:
            print(f"\nWARNING: Only {len(flood_chips)} positive chips — model will be "
                  "unreliable. Add more imagery tiles that cover your labeled features.")

        print("Training Random Forest ...")
        train(manifest, args.model)

    print(f"Model saved to {args.model}")


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

    stream_mask = _load_mask(args.hydro)

    with tempfile.TemporaryDirectory() as chip_dir:
        print("Extracting training chips ...")
        manifest = build_training_dataset(
            jp2_paths=jp2_files,
            kml_paths=kml_files,
            stream_mask=stream_mask,
            out_dir=chip_dir,
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

def cmd_detect(args: argparse.Namespace) -> None:
    from polygonizer import detect_rois, detect_rois_cnn
    from export import export_kml

    jp2_files = _find_files(args.imagery, ".jp2")
    if not jp2_files:
        sys.exit(f"No .jp2 files found in {args.imagery}")

    stream_mask = _load_mask(args.hydro)
    method = args.method

    # Load models as needed
    rf_clf = cnn_model = norm_stats = None

    if method in ("rf", "both"):
        from models.random_forest import load_model as load_rf
        rf_clf = load_rf(args.rf_model)

    if method in ("cnn", "both"):
        import json
        from models.cnn_handler import load_cnn
        cnn_model = load_cnn(args.cnn_model)
        with open(args.norm_stats) as f:
            norm_stats = json.load(f)

    all_rois: list[tuple] = []

    for jp2_path in jp2_files:
        print(f"Processing {jp2_path} ...")

        if method == "rf":
            rois = detect_rois(jp2_path, rf_clf, stream_mask, args.threshold)
            print(f"  RF detections: {len(rois)}")
            all_rois.extend(rois)

        elif method == "cnn":
            rois = detect_rois_cnn(jp2_path, cnn_model, norm_stats, stream_mask, args.threshold)
            print(f"  CNN detections: {len(rois)}")
            all_rois.extend(rois)

        elif method == "both":
            rf_rois  = detect_rois(jp2_path, rf_clf, stream_mask, args.threshold)
            cnn_rois = detect_rois_cnn(jp2_path, cnn_model, norm_stats, stream_mask, args.threshold)
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
        test_fraction=args.test_fraction,
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
    p_train.add_argument("--hydro",   default=None,  help="Hydrography directory or file (optional)")

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
    p_detect.add_argument("--threshold", type=float, default=0.5,
                          help="Confidence threshold (default 0.5)")

    # -- evaluate --
    p_eval = sub.add_parser("evaluate", help="Compare RF vs CNN on a held-out test set")
    p_eval.add_argument("--manifest",      required=True, help="Training manifest CSV")
    p_eval.add_argument("--rf-model",      required=True, dest="rf_model",  help="RF model path (.pkl)")
    p_eval.add_argument("--cnn-model",     required=True, dest="cnn_model", help="CNN weights path (.pth)")
    p_eval.add_argument("--norm-stats",    required=True, dest="norm_stats", help="Norm stats JSON")
    p_eval.add_argument("--test-fraction", type=float, default=0.2, dest="test_fraction",
                        help="Fraction of manifest to hold out (default 0.2)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate model paths for detect command
    if args.command == "detect":
        if args.method in ("rf", "both") and not args.rf_model:
            parser.error("--rf-model is required when --method is 'rf' or 'both'")
        if args.method in ("cnn", "both") and not args.cnn_model:
            parser.error("--cnn-model is required when --method is 'cnn' or 'both'")
        if args.method in ("cnn", "both") and not args.norm_stats:
            parser.error("--norm-stats is required when --method is 'cnn' or 'both'")

    dispatch = {
        "train":     cmd_train,
        "cnn-train": cmd_cnn_train,
        "detect":    cmd_detect,
        "evaluate":  cmd_evaluate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
