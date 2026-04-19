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


def cmd_train(args: argparse.Namespace) -> None:
    from training_data import build_training_dataset
    from classifier import train

    jp2_files = _find_files(args.imagery, ".jp2")
    kml_files = _find_files(args.labels, ".kml") + _find_files(args.labels, ".kmz")

    if not jp2_files:
        sys.exit(f"No .jp2 files found in {args.imagery}")
    if not kml_files:
        sys.exit(f"No KML/KMZ files found in {args.labels}")

    stream_mask = _load_mask(args.hydro)

    with tempfile.TemporaryDirectory() as chip_dir:
        print(f"Extracting training chips ...")
        manifest = build_training_dataset(
            jp2_paths=jp2_files,
            kml_paths=kml_files,
            stream_mask=stream_mask,
            out_dir=chip_dir,
        )

        import csv
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        pos = [r for r in rows if int(r["label"]) == 1]
        neg = [r for r in rows if int(r["label"]) == 0]
        by_type: dict[str, int] = {}
        for r in pos:
            by_type[r["feature_type"]] = by_type.get(r["feature_type"], 0) + 1

        print(f"  Positive chips : {len(pos)}")
        for ftype, count in sorted(by_type.items()):
            print(f"    {ftype}: {count}")
        print(f"  Negative chips : {len(neg)}")

        if len(pos) == 0:
            sys.exit(
                "\nERROR: No positive chips were extracted.\n"
                "Check that your imagery tiles cover the labeled feature locations.\n"
                "Run: python src/cli.py check-labels --imagery ... --labels ..."
            )
        if len(pos) < 20:
            print(f"\nWARNING: Only {len(pos)} positive chips — model will be unreliable. "
                  "Add more imagery tiles that cover your labeled features.")

        print(f"Training Random Forest ...")
        train(manifest, args.model)

    print(f"Model saved to {args.model}")


def cmd_detect(args: argparse.Namespace) -> None:
    from classifier import load_model
    from masking import load_stream_lines
    from polygonizer import detect_rois
    from export import export_kml

    jp2_files = _find_files(args.imagery, ".jp2")
    if not jp2_files:
        sys.exit(f"No .jp2 files found in {args.imagery}")

    clf = load_model(args.model)
    stream_mask = _load_mask(args.hydro)
    stream_lines = load_stream_lines(args.hydro) if args.hydro else None

    all_dams: list[tuple] = []
    all_floods: list[tuple] = []

    for jp2_path in jp2_files:
        print(f"Processing {jp2_path} ...")
        dam_lines, flood_rois = detect_rois(
            jp2_path,
            clf,
            stream_mask=stream_mask,
            stream_lines=stream_lines,
            confidence_threshold=args.threshold,
        )
        print(f"  Dams: {len(dam_lines)}, Flooded areas: {len(flood_rois)}")
        all_dams.extend(dam_lines)
        all_floods.extend(flood_rois)

    print(f"Exporting {len(all_dams)} dam(s) and {len(all_floods)} flooded area(s) "
          f"to {args.output} ...")
    export_kml(all_dams, all_floods, args.output)
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="castor",
        description="CastorDetector — detect beaver activity in MML aerial imagery",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the classifier from KML labels")
    p_train.add_argument("--imagery", required=True, help="Directory of .jp2 files")
    p_train.add_argument("--labels", required=True, help="KML/KMZ file or directory")
    p_train.add_argument("--model", required=True, help="Output model path (.pkl)")
    p_train.add_argument("--hydro", default=None, help="Hydrography GeoPackage/Shapefile (optional)")

    # detect
    p_detect = sub.add_parser("detect", help="Run detection on .jp2 files")
    p_detect.add_argument("--imagery", required=True, help="Directory of .jp2 files")
    p_detect.add_argument("--model", required=True, help="Trained model path (.pkl)")
    p_detect.add_argument("--output", required=True, help="Output KML path")
    p_detect.add_argument("--hydro", default=None, help="Hydrography GeoPackage/Shapefile (optional)")
    p_detect.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default 0.5)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "detect":
        cmd_detect(args)


if __name__ == "__main__":
    main()
