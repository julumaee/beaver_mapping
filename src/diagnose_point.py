"""Diagnostic: evaluate the classifier at a known point and save a chip PNG."""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
from ingestion import TILE_SIZE
from models.random_forest import load_model, predict

_WGS84_TO_ETRS = Transformer.from_crs(4326, 3067, always_xy=True)


def find_covering_jp2(jp2_dir: str, x: float, y: float) -> list[str]:
    hits = []
    for p in Path(jp2_dir).rglob("*.jp2"):
        with rasterio.open(str(p)) as src:
            b = src.bounds
            if b.left <= x <= b.right and b.bottom <= y <= b.top:
                hits.append(str(p))
    return hits


def extract_chip_at(jp2_path: str, x: float, y: float) -> np.ndarray:
    half = TILE_SIZE // 2
    with rasterio.open(jp2_path) as src:
        col, row = ~src.transform * (x, y)
        col, row = int(col), int(row)
        col_off = max(col - half, 0)
        row_off = max(row - half, 0)
        col_off = min(col_off, src.width - TILE_SIZE)
        row_off = min(row_off, src.height - TILE_SIZE)
        win = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
        data = src.read(window=win)
    return data


def save_png(chip: np.ndarray, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping PNG)")
        return

    # Display as false-colour CIR: NIR→R, Red→G, Green→B
    rgb = np.stack([chip[0], chip[1], chip[2]], axis=-1).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title("CIR chip (NIR=R, Red=G, Green=B)")
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chip image saved → {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose classifier at a known point")
    parser.add_argument("--lon", type=float, required=True, help="WGS84 longitude")
    parser.add_argument("--lat", type=float, required=True, help="WGS84 latitude")
    parser.add_argument("--imagery", required=True, help="Directory of .jp2 files")
    parser.add_argument("--model", required=True, help="Trained model .pkl")
    parser.add_argument("--out", default="chip_debug.png", help="Output PNG path")
    args = parser.parse_args()

    x, y = _WGS84_TO_ETRS.transform(args.lon, args.lat)
    print(f"\nWGS84  : {args.lat:.6f}°N  {args.lon:.6f}°E")
    print(f"EPSG:3067: x={x:.1f}  y={y:.1f}")

    jp2s = find_covering_jp2(args.imagery, x, y)
    if not jp2s:
        print("\nERROR: no .jp2 file covers this point.")
        sys.exit(1)
    print(f"\nCovering tile(s): {jp2s}")

    jp2 = jp2s[0]
    chip = extract_chip_at(jp2, x, y)
    print(f"Chip shape: {chip.shape}  dtype: {chip.dtype}")
    print(f"Band stats (NIR / Red / Green):")
    for i, name in enumerate(["NIR", "Red", "Green"]):
        b = chip[i]
        print(f"  {name}: min={b.min():3d}  max={b.max():3d}  mean={b.mean():.1f}")

    clf = load_model(args.model)
    label, confidence = predict(clf, chip)
    label_name = {0: "negative", 1: "dam", 2: "flood"}.get(label, "?")
    print(f"\nClassifier result: label={label} ({label_name})  confidence={confidence:.3f}")

    # Per-class probabilities
    from spectral import extract_features
    feats = extract_features(chip).reshape(1, -1)
    probs = clf.predict_proba(feats)[0]
    classes = clf.classes_
    print("Per-class probabilities:")
    name_map = {0: "negative", 1: "dam", 2: "flood"}
    for cls, prob in zip(classes, probs):
        print(f"  {name_map.get(cls, cls)}: {prob:.3f}")

    save_png(chip, args.out)


if __name__ == "__main__":
    main()
