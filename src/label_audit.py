"""Label audit — find training label points whose out-of-fold prediction
disagrees with their hand-assigned label.

R1.5 / G2.2. Reads the out-of-fold predictions CSV written by
``models.evaluate.evaluate_rf_spatial`` (``<chip_dir>/oof.csv``, columns
``path,feature_type,label,x,y,cluster,fold,prob``) and aggregates per label
*point* — keyed by ``(x, y, feature_type)``, the same key
``evaluate_rf_spatial`` uses for its point-level metrics, so augmented chips
of one placemark are pooled into a single mean probability rather than
counted separately.

A point is flagged as suspicious when the model's out-of-fold opinion
strongly disagrees with the hand-assigned label:
- a positive (label 1) with a low mean probability (looks like it was
  mislabelled, or isn't actually visible in the imagery), or
- a negative (label 0) with a high mean probability (looks like a missed
  positive, or a hard negative that's actually borderline).

Auto-sampled negatives (``feature_type == "auto_negative"``) are excluded
from the negative-side check by default: they're placed by random sampling,
not by a human decision, so a high score there is expected occasionally and
much less informative than a hand-labelled negative scoring high.
"""
import csv
from collections import defaultdict

# feature_type values used for negatives sampled automatically (not hand-labelled).
AUTO_NEGATIVE_TYPES = frozenset({"auto_negative"})


def audit_labels(
    oof_csv: str,
    low: float = 0.2,
    high: float = 0.8,
    include_auto_negatives: bool = False,
) -> list[dict]:
    """Return suspicious label points from an out-of-fold predictions CSV.

    Args:
        oof_csv: path to the CSV written by evaluate_rf_spatial (or any CSV
            with the same columns: path, feature_type, label, x, y, cluster,
            fold, prob).
        low: positives with mean OOF probability below this are flagged.
        high: negatives with mean OOF probability above this are flagged.
        include_auto_negatives: if False (default), negatives with
            feature_type in AUTO_NEGATIVE_TYPES are never flagged.

    Returns:
        A list of dicts — one per suspicious label point — each with keys
        "x", "y" (EPSG:3067, matching the input CSV), "feature_type",
        "label" (0/1), "mean_prob", and "n_chips" (how many chip rows,
        including augmented copies, were averaged). Sorted most-suspicious
        first (largest disagreement between label and mean_prob).
    """
    with open(oof_csv) as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple, list[float]] = defaultdict(list)
    meta: dict[tuple, dict] = {}
    for r in rows:
        key = (r["x"], r["y"], r["feature_type"])
        try:
            prob = float(r["prob"])
        except (TypeError, ValueError):
            continue
        groups[key].append(prob)
        if key not in meta:
            meta[key] = {
                "x": float(r["x"]),
                "y": float(r["y"]),
                "feature_type": r["feature_type"],
                "label": int(r["label"]),
            }

    suspicious: list[dict] = []
    for key, probs in groups.items():
        info = meta[key]
        mean_prob = sum(probs) / len(probs)
        label = info["label"]
        ftype = info["feature_type"]

        if label == 1 and mean_prob < low:
            score = low - mean_prob
        elif label == 0 and mean_prob > high:
            if ftype in AUTO_NEGATIVE_TYPES and not include_auto_negatives:
                continue
            score = mean_prob - high
        else:
            continue

        suspicious.append({
            "x": info["x"],
            "y": info["y"],
            "feature_type": ftype,
            "label": label,
            "mean_prob": mean_prob,
            "n_chips": len(probs),
            "_score": score,
        })

    suspicious.sort(key=lambda d: d["_score"], reverse=True)
    for d in suspicious:
        d.pop("_score", None)
    return suspicious
