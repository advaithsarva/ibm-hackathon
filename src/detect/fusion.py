"""Evidence fusion (sections 5.1 and 5.3).

Two kinds of fusion, both here because both answer "is the same person being seen twice":

  late_nms_merge   spatial. RGB and thermal boxes over an aligned pair are the same
                   person when they overlap, so merge them instead of counting two.
  fuse_log_odds    evidential. Independent sensor types compound into one calibrated
                   P(alive) instead of four numbers an operator has to weigh by eye.

The log-odds table lives in configs/global.yaml, not in this file (section 12.5), so
the likelihood ratios can be recalibrated without a code change.

    logit(P) = logit(p0) + sum_k z_k * log(LR_k)

A thermal blob alone could be an engine block. Thermal plus rPPG plus audio compounds
past 0.95, which is the principled answer to "how do you avoid false positives".

Standalone:
    python -m src.detect.fusion --demo
"""
import argparse
import datetime as dt
import json
import math

from src.hazard.formulas import REPO_ROOT
from src.hazard.zones import load_global

DEFAULT_PRIOR = 0.05      # base rate of a person being in an arbitrary collapsed cell
IOU_MERGE = 0.4


# --- spatial: RGB-T late fusion (section 5.1) ------------------------------------

def iou(a, b):
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def late_nms_merge(rgb_detections, thermal_detections, iou_threshold=IOU_MERGE):
    """Merge aligned RGB and thermal boxes of the same person into one record.

    A box seen in both modalities is stronger evidence than either alone, so the merged
    record keeps both confidences and is marked rgb_thermal. Unmatched detections pass
    through unchanged rather than being discarded: thermal sees through smoke that
    blinds RGB, and RGB sees people who are not warmer than their surroundings.
    """
    merged, used_thermal = [], set()

    for rgb in rgb_detections:
        best, best_iou = None, iou_threshold
        for i, thermal in enumerate(thermal_detections):
            if i in used_thermal or "bbox" not in thermal:
                continue
            overlap = iou(rgb["bbox"], thermal["bbox"])
            if overlap >= best_iou:
                best, best_iou = i, overlap

        if best is None:
            merged.append(dict(rgb))
            continue

        thermal = thermal_detections[best]
        used_thermal.add(best)
        merged.append({
            "type": "rgb_thermal",
            "conf": round(max(rgb["conf"], thermal["conf"]), 3),
            "bbox": rgb["bbox"],
            "rgb_conf": rgb["conf"],
            "thermal_conf": thermal["conf"],
            "temp_c": thermal.get("temp_c"),
            "iou": round(best_iou, 3),
        })

    merged.extend(dict(t) for i, t in enumerate(thermal_detections) if i not in used_thermal)
    return merged


# --- evidential: Bayesian log-odds (section 5.3) ---------------------------------

def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)         # keep the log finite at the boundaries
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def sensor_key(detection):
    """Detection record -> the likelihood-ratio key in configs/global.yaml."""
    kind = detection.get("type", "")
    if kind == "rgb_thermal":
        return "rgb_person"                  # the thermal half is counted separately
    return {
        "rgb_person": "rgb_person",
        "thermal_hotspot": "thermal_hotspot",
        "rppg_pulse": "rppg_pulse",
        "acoustic_distress": "acoustic_distress",
        "radar_vital": "radar_vital",
        "phone_ping": "phone_ping",
    }.get(kind)


def fuse_log_odds(detections, prior=DEFAULT_PRIOR, ratios=None, min_conf=0.3):
    """Independent sensor types -> P(alive).

    Each sensor TYPE fires at most once. Ten RGB boxes of one person are one camera
    agreeing with itself, not ten independent witnesses, and treating them as
    independent is how a fusion layer talks itself into false certainty.
    """
    if ratios is None:
        ratios = load_global()["fusion"]["likelihood_ratios"]

    logit = _logit(prior)
    fired = {}
    for d in detections:
        key = sensor_key(d)
        if key is None or key not in ratios:
            continue
        conf = d.get("conf", 0.0)
        if conf < min_conf:
            continue
        # Keep the strongest instance per sensor type.
        if key not in fired or conf > fired[key]:
            fired[key] = conf

        if d.get("type") == "rgb_thermal" and "thermal_hotspot" in ratios:
            t_conf = d.get("thermal_conf", 0.0)
            if t_conf >= min_conf and t_conf > fired.get("thermal_hotspot", 0.0):
                fired["thermal_hotspot"] = t_conf

    # z_k is binary (section 5.3): a sensor either fired or it did not. min_conf is the
    # gate that decides which. Scaling the log-LR by confidence instead would double-count
    # the detector's own uncertainty, which the likelihood ratio already prices in.
    for key in fired:
        logit += math.log(ratios[key])

    return _sigmoid(logit), fired


def estimate_count(detections, iou_threshold=IOU_MERGE):
    """How many distinct people the boxes imply. Non-spatial sensors do not add heads."""
    boxes = [d["bbox"] for d in detections if "bbox" in d]
    kept = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        if all(iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    if kept:
        return len(kept)
    # Audio or a pulse with no box still means at least one person is there.
    return 1 if any(sensor_key(d) for d in detections) else 0


def fuse_cell(cell_id, detections, source, prior=DEFAULT_PRIOR, ratios=None,
              timestamp=None):
    """Assemble the section 2.2 /api/detections record for one cell."""
    p_alive, fired = fuse_log_odds(detections, prior, ratios)
    return {
        "cell_id": cell_id,
        "source": source,
        "timestamp": timestamp or dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "detections": detections,
        "p_alive": round(p_alive, 4),
        "n_est": estimate_count(detections),
        "sensors_fired": sorted(fired),
    }


def _demo():
    """The demo beat from section 11: four weak sensors, one calibrated answer."""
    ratios = load_global()["fusion"]["likelihood_ratios"]
    steps = [
        ("thermal only", [{"type": "thermal_hotspot", "conf": 0.71, "temp_c": 34.2}]),
        ("+ rgb person", [{"type": "thermal_hotspot", "conf": 0.71, "temp_c": 34.2},
                          {"type": "rgb_person", "conf": 0.66, "bbox": [120, 88, 164, 190]}]),
        ("+ rppg pulse", [{"type": "thermal_hotspot", "conf": 0.71, "temp_c": 34.2},
                          {"type": "rgb_person", "conf": 0.66, "bbox": [120, 88, 164, 190]},
                          {"type": "rppg_pulse", "conf": 0.88, "bpm": 92}]),
        ("+ acoustic", [{"type": "thermal_hotspot", "conf": 0.71, "temp_c": 34.2},
                        {"type": "rgb_person", "conf": 0.66, "bbox": [120, 88, 164, 190]},
                        {"type": "rppg_pulse", "conf": 0.88, "bpm": 92},
                        {"type": "acoustic_distress", "conf": 0.54, "class": "Shout"}]),
    ]
    print(f"prior P(alive) = {DEFAULT_PRIOR}")
    for label, detections in steps:
        p, fired = fuse_log_odds(detections, ratios=ratios)
        print(f"  {label:<14} P(alive) = {p:.3f}   [{', '.join(sorted(fired))}]")

    record = fuse_cell("12.9716_77.5946", steps[-1][1], "drone_feed_03",
                       timestamp="2026-09-12T08:41:12Z")
    print(json.dumps(record, indent=2))


def _cli():
    ap = argparse.ArgumentParser(description="Detection fusion.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--detections", help="JSON file of detection records for one cell")
    ap.add_argument("--cell-id", default="unknown")
    ap.add_argument("--source", default="offline")
    ap.add_argument("--prior", type=float, default=DEFAULT_PRIOR)
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if not args.detections:
        ap.error("pass --demo or --detections")

    import pathlib
    path = pathlib.Path(args.detections)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"detections file not found: {path}")

    record = fuse_cell(args.cell_id, json.loads(path.read_text()), args.source, args.prior)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    _cli()
