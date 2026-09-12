"""Thermal hotspot detection (section 5.1).

Two paths, and the cheap one is not a fallback:

  blob threshold  adaptive percentile over a radiometric frame. Pure numpy, no weights,
                  no GPU, works the moment a thermal camera is plugged in.
  YOLO thermal    a model trained on FLIR ADAS / LLVIP, for when shape matters more than
                  temperature. Shares the GPU guard with rgb_yolo, never concurrent.

A hotspot on its own is weak evidence: an engine block reads warm too. It earns its
weight only through the log-odds fusion in fusion.py.

Standalone:
    python -m src.detect.thermal --npy thermal_frame.npy
"""
import argparse
import pathlib

import numpy as np

from src.detect.loader import gpu_model

# Human skin through clothing in open air. Body core is ~37 C but the surface a camera
# sees is cooler, and rubble cools it further, so the window is deliberately wide.
HUMAN_MIN_C, HUMAN_MAX_C = 26.0, 40.0
PERCENTILE = 97.0        # adaptive: warm relative to this scene, not to a fixed number
MIN_BLOB_PIXELS = 12


def adaptive_threshold(frame_c, percentile=PERCENTILE):
    """Threshold at a scene percentile, floored at the coldest plausible human reading.

    Fixed thresholds fail twice over: at noon everything clears 30 C, and at night a
    survivor may be the only warm object in frame. A percentile adapts to both.
    """
    frame = np.asarray(frame_c, dtype=float)
    if frame.ndim != 2:
        raise ValueError(f"expected a 2-D temperature frame, got shape {frame.shape}")
    if not np.isfinite(frame).any():
        raise ValueError("thermal frame contains no finite readings")
    return max(float(np.nanpercentile(frame, percentile)), HUMAN_MIN_C)


def _label_blobs(mask):
    """Connected components, 4-connectivity, iterative flood fill.

    Written out rather than pulled from scipy.ndimage: it is twenty lines, and it keeps
    the CPU detection path free of a dependency the GPU path does not need.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    current = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            current += 1
            stack = [(sy, sx)]
            labels[sy, sx] = current
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current


def detect_hotspots(frame_c, percentile=PERCENTILE, min_pixels=MIN_BLOB_PIXELS):
    """Radiometric frame in Celsius -> thermal_hotspot detection records.

    Blobs hotter than the human window are reported with reduced confidence rather than
    dropped: a fire matters to the operator even though it is not a survivor.
    """
    frame = np.asarray(frame_c, dtype=float)
    threshold = adaptive_threshold(frame, percentile)
    labels, count = _label_blobs(np.nan_to_num(frame, nan=-np.inf) >= threshold)

    detections = []
    for blob_id in range(1, count + 1):
        ys, xs = np.where(labels == blob_id)
        if len(ys) < min_pixels:
            continue
        peak = float(frame[ys, xs].max())
        mean = float(frame[ys, xs].mean())

        if HUMAN_MIN_C <= peak <= HUMAN_MAX_C:
            # Confidence rises with how far the blob sits above the scene threshold,
            # capped so a single hot pixel cannot claim certainty.
            conf = min(0.9, 0.35 + (peak - threshold) / 12.0)
            kind = "human_range"
        else:
            conf = 0.2
            kind = "too_hot_for_a_person"

        detections.append({
            "type": "thermal_hotspot",
            "conf": round(max(0.05, conf), 3),
            "temp_c": round(peak, 1),
            "mean_temp_c": round(mean, 1),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "pixels": int(len(ys)),
            "classification": kind,
        })

    return sorted(detections, key=lambda d: d["conf"], reverse=True)


def load_thermal_yolo(weights, device=None):
    """Loader for a YOLO model trained on 3-channel thermal (FLIR ADAS / LLVIP)."""
    path = pathlib.Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"thermal YOLO weights not found: {path}")

    def _load():
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("thermal YOLO needs ultralytics: pip install ultralytics")
        model = YOLO(str(path))
        if device:
            model.to(device)
        return model

    return _load


def detect_yolo(image_path, weights, conf=0.25, device=None):
    """Shape-based thermal detection. Same GPU guard as the RGB path."""
    from src.detect.rgb_yolo import to_detections

    path = pathlib.Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"thermal image not found: {path}")
    with gpu_model("thermal_yolo", load_thermal_yolo(weights, device)) as model:
        results = model(str(path), conf=conf, verbose=False)
        return to_detections(results, label="thermal_hotspot")


def _cli():
    ap = argparse.ArgumentParser(description="Thermal hotspot detection.")
    ap.add_argument("--npy", required=True, help=".npy radiometric frame in Celsius")
    ap.add_argument("--percentile", type=float, default=PERCENTILE)
    args = ap.parse_args()

    path = pathlib.Path(args.npy)
    if not path.exists():
        raise FileNotFoundError(f"thermal frame not found: {path}")

    detections = detect_hotspots(np.load(path), args.percentile)
    print(f"{len(detections)} hotspot(s)")
    for d in detections:
        print(f"  {d['temp_c']:.1f} C  conf={d['conf']:.2f}  {d['classification']}  "
              f"bbox={d['bbox']}")


if __name__ == "__main__":
    _cli()
