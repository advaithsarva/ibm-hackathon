"""RGB person detection with YOLO (section 5.1).

Aerial-tuned YOLO11n/s. Roughly 0.6 GB VRAM, so it loads through the sequential guard
in loader.py and never sits resident alongside another GPU model.

Weights are not in the repo. Point --weights at a file your team trained or downloaded;
a missing file raises rather than returning an empty detection list, because "nobody is
here" and "the model never ran" must not look the same to a rescue coordinator.

Standalone:
    python -m src.detect.rgb_yolo --image frame.jpg --weights data/raw/weights/yolo11n.pt
"""
import argparse
import pathlib

from src.detect.loader import gpu_model

DEFAULT_WEIGHTS = "data/raw/weights/yolo11n.pt"
PERSON_CLASS = 0          # COCO index; retrained aerial models may differ
DEFAULT_CONF = 0.25       # low, because a missed survivor costs more than a false box


def load_yolo(weights=DEFAULT_WEIGHTS, device=None):
    """Returns a zero-argument loader for gpu_model(). Raises if anything is missing."""
    path = pathlib.Path(weights)
    if not path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {path}. Pre-download them (section 9.4); the demo "
            f"never fetches at runtime."
        )

    def _load():
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("RGB detection needs ultralytics: pip install ultralytics")
        model = YOLO(str(path))
        if device:
            model.to(device)
        return model

    return _load


def to_detections(results, class_id=PERSON_CLASS, label="rgb_person"):
    """Ultralytics results -> the detection records used in the section 2.2 contract."""
    out = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            if int(box.cls) != class_id:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append({
                "type": label,
                "conf": round(float(box.conf), 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
    return out


def detect(image_path, weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF, device=None):
    """Detect people in one image. Holds the GPU only for the duration of the call."""
    path = pathlib.Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    with gpu_model("rgb_yolo", load_yolo(weights, device)) as model:
        results = model(str(path), conf=conf, verbose=False)
        return to_detections(results)


def _cli():
    ap = argparse.ArgumentParser(description="YOLO person detection on one image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--device", default=None, help="cuda:0, cpu, ...")
    args = ap.parse_args()

    detections = detect(args.image, args.weights, args.conf, args.device)
    print(f"{len(detections)} person detection(s)")
    for d in detections:
        print(f"  conf={d['conf']:.2f}  bbox={d['bbox']}")


if __name__ == "__main__":
    _cli()
