"""Acoustic distress detection with YAMNet (section 5.1).

YAMNet is a 4 MB AudioSet classifier over 521 classes. It runs on CPU, so it costs
nothing against the VRAM budget and can run while a GPU model is loaded.

We keep the handful of classes that mean a person is alive and calling out. Speech is
included but weighted down by the caller: a radio left playing is speech too.

Standalone:
    python -m src.detect.audio_yamnet --wav clip.wav --model data/raw/weights/yamnet
"""
import argparse
import pathlib

import numpy as np

SAMPLE_RATE = 16000       # YAMNet is fixed at 16 kHz mono

# AudioSet display names -> how strongly each implies a living person nearby.
DISTRESS_CLASSES = {
    "Screaming": 1.0,
    "Shout": 0.95,
    "Yell": 0.9,
    "Crying, sobbing": 0.9,
    "Baby cry, infant cry": 0.95,
    "Whimper": 0.8,
    "Groan": 0.75,
    "Cough": 0.7,
    "Knock": 0.6,          # tapping on debris is the classic entrapment signal
    "Speech": 0.5,
}
DEFAULT_MODEL = "data/raw/weights/yamnet"
DEFAULT_THRESHOLD = 0.3


def load_yamnet(model_dir=DEFAULT_MODEL):
    """Load a pre-downloaded YAMNet SavedModel plus its class map.

    A local directory, not a TF Hub URL: the demo assumes the network is gone.
    """
    path = pathlib.Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"YAMNet model not found: {path}. Download it before the demo (section 9.4)."
        )
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("acoustic detection needs tensorflow: pip install tensorflow")

    model = tf.saved_model.load(str(path))
    class_map = path / "assets" / "yamnet_class_map.csv"
    if not class_map.exists():
        raise FileNotFoundError(f"YAMNet class map not found: {class_map}")

    names = []
    for i, line in enumerate(class_map.read_text().splitlines()):
        if i == 0 or not line.strip():
            continue                                  # header
        names.append(line.split(",")[-1].strip().strip('"'))
    return model, names


def load_wav(wav_path, sample_rate=SAMPLE_RATE):
    """16 kHz mono float32 in [-1, 1]. Uses the stdlib wave module, no soundfile."""
    import wave

    path = pathlib.Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
        if w.getframerate() != sample_rate:
            raise ValueError(
                f"{path}: YAMNet needs {sample_rate} Hz, file is {w.getframerate()} Hz. "
                f"Resample it during the download phase, not on stage."
            )
    return audio


def to_detections(scores, class_names, threshold=DEFAULT_THRESHOLD):
    """YAMNet frame scores -> acoustic_distress records.

    Scores are per 0.48 s frame; we take the max over the clip per class, because a
    single scream in ten seconds is the whole point.
    """
    peak = np.asarray(scores).max(axis=0)
    detections = []
    for name, weight in DISTRESS_CLASSES.items():
        if name not in class_names:
            continue
        score = float(peak[class_names.index(name)])
        if score >= threshold:
            detections.append({
                "type": "acoustic_distress",
                "conf": round(score * weight, 3),
                "class": name,
                "raw_score": round(score, 3),
            })
    return sorted(detections, key=lambda d: d["conf"], reverse=True)


def detect(wav_path, model_dir=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD):
    audio = load_wav(wav_path)
    model, class_names = load_yamnet(model_dir)
    scores, _embeddings, _spectrogram = model(audio)
    return to_detections(scores.numpy(), class_names, threshold)


def _cli():
    ap = argparse.ArgumentParser(description="YAMNet acoustic distress detection.")
    ap.add_argument("--wav", required=True, help="16 kHz mono 16-bit PCM")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    detections = detect(args.wav, args.model, args.threshold)
    if not detections:
        print("no distress audio above threshold")
    for d in detections:
        print(f"  {d['class']:<20} conf={d['conf']:.2f}  raw={d['raw_score']:.2f}")


if __name__ == "__main__":
    _cli()
