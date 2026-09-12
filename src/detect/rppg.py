"""Remote photoplethysmography by POS (section 5.1).

POS (Plane-Orthogonal-to-Skin, Wang et al. 2017) recovers a pulse waveform from the
colour changes in a patch of skin. Classical signal processing: no weights, no training,
no GPU. That is why it is in the ship list on a 4 GB budget.

This is a triage hint, not a medical reading, and the CLI prints that on every run.

Standalone:
    python -m src.detect.rppg --demo                  synthetic signal, no camera
    python -m src.detect.rppg --webcam --seconds 12   live, needs opencv-python
"""
import argparse

import numpy as np

# Physiological band. 0.7-3.0 Hz is 42-180 bpm, which covers resting through exertion
# without letting a 4 Hz lighting flicker masquerade as a pulse.
MIN_HZ, MAX_HZ = 0.7, 3.0
WINDOW_SEC = 1.6          # POS window length from the paper
MIN_SNR_DB = 3.0          # below this the BPM is noise; report no lock

# POS projection: S = P . Cn, with P = [[0, 1, -1], [-2, 1, 1]].
_PROJECTION = np.array([[0.0, -2.0], [1.0, 1.0], [-1.0, 1.0]])


def pos_pulse(rgb_trace, fps, window_sec=WINDOW_SEC):
    """Spatially averaged RGB per frame, shape (n_frames, 3) -> 1-D pulse signal.

    Overlap-added sliding windows, each temporally normalized then projected onto the
    plane orthogonal to the skin-tone direction, which is what cancels the intensity
    changes caused by motion and lighting.
    """
    rgb = np.asarray(rgb_trace, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb_trace must be (n_frames, 3), got {rgb.shape}")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    n = len(rgb)
    length = int(window_sec * fps)
    if length < 2 or n < length:
        raise ValueError(f"need at least {max(2, length)} frames at {fps} fps, got {n}")

    signal = np.zeros(n)
    for start in range(n - length + 1):
        block = rgb[start:start + length]
        mean = block.mean(axis=0)
        if np.any(mean <= 0):
            continue                                    # dark or clipped patch
        projected = (block / mean) @ _PROJECTION        # (length, 2)
        s1, s2 = projected[:, 0], projected[:, 1]
        std2 = s2.std()
        h = s1 + (s1.std() / std2) * s2 if std2 > 0 else s1
        signal[start:start + length] += h - h.mean()    # overlap-add

    return signal


def bandpass(signal, fps, low_hz=MIN_HZ, high_hz=MAX_HZ):
    """Zero everything outside the physiological band. FFT mask, so no scipy."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    return np.fft.irfft(spectrum, n=len(signal))


def heart_rate(signal, fps, low_hz=MIN_HZ, high_hz=MAX_HZ):
    """Dominant frequency in the band -> (bpm, snr_db).

    SNR compares the power at the peak and its first harmonic against the rest of the
    band. A real pulse concentrates power; noise spreads it.

    Resolution is fps/n_samples Hz, so a 12 s clip at 30 fps resolves to about 5 bpm.
    Capture longer if you need a tighter reading than that.
    """
    if len(signal) < 2:
        raise ValueError("signal too short for a spectrum")

    windowed = signal * np.hanning(len(signal))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)

    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not band.any() or power[band].sum() == 0:
        return 0.0, -np.inf

    peak_hz = freqs[band][np.argmax(power[band])]
    bpm = peak_hz * 60.0

    # Peak plus first harmonic, each with a narrow skirt, count as signal.
    tol = 0.15
    is_signal = np.zeros_like(freqs, dtype=bool)
    for centre in (peak_hz, 2 * peak_hz):
        is_signal |= np.abs(freqs - centre) <= tol
    signal_power = power[band & is_signal].sum()
    noise_power = power[band & ~is_signal].sum()
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    return float(bpm), float(snr_db)


def estimate(rgb_trace, fps, min_snr_db=MIN_SNR_DB):
    """Full chain -> the rppg_pulse detection record for the section 2.2 contract.

    Returns locked=False rather than a confident wrong number when the SNR is too low.
    A refusal is useful to a rescue team; a fabricated heart rate is not.
    """
    pulse = bandpass(pos_pulse(rgb_trace, fps), fps)
    bpm, snr_db = heart_rate(pulse, fps)
    locked = snr_db >= min_snr_db and MIN_HZ * 60 <= bpm <= MAX_HZ * 60

    # Confidence saturates around 12 dB; beyond that the extra SNR tells us little.
    conf = max(0.0, min(1.0, (snr_db - min_snr_db) / 9.0)) if locked else 0.0

    return {
        "type": "rppg_pulse",
        "conf": round(conf, 3),
        "bpm": round(bpm, 1) if locked else None,
        "snr_db": round(snr_db, 2) if np.isfinite(snr_db) else None,
        "locked": bool(locked),
        "caveat": "triage hint, not a medical reading",
    }


def synthetic_trace(bpm=72.0, fps=30.0, seconds=10.0, noise=0.005, seed=0):
    """A skin-tone trace with a known pulse. Used by the tests and the offline demo.

    The default noise is what survives averaging an ROI of a few thousand pixels, which
    is well under the pulse amplitude. Raise it to 0.02 and the pulse sits at the noise
    floor: estimate() then refuses to lock, which is the correct answer, not a bug.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1.0 / fps)
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
    skin = np.array([0.68, 0.52, 0.46])              # mean R,G,B of lit skin
    # Green carries the strongest blood-volume signal, red the weakest.
    gain = np.array([0.3, 1.0, 0.6]) * 0.02
    trace = skin + np.outer(pulse, gain)
    trace += rng.normal(0, noise, trace.shape)       # sensor and motion noise
    return trace


def _webcam_trace(seconds, camera=0):
    """Mean RGB of a centre-face ROI per frame. Needs opencv-python."""
    try:
        import cv2
    except ImportError:
        raise ImportError("webcam capture needs opencv-python: pip install opencv-python")

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        frames, roi = [], None
        for _ in range(int(fps * seconds)):
            ok, frame = cap.read()
            if not ok:
                break
            if roi is None:
                faces = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                 1.3, 5)
                if len(faces):
                    x, y, w, h = faces[0]
                    roi = (y + h // 5, y + h // 2, x + w // 4, x + 3 * w // 4)  # forehead
            if roi is None:
                continue
            y0, y1, x0, x1 = roi
            patch = frame[y0:y1, x0:x1]
            if patch.size:
                b, g, r = patch.reshape(-1, 3).mean(axis=0) / 255.0
                frames.append([r, g, b])
        if not frames:
            raise RuntimeError("no face found; sit facing the camera in even lighting")
        return np.array(frames), fps
    finally:
        cap.release()


def _cli():
    ap = argparse.ArgumentParser(description="POS rPPG pulse estimation.")
    ap.add_argument("--demo", action="store_true", help="synthetic 72 bpm trace")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.webcam:
        trace, fps = _webcam_trace(args.seconds, args.camera)
    elif args.demo:
        trace, fps = synthetic_trace(72.0, args.fps, args.seconds), args.fps
    else:
        ap.error("pass --demo or --webcam")

    result = estimate(trace, fps)
    if result["locked"]:
        print(f"VITALS: {result['bpm']:.1f} bpm  snr={result['snr_db']} dB  "
              f"conf={result['conf']}")
    else:
        print(f"no pulse lock (snr={result['snr_db']} dB); "
              f"hold still, improve lighting, or lengthen the capture")
    print(result["caveat"])


if __name__ == "__main__":
    _cli()
