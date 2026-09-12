"""
The Sentinel Grid — FastAPI backend
Serves zone data, priority queue, evacuation routes, survivor detections,
and radar/heartbeat simulation for the dashboard.
"""
import json
import math
import os
import pathlib
import random
import time
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
MOCK = ROOT / "data" / "mock"
FRONTEND = ROOT / "frontend"

# ── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentinel Grid", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_mock(name: str):
    return json.loads((MOCK / name).read_text())


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/zones")
def get_zones(disaster: str = "flood"):
    """Return zone grid for the given disaster type."""
    data = load_mock("zones.json")
    # Patch disaster name for demo
    data["disaster"] = disaster
    return data


@app.get("/api/priority")
def get_priority():
    """Ranked rescue priority queue."""
    return load_mock("priority.json")


@app.get("/api/evacuation")
def get_evacuation():
    """Evacuation flows and shelter capacity."""
    return load_mock("evacuation.json")


@app.get("/api/detections")
def get_detections():
    """Drone and ground sensor survivor detections."""
    return load_mock("detections.json")


@app.get("/api/radar/heartbeat")
def get_radar_heartbeat(cell_id: Optional[str] = None):
    """
    Simulate a Doppler radar bio-signal for a cell.
    Returns 512 samples of a mixed respiration + cardiac signal
    with realistic noise, plus FFT-derived peak frequencies.
    """
    t = [i / 100.0 for i in range(512)]  # 100 Hz, 5.12 s window
    seed = hash(cell_id or "default") % 9999
    rng = random.Random(seed + int(time.time()) // 10)

    # randomise a little per-call so the wave "lives"
    bpm = rng.uniform(65, 110)
    resp_rate = rng.uniform(14, 22)  # breaths per minute
    f_heart = bpm / 60.0
    f_resp = resp_rate / 60.0

    signal = []
    noise_amp = rng.uniform(0.04, 0.12)
    for ti in t:
        s = (0.6 * math.sin(2 * math.pi * f_resp * ti)        # breathing
             + 0.35 * math.sin(2 * math.pi * f_heart * ti)    # heartbeat
             + 0.15 * math.sin(2 * math.pi * (f_heart * 2) * ti)  # 2nd harmonic
             + noise_amp * rng.gauss(0, 1))                    # sensor noise
        signal.append(round(s, 4))

    return {
        "cell_id": cell_id,
        "fs_hz": 100,
        "n_samples": 512,
        "signal": signal,
        "peaks": {
            "respiration_hz": round(f_resp, 3),
            "respiration_rpm": round(resp_rate, 1),
            "cardiac_hz": round(f_heart, 3),
            "bpm": round(bpm, 1),
        },
        "alive": True,
        "confidence": round(rng.uniform(0.78, 0.97), 2),
    }


@app.get("/api/alert/summary")
def get_alert_summary():
    """High-level incident summary card for the dashboard header."""
    zones = load_mock("zones.json")
    cells = zones["grid_cells"]
    red = sum(1 for c in cells if c["zone"] == "RED")
    blue = sum(1 for c in cells if c["zone"] == "BLUE")
    green = sum(1 for c in cells if c["zone"] == "GREEN")
    total_pop = sum(c["pop"] for c in cells if c["zone"] in ("RED", "BLUE"))
    return {
        "disaster": zones["disaster"],
        "generated_at": zones["generated_at"],
        "red_cells": red,
        "blue_cells": blue,
        "green_cells": green,
        "at_risk_population": total_pop,
        "alert_level": "CRITICAL" if red >= 3 else "WARNING" if red >= 1 else "MONITORING",
    }


# ── static files ─────────────────────────────────────────────────────────────
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
