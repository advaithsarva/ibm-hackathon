# Disaster Response Intelligence Pipeline

Hazard scoring, uncertainty-aware zoning, multi-sensor survivor detection, and
capacity-aware evacuation routing — in one dashboard that runs with the network unplugged.

Build spec: **[`DISASTER_PIPELINE_SPEC.md`](DISASTER_PIPELINE_SPEC.md)** — read §0 and §2 first.

---

## The idea

Every cell on a 100 m grid carries three numbers — hazard `X`, epistemic uncertainty `U`, and
green-zone suitability `G`. Those produce three zones:

| Zone | Meaning | Action |
|---|---|---|
| 🔴 **RED** | known danger | dispatch rescue |
| 🔵 **BLUE** | *we don't know enough to call it safe* | send a drone here next |
| 🟢 **GREEN** | safe, reachable, has shelter capacity | evacuate here |

**BLUE is the contribution.** Most risk maps bury uncertainty in a confidence interval nobody
reads. Here it's a zone class with an owner and a queue position — uncertainty becomes a
dispatchable recon task instead of a footnote.

```
RED    if  X ≥ 0.75  OR  (X ≥ 0.40 AND U ≥ 0.50)
GREEN  if  X < 0.40  AND  U < 0.35  AND  G ≥ 0.60  AND reachable
BLUE   otherwise
```

`U` is a noisy-OR over three failure modes: model disagreement, data staleness, and sensor
coverage gaps. A cell goes blue when SAR is 11 hours stale, not when the hazard is middling.

## Pipeline

```
  ingest ─── level 1  meteorological   rainfall · gauges · CAPE · SST
        ├─── level 2  geospatial       DEM · slope · Sentinel-1 SAR · NDVI
        └─── level 3  infrastructure   OSM roads/hospitals · WorldPop density
                │
                ▼   common grid · common CRS · common timestamp
         hazard engine ──▶  X ∈ [0,1]        physics: Manning · Holland · GMPE · FS · VCI
                │
         zone engine  ──▶  U, G  →  RED / BLUE / GREEN      ← core novelty
                │
         detection    ──▶  P_alive           RGB · thermal · rPPG · acoustic
                │                            fused by Bayesian log-odds
         priority     ──▶  Π = N · P_alive · S(t + ETA) · κ
                │
         routing      ──▶  risk-weighted Dijkstra + min-cost flow
                │
              dashboard
```

Two design decisions worth defending in Q&A:

- **`ETA` sits inside the survivability decay.** So a smaller, reachable cluster correctly
  outranks a larger unreachable one — a real triage call no naive risk map makes.
- **Evacuation is min-cost flow, not shortest path.** Shortest path sends everyone to the
  nearest shelter and overflows it. Flow respects capacity.

## API

Four read-only contracts, frozen at minute 15. Mock JSON lives in `data/mock/` so the
dashboard can be built before any engine exists.

| Endpoint | Returns |
|---|---|
| `GET /api/zones` | grid cells with `X`, `U`, `G`, `zone`, population, assets |
| `GET /api/detections` | per-cell sensor hits, fused `p_alive`, `n_est` |
| `GET /api/priority` | ranked dispatch plan with `pi`, `eta_min`, team, route |
| `GET /api/evacuation` | shelter flow assignment, utilization, overflow |

Cells join on `cell_id` — `"{lat:.4f}_{lon:.4f}"` of the centroid. Everything keys off that.

## Layout

```
configs/
  global.yaml              grid size, CRS, thresholds, weights
  disasters/*.yaml         seven hazards, one shared engine — only the formula for X changes
src/
  ingest/                  level1_meteo · level2_geo · level3_infra · sync
  hazard/                  formulas (config-driven) · physics · zones
  detect/                  rgb_yolo · thermal · rppg · audio_yamnet · fusion
  priority/                survivability · expected_lives · dispatch
  routing/                 graph · evacuation
  report/                  generate.py — Claude API
  api/                     server.py — FastAPI, serves the four contracts
dashboard/                 builds against data/mock/
data/{raw,cache,mock}/     pre-downloaded, gitignored
run_demo.py
```

## Constraints

- **4 GB VRAM**, single machine. Models load sequentially, never concurrently. ≤ 2.5 GB resident.
- **Train nothing.** Inference-only or classical. Where weights don't exist, the physics formula
  runs directly and is labelled a physics-based baseline.
- **Assume the network fails at demo time.** All data pre-downloaded; `run_demo.py` reads only
  from disk.
- Earthquake is the shipping vertical slice. The other six disasters are config over the same engine.

## Run

```bash
pip install -r requirements.txt
python run_demo.py          # offline; reads only from data/
```

## Status

Scaffolding. Spec is frozen; modules land per §10 of the build plan.

- [ ] `data/mock/*.json` — the four contracts
- [ ] hazard `X` + zone engine with hysteresis
- [ ] detection stack + log-odds fusion
- [ ] priority ranker + risk-aware routing
- [ ] dashboard
