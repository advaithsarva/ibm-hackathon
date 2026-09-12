# Disaster Response Intelligence Pipeline

Hazard scoring, uncertainty-aware zoning, multi-sensor survivor detection, and
capacity-aware evacuation routing, in one dashboard that runs with the network unplugged.

Build spec: [`DISASTER_PIPELINE_SPEC.md`](DISASTER_PIPELINE_SPEC.md). Read §0 and §2 first.

## The idea

Every cell on a 100 m grid carries three numbers: hazard `X`, epistemic uncertainty `U`, and
green-zone suitability `G`. Those produce three zones.

| Zone | Meaning | Action |
|---|---|---|
| RED | known danger | dispatch rescue |
| BLUE | we don't know enough to call it safe | send a drone here next |
| GREEN | safe, reachable, has shelter capacity | evacuate here |

```
RED    if  X ≥ 0.75  OR  (X ≥ 0.40 AND U ≥ 0.50)
GREEN  if  X < 0.40  AND  U < 0.35  AND  G ≥ 0.60  AND reachable
BLUE   otherwise
```

BLUE is what the rest of the system is built around. Most risk maps put uncertainty in a
confidence interval nobody acts on — here it's a zone class with a queue position. A cell
turns blue when the SAR revisit is 11 hours stale, or when the sensors only covered part of
it, and that cell goes to the top of the recon list.

`U` is a noisy-OR over those three failure modes: model disagreement, data staleness, and
sensor coverage gaps.

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

Two decisions in there are worth knowing about before you read the code.

`ETA` sits inside the survivability decay term of `Π`, not outside it. A cluster of six people
18 minutes away outranks a larger cluster 90 minutes away, because the ranking accounts for who
is still alive when the team arrives.

Evacuation is min-cost flow rather than shortest path. Shortest path sends everyone to the
nearest shelter and overflows it; the flow formulation holds each shelter to its capacity and
spills the remainder to the next one.

## API

Four read-only contracts, frozen at minute 15. Mock JSON lives in `data/mock/` so the
dashboard can be built before any engine exists.

| Endpoint | Returns |
|---|---|
| `GET /api/zones` | grid cells with `X`, `U`, `G`, `zone`, population, assets |
| `GET /api/detections` | per-cell sensor hits, fused `p_alive`, `n_est` |
| `GET /api/priority` | ranked dispatch plan with `pi`, `eta_min`, team, route |
| `GET /api/evacuation` | shelter flow assignment, utilization, overflow |

Every response joins on `cell_id`, the `"{lat:.4f}_{lon:.4f}"` centroid string.

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

- 4 GB VRAM on a single machine. Models load sequentially, never concurrently, and stay under
  2.5 GB resident.
- Train nothing. Everything is inference-only or classical. Where weights don't exist the
  physics formula runs directly, labelled as a physics-based baseline.
- Assume the network fails at demo time. All data is pre-downloaded and `run_demo.py` reads
  only from disk.
- Earthquake is the shipping vertical slice. The other six disasters are config over the same
  engine.

## Run

```bash
pip install -r requirements.txt

# hazard score for one cell
python -m src.hazard.formulas --config configs/disasters/earthquake.yaml --pga_ms2 3.4

# zone engine over a grid, writes the /api/zones contract
python -m src.hazard.zones --config configs/disasters/earthquake.yaml     --cells data/mock/cell_inputs.earthquake.json -o data/mock/zones.json

python -m src.hazard.test_hazard     # 10 checks, no framework
python data/mock/check_mocks.py      # fixtures obey the contracts
```

Every module runs standalone and fails loudly on missing data rather than
substituting zeros. A zero hazard score and an unmeasured cell are opposite claims.

## Status

Scaffolding. The spec is frozen; modules land per §10 of the build plan.

- [x] `data/mock/*.json`, the four contracts
- [x] hazard `X`, config-driven over all seven disasters
- [x] zone engine: noisy-OR `U`, green suitability `G`, hysteresis, exposure score
- [ ] detection stack and log-odds fusion
- [ ] priority ranker and risk-aware routing
- [ ] dashboard
