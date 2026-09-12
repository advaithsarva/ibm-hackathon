# Disaster Response Intelligence Pipeline

**IBM Hackathon — Problem Statement 1: AI-Powered Disaster Early Warning & Rescue
Intelligence Platform**

A district faces a monsoon over the next 24–48 hours. Heavy rainfall, urban flooding,
river overflow, landslides, roads and hospitals cut off. This turns the heterogeneous data
emergency teams already receive — rainfall grids, river gauges, satellite imagery,
terrain, population, infrastructure — into four answers a district operations director can
act on.

Build spec: [`DISASTER_PIPELINE_SPEC.md`](DISASTER_PIPELINE_SPEC.md).

## What PS-1 asks, and where it is answered

| The question | Answered by | Code |
|---|---|---|
| Where is the disaster likely to occur? | hazard score `X` per 100 m cell, then RED/BLUE/GREEN zoning | `src/hazard/` |
| Who and what is likely to be affected? | population and weighted critical assets per cell, plus an isolation factor | `src/ingest/level3_infra.py` |
| Which locations should teams respond to first? | expected lives saved, `Π = N · P_alive · S(t + ETA) · κ`, then greedy dispatch under a team-hour budget | `src/priority/` |
| What is the safest and fastest evacuation route? | risk-weighted Dijkstra, then capacity-aware min-cost flow to shelters | `src/routing/` |

## Quick start

```bash
git clone https://github.com/advaithsarva/ibm-hackathon
cd ibm-hackathon
pip install -r requirements.txt
python run_demo.py                    # API on http://localhost:8000
```

Two dependencies carry the whole engine: PyYAML and numpy. Nothing touches the network
after the install.

## The idea

Every cell on a 100 m grid carries three numbers: hazard `X`, epistemic uncertainty `U`,
and green-zone suitability `G`. Those produce three zones.

| Zone | Meaning | Action |
|---|---|---|
| RED | known danger | dispatch rescue |
| BLUE | we don't know enough to call it safe | send a drone here next |
| GREEN | safe, reachable, has shelter capacity | evacuate here |

```
RED    if  X >= 0.75  OR  (X >= 0.40 AND U >= 0.50)
GREEN  if  X < 0.40   AND U < 0.35  AND G >= 0.60  AND reachable
BLUE   otherwise
```

BLUE is the contribution. Most risk maps bury uncertainty in a confidence interval nobody
acts on — here it is a zone class with a queue position. `U` is a noisy-OR over three
failure modes, so any one is enough to raise it and they compound rather than average:

```
U = 1 − (1 − u_model)(1 − u_stale)(1 − u_cover)
```

A cell turns blue when the models disagree, when the satellite pass is eleven hours old,
or when the sensors only covered part of it — and the API says which, per cell. Cells
change class only after crossing a threshold by 0.05 and holding two cycles, so a cell
sitting on a boundary does not flicker.

## Pipeline

```
 LEVEL 1  ATMOSPHERIC & METEOROLOGICAL    IMD rainfall · CWC gauges · GloFAS · CAPE · SST
 LEVEL 2  GEOSPATIAL & TERRAIN            Bhuvan DEM/slope · Sentinel-1/2 · SRTM · NDVI
 LEVEL 3  INFRASTRUCTURE & SOCIO-ECONOMIC OSM roads/bridges/hospitals · WorldPop · HRSL
                          │
                          ▼
        [CLEANING & SYNCHRONIZATION]          common grid · common CRS · common timestamp
                          │
                          ▼
        [PREPROCESSING & FEATURE ENGINEERING]
                          │
                          ▼
        [ML INFERENCE ENGINE]                 physics: Manning · Holland · GMPE · FS · VCI
                          │
                          ▼
        [NORMALIZED THRESHOLD MAPPING]        X ∈ [0,1]
                          │
                          ▼
        [ZONE ENGINE]                         U, G  →  RED / BLUE / GREEN
                          │
                          ▼
        [RISK & IMPACT MAP GENERATION]
          ├── Affected Areas
          ├── Critical Infrastructure Exposure
          └── Population Risk & Rescue Priority Score
                          │
                          ▼
        [DETECTION & VITALS FUSION]           RGB · thermal · rPPG · acoustic
                          │                   → P_alive by Bayesian log-odds
                          ▼
        [EVACUATION PRIORITY RANKER]          Π = N · P_alive · S(t + ETA) · κ
                          │
                          ▼
        [ROUTE OPTIMIZER]                     risk-weighted Dijkstra + min-cost flow
                          │
                          ▼
        [RESCUE PLAN GENERATOR & DASHBOARD]
          ├── Interactive GIS Map
          ├── Alert Dashboard
          └── Report Generator
```

Two decisions in there are worth defending.

**`ETA` sits inside the survivability decay term of `Π`, not outside it.** The ranking
accounts for who is still alive when the team arrives, not only how many are there now.
How much that changes the order depends on the disaster: at flood's τ of 8 hours a larger
distant cluster still wins, while at wildfire's τ of 2 hours the same 90-minute ETA flips
it. `python -m src.priority.expected_lives --demo` prints both cases side by side.

**Evacuation is min-cost flow rather than shortest path.** Shortest path sends everyone to
the nearest shelter and overflows it. The flow formulation holds each shelter to capacity,
spills the remainder to the next, and reports whoever could not be placed instead of
quietly dropping them.

## Real data

`src/ingest/imd_grid.py` reads IMD's 0.25° gridded daily rainfall, the National Water Data
Portal product PS-1 names first.

```bash
python -m src.ingest.imd_grid --grd data/raw/imd/rain_ind0.25_26_09_12.grd --top 10
```

```
4964 land cells, mean 6.8 mm, max 141.89 mm at [26.5, 94.0]
45 cells over IMD's 64.5 mm heavy-rain threshold, 4 over 100 mm
```

The `.grd` is headerless little-endian float32, 129 latitudes by 135 longitudes, so
nothing in the file tells you whether your reshape is transposed — and a transposed read
still produces plausible-looking rainfall in entirely the wrong places.
`verify_orientation()` checks the land mask against geography instead: no land below 8°N,
widest across 20–28°N. It raises rather than mapping one district's rainfall onto another.

Straight into the engine:

```bash
python -m src.ingest.imd_grid --grd data/raw/imd/rain_ind0.25_26_09_12.grd \
    --cells --top 40 -o data/cache/cell_inputs.imd.json
python -m src.hazard.zones --config configs/disasters/flood_rainfall.yaml \
    --cells data/cache/cell_inputs.imd.json -o data/cache/zones.imd.json
```

The 141.9 mm cell scores `X = 0.89`, RED.

## The seven Kaggle datasets

One per hazard config, registered in `src/ingest/kaggle_sets.py`:

```bash
python -m src.ingest.kaggle_sets --list                 # what is downloaded
python -m src.ingest.kaggle_sets --inspect earthquake   # what columns the file really has
python -m src.ingest.kaggle_sets --load earthquake -o data/cache/cell_inputs.eq.json
python -m src.hazard.zones --config configs/disasters/earthquake.yaml \
    --cells data/cache/cell_inputs.eq.json -o data/cache/zones.eq.json
```

Nothing hardcodes a schema. Each disaster config carries a `dataset.columns` block of
candidate column names, and the loader resolves the real header against them — exact
match first, then whole-token, so `Magnitude (Mw)` and `Focal Depth (km)` resolve without
anyone editing code. Matching is never on bare substrings: against these files the alias
`y` matches `YEAR`, `x` matches `Proximity_to_Water` and `lon` matches `Cyclone`, each of
which would silently bind unrelated numbers to a coordinate. `--inspect` prints the header, says which aliases matched, and names
the ones that did not, so adapting to a new file is a YAML edit.

Two mappings are deliberate rather than obvious:

**Earthquake catalogues carry magnitude and depth, never PGA.** Rather than demanding a
column that cannot exist, the loader derives it through the GMPE in
`src/hazard/physics.py`. Epicentral distance is taken as zero, so the figure is the
shaking directly above the hypocentre — the right number for a worst-case impact zone,
the wrong one for anywhere further out.

**The rainfall dataset maps to `flood_rainfall`, not `flood`.** `flood.yaml` needs river
discharge, this dataset has none, and deriving discharge from rainfall through an invented
rating coefficient is exactly the guesswork `flood_rainfall` exists to avoid.

Four of the seven are historical event catalogues rather than current conditions. They are
the right input for base rates, validation and training, and the wrong input for "what is
happening right now" — that still needs a live feed.

## Preprocessing for the ML layer

`src/features.py` turns a raw dataset into arrays a model loads in two lines:

```bash
python -m src.features --profile landslide    # what is in it, before spending GPU time
python -m src.features --build-all            # write model-ready arrays
python -m src.features --baseline landslide   # what a trivial model already scores
```

```python
d = np.load("data/cache/features/landslide/train.npz")
X, y = d["X"], d["y"]
```

Standardisation is fitted on the training rows only, one-hot columns are left unscaled,
and the split is stratified. Each build writes a `metadata.json` carrying the feature
names, the scaler statistics, the class balance and each feature's point-biserial
correlation with the target.

Three findings came out of running it, and all three change what is worth training:

**The cyclone file leaks its own target.** `Pre_existing_Disturbance` is identical to the
`Cyclone` label in all 2000 rows, so a one-column model scores 100% and learns nothing.
It is excluded, and `check_leakage()` now refuses to build any feature set containing a
column correlated with the target above r = 0.99.

**Both feature tables are near-trivially separable.** Plain logistic regression, one
second on a CPU, scores **1.0000** on landslide and **0.9975** on cyclone. A shuffled-label
control collapses to 0.50 and 0.53, so the baseline is sound and the data really is that
easy. Training anything heavier on these two overnight buys nothing — the GPU time
belongs on flood segmentation, which uses real Sen1Floods11 imagery.

**478 of 2000 landslide rows have every soil indicator at zero.** That is a dropped
reference category, not missing data. Imputing it or adding a fourth column reintroduces
the dummy trap.

## API

Four contracts frozen early so the dashboard could be built before any engine existed,
plus two the alert panel adds. Responses come from `data/mock/` until engine output
replaces them; the shapes are identical either way.

| Endpoint | Returns |
|---|---|
| `GET /api/zones` | grid cells with `X`, `U`, `G`, `zone`, `zone_reason`, population, assets |
| `GET /api/detections` | per-cell sensor hits, fused `p_alive`, `n_est` |
| `GET /api/priority` | ranked dispatch plan with `pi`, `eta_min`, team, route |
| `GET /api/evacuation` | shelter flow assignment, utilization, overflow |
| `GET /api/alert/summary` | zone counts, at-risk population, alert level |
| `GET /api/radar/heartbeat` | Doppler bio-signal waveform and peaks |

Every response joins on `cell_id`, the `"{lat:.4f}_{lon:.4f}"` centroid string.

## What is real and what is not

Worth stating plainly, because a demo that overclaims loses the room in Q&A.

| Component | Status |
|---|---|
| IMD gridded rainfall | real data, decoded and orientation-verified |
| Hazard formulas, zone engine, uncertainty, hysteresis | real, implemented from the spec |
| rPPG pulse extraction | real POS signal processing; refuses to report a BPM below 3 dB SNR |
| Thermal hotspot detection | real adaptive-percentile blob detection |
| Priority, dispatch, routing, evacuation flow | real algorithms, currently on synthetic inputs |
| River discharge, terrain elevation, population join | modelled stand-ins, labelled in every record |
| YOLO and YAMNet detections | wrappers ready, weights train separately |
| `GET /api/radar/heartbeat` | **simulated.** FINDER-class radar is hardware we do not have |

`configs/disasters/flood.yaml` needs river discharge. With no CWC gauge export, both the
rating coefficient and bankfull `q_max` would be invented, and inventing them put every
real IMD cell in the wrong tier. `configs/disasters/flood_rainfall.yaml` drops discharge
and anchors `X` to IMD's own published rainfall categories instead, so every number traces
to a documented threshold.

## Layout

```
configs/
  global.yaml              grid, thresholds, zone rule, uncertainty, fusion table, routing
  disasters/*.yaml         eight configs, one shared engine; only the formula for X changes
src/
  grid.py                  cell_id conventions
  hazard/
    formulas.py            config-driven X evaluator
    physics.py             Manning · Holland · GMPE · factor of safety · VCI · flash rate
    zones.py               U, G, RED/BLUE/GREEN, hysteresis, exposure score
    test_hazard.py         10 checks
  ingest/
    sources.py             registry of every PS-1 dataset, offline-first fetcher
    imd_grid.py            IMD 0.25° .grd reader with an orientation guard
    level1_meteo.py        rainfall accumulation, gauge levels
    level2_geo.py          slope, inundation depth, SAR water mask
    level3_infra.py        OSM asset weighting, population binning, isolation
    sync.py                regrid, age layers, emit cell inputs
  detect/
    rppg.py                POS pulse extraction, pure numpy, CPU
    thermal.py             adaptive-percentile hotspot blobs
    rgb_yolo.py            YOLO person detection
    audio_yamnet.py        YAMNet acoustic distress
    fusion.py              RGB-T late NMS merge, Bayesian log-odds
    loader.py              sequential GPU model guard
    test_detect.py         14 checks
  priority/
    survivability.py       S(t) = exp(−t/τ)
    expected_lives.py      Π = N · P_alive · S(t+ETA) · κ
    dispatch.py            greedy knapsack under a team-hour budget
  routing/
    graph.py               risk-weighted edges, Dijkstra, reachability
    evacuation.py          min-cost flow to shelters
  report/generate.py       incident brief, Claude API with a deterministic fallback
  test_pipeline.py         21 checks
backend/main.py            FastAPI, serves the six endpoints
data/
  mock/                    the frozen contracts, plus generated cell inputs
  raw/, cache/             downloaded inputs and weights, gitignored
run_demo.py                starts the API server
```

The dashboard is the remaining gap. `backend/main.py` already mounts `frontend/` as static
files, so dropping an `index.html` there serves it at the root with no further wiring.

## Running the modules

Every module runs standalone and fails loudly on missing data rather than substituting
zeros. A zero hazard score and an unmeasured cell are opposite claims.

```bash
# hazard score for one cell
python -m src.hazard.formulas --config configs/disasters/earthquake.yaml --pga_ms2 3.4

# the flood path end to end
python -m src.ingest.sync --demo --config configs/disasters/flood.yaml \
    -o data/mock/cell_inputs.flood.json
python -m src.hazard.zones --config configs/disasters/flood.yaml \
    --cells data/mock/cell_inputs.flood.json -o data/cache/zones.flood.json

python -m src.ingest.sources --list        # which datasets are present
python -m src.hazard.physics               # governing physics, worked examples
python -m src.detect.rppg --demo           # synthetic pulse, no camera
python -m src.detect.rppg --webcam         # live, needs opencv-python
python -m src.detect.fusion --demo         # the p_alive ladder
python -m src.priority.expected_lives --demo
python -m src.priority.dispatch --demo
python -m src.routing.graph --demo
python -m src.routing.evacuation --demo
python -m src.report.generate --offline
```

`data/mock/zones.json` is generated by the zone engine from `cell_inputs.earthquake.json`
rather than hand-written, so the fixture cannot drift from the code. Regenerate and diff.

## Configuration

Adding a disaster is a YAML file, not a code change:

```yaml
name: earthquake
hazard:
  formula: "min(1.0, pga_ms2 / 4.0)"
  inputs: [pga_ms2]
  tau_survivability_hours: 40
uncertainty:
  tau_data_hours: 1.0
```

`configs/global.yaml` holds the shared values: the zone rule, noisy-OR parameters,
green-suitability weights, log-odds likelihood ratios and routing penalties.

Two blocks there are calibration knobs rather than findings, and both say so in place. The
GMPE attenuation coefficients need refitting per region before any absolute PGA is
trustworthy, and the §4.4 exposure weights have no values in the spec, so they sit at
equal thirds.

## Testing

```bash
python -m src.hazard.test_hazard     # 10 checks
python -m src.detect.test_detect     # 14 checks
python -m src.test_pipeline          # 25 checks
python -m src.audit_math             # 104 formula checks
python data/mock/check_mocks.py      # fixtures obey the contracts
```

`src.audit_math` is separate from the tests on purpose. The tests check behaviour, so
they pass even if a constant is quietly wrong; the audit recomputes every formula from
the spec text by an independent route and compares — all seven hazard scores, the six
physics models, noisy-OR, the zone rule on its exact boundaries, the green-suitability
weights, the log-odds table against the spec's own printed log values, survivability,
expected lives, the BPR edge cost, the POS projection matrix, and Horn slope against an
analytic plane.

153 checks in total, no framework and no fixtures. They test behaviour rather than restating the
code: that a pulse buried at the noise floor is refused rather than guessed, that a road
through a cell at `X >= 0.75` is never routed, that shelter capacity is never exceeded,
that a cell with no route is reported instead of dropped, and that the IMD grid raises if
its orientation looks wrong.

## Constraints

- Models load sequentially, never concurrently, so the stack runs on a single GPU.
  `src/detect/loader.py` raises rather than letting a second model load.
- Train nothing at build time. Everything is inference-only or classical. Where weights do
  not exist the physics formula runs directly, labelled as a physics-based baseline.
- Assume the network fails at demo time. All data is pre-downloaded to `data/raw/` and
  nothing is fetched at runtime.
- Model weights are not in the repository. Put them under `data/raw/weights/`; the YOLO
  and YAMNet wrappers raise a named error when a file is missing, so a model that never
  ran is never mistaken for a cell with nobody in it.

## Status

- [x] Four frozen API contracts with mock fixtures
- [x] Hazard `X`, config-driven across eight disaster configs
- [x] Zone engine: noisy-OR `U`, green suitability `G`, hysteresis, exposure score
- [x] Real IMD gridded rainfall, decoded and orientation-verified
- [x] Ingest: rainfall accumulation, terrain, infrastructure, layer ageing
- [x] Detection: POS rPPG, thermal blobs, RGB-T merge, Bayesian log-odds fusion
- [x] Priority: survivability decay, expected lives, greedy dispatch
- [x] Routing: risk-weighted Dijkstra, capacity-aware evacuation flow
- [x] Report generator with an offline template
- [x] FastAPI backend serving all six endpoints
- [ ] Dashboard
- [ ] YOLO and YAMNet weights, training on the team's own machines
- [ ] Real Bhuvan, Sentinel-1 and WorldPop exports replacing the modelled stand-ins
