# The Sentinel Grid — Disaster Response Intelligence Pipeline

**IBM BOB National Hackathon 2026 · Problem Statement 1: AI-Powered Disaster Early
Warning & Rescue Intelligence Platform**

A district faces a monsoon over the next 24–48 hours. Heavy rainfall, urban flooding,
river overflow, landslides, roads and hospitals cut off. This turns the heterogeneous
data emergency teams already receive — rainfall grids, river gauges, satellite imagery,
terrain, population, infrastructure — into four answers a district operations director
can act on.

The working scenario is **Coimbatore, Tamil Nadu**, 30 municipal wards along the
**Noyyal**, at a river stage of 4.6 m against a 3.2 m bankfull.

---

## 1. What PS-1 asks, and where it is answered

| The question | How it is answered | Code |
|---|---|---|
| **Where is the disaster likely to occur?** | hazard `X` per ward from rainfall and terrain retention, then RED / BLUE / GREEN zoning with uncertainty | `src/hazard/` |
| **Who and what is likely to be affected?** | population per ward, weighted critical assets, isolation factor, submerged chokepoints | `src/ingest/level3_infra.py`, `src/scenario.py` |
| **Which locations should teams respond to first?** | expected lives saved, `Π = N · P_alive · S(t + ETA) · κ`, then a greedy dispatch under a team-hour budget | `src/priority/` |
| **What is the safest and fastest evacuation route?** | risk-weighted Dijkstra, then capacity-aware assignment to high-ground relief centres | `src/routing/`, `backend/main.py` |

All four are exported as a single NDRF-format operational order at
`GET /api/export/incident_action_plan`.

---

## 2. Quick start

```bash
git clone https://github.com/advaithsarva/ibm-hackathon
cd ibm-hackathon
pip install -r requirements.txt
python run_demo.py                    # API + dashboard on http://localhost:8000
```

The engine needs three dependencies: PyYAML, numpy and networkx. Nothing reaches the
network after install, which is deliberate — the demo is built to run with the cable out.

To serve the dashboard from the same origin, drop the
[sentinel-grid-frontend](https://github.com/karthikeyachalla/sentinel-grid-frontend)
files into `frontend/`. `backend/main.py` mounts that directory automatically, so the UI
appears at `/` with no configuration and no CORS.

---

## 3. The idea

Every ward carries three numbers: hazard `X`, epistemic uncertainty `U`, and green-zone
suitability `G`.

| Zone | Meaning | Action |
|---|---|---|
| **RED** | known danger | dispatch rescue |
| **BLUE** | we don't know enough to call it safe | send a drone here next |
| **GREEN** | safe, reachable, has shelter capacity | evacuate here |

```
RED    if  X >= 0.75  OR  (X >= 0.40 AND U >= 0.50)
GREEN  if  X < 0.40   AND U < 0.35  AND G >= 0.60  AND reachable
BLUE   otherwise
```

**BLUE is the contribution.** Most risk maps bury uncertainty in a confidence interval
nobody acts on. Here it is a zone class with a queue position. `U` is a noisy-OR over
three failure modes, so any one is enough to raise it and they compound rather than
average:

```
U = 1 − (1 − u_model)(1 − u_stale)(1 − u_cover)
```

A ward turns blue when the models disagree, when the satellite pass is eleven hours old,
or when the sensors only covered part of it — and the API says which, per ward, so the
dashboard can show the operator *why*. Wards change class only after crossing a threshold
by 0.05 and holding two cycles, so a ward on a boundary does not flicker.

### What the Coimbatore scenario produces

At the current storm state: **23 RED, 6 BLUE, 1 GREEN.** Only Annur — highest, furthest
out, and reporting from the ground — is confidently safe. Six high-ground wards are BLUE
because their only observation is an 11-hour-old satellite pass.

Move the sliders and the model genuinely re-runs:

| River stage | Rainfall | Result |
|---|---|---|
| 2.0 m (receding) | ×0.4 | 7 RED, 16 BLUE, 7 GREEN, no evacuation needed |
| 4.6 m (current) | ×1.0 | 23 RED, 6 BLUE, 1 GREEN, 27,581 evacuated, no overflow |
| 5.8 m (worsening) | ×1.4 | 24 RED, 6 BLUE, 0 GREEN, shelters saturate, 6,260 unplaced |

---

## 4. Pipeline

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
        [ROUTE OPTIMIZER]                     risk-weighted Dijkstra + capacity assignment
                          │
                          ▼
        [RESCUE PLAN GENERATOR & DASHBOARD]
          ├── Interactive GIS Map
          ├── Alert Dashboard
          └── Incident Action Plan
```

### Two decisions worth defending

**`ETA` sits inside the survivability decay term of `Π`, not outside it.** The ranking
accounts for who is still alive when the team arrives, not only how many are there now.
How much that changes the order depends on the hazard: at flood's τ of 8 hours a larger
distant cluster still wins, while at wildfire's τ of 2 hours the same 90-minute ETA flips
it. `python -m src.priority.expected_lives --demo` prints both cases side by side. The
claim is conditional, and the code says so rather than overselling it.

**Rainfall is near-uniform across a district; flooding is not.** What separates wards is
how much rain *stays*. A ward 20 m above the floodplain sheds it; a ward 1.5 m above
retains it. Feeding raw rainfall into the hazard formula scores every ward identically
and paints the map uniformly red — the same result as having no model at all.

---

## 5. Data

### Real data in the repository

| Source | What it is | Status |
|---|---|---|
| **IMD 0.25° gridded daily rainfall** | the NWDP product PS-1 names first | decoded, orientation-verified |
| **Kaggle earthquake catalogue** | 17,805 events, time / lat / lon / depth / magnitude | runs end to end |
| **GLDAS soil moisture** | 473,256 district-month readings, 2018–2023 | feature set built |
| **IMD subdivision rainfall** | monsoon totals by subdivision back to 1901 | feature set built |
| Kaggle cyclone / landslide | 2,000-row classifier feature tables | feature sets built |
| Kaggle forest fires | 35 states × 3 years | base rates only |
| Lightning strike records | 2.8 GB of parquet | downloaded, kept out of git |

```bash
python -m src.ingest.imd_grid --grd data/raw/imd/rain_ind0.25_26_09_12.grd --top 10
```

```
4964 land cells, mean 6.8 mm, max 141.89 mm at [26.5, 94.0]
45 cells over IMD's 64.5 mm heavy-rain threshold, 4 over 100 mm
```

The `.grd` is headerless little-endian float32, 129 latitudes by 135 longitudes, so
nothing in the file tells you whether your reshape is transposed — and a transposed read
still produces plausible-looking rainfall in entirely the wrong districts.
`verify_orientation()` checks the land mask against geography instead: no land below 8°N,
widest across 20–28°N. It raises rather than mapping one district's rainfall onto another.

### Column mapping

Nothing hardcodes a schema. Each disaster config carries a `dataset.columns` block of
candidate names, and the loader resolves the real header against them — exact match
first, then whole-token, so `Magnitude (Mw)` and `Focal Depth (km)` resolve without
anyone editing code.

Matching is never on bare substrings. Against these files the alias `y` matches `YEAR`,
`x` matches `Proximity_to_Water`, and `lon` matches `Cyclone`; each would silently bind a
column of unrelated numbers to a coordinate and place hazard where nothing happened.

```bash
python -m src.ingest.kaggle_sets --list
python -m src.ingest.kaggle_sets --inspect earthquake
```

---

## 6. The ML layer

```bash
python -m src.features --profile landslide     # what is in it, before spending GPU time
python -m src.features --build-all             # model-ready arrays for every dataset
python -m src.features --baseline flood_season # what a trivial model already scores
```

```python
d = np.load("data/cache/features/flood_season/train.npz")
X, y = d["X"], d["y"]
```

Standardisation is fitted on the training rows only, one-hot columns are left unscaled,
and every split is stratified. Each build writes a `metadata.json` with feature names,
scaler statistics, class balance and each feature's correlation with the target.

### The five feature sets, and what they are for

| Set | Rows | Task | Feeds |
|---|---|---|---|
| `flood_season` | 4,322 | extreme monsoon from June–July rainfall | early warning |
| `drought_anomaly` | 473,256 | soil-moisture deficit vs the district's own distribution | `drought.yaml` SMI input |
| `landslide` | 2,000 | susceptibility from rainfall, slope, saturation | `landslide.yaml` over the FS formula |
| `cyclone` | 2,000 | formation likelihood | `cyclone.yaml` formation prior |
| `earthquake_prior` | 1,715 cells | spatial base rate: count, max magnitude, median depth | `earthquake.yaml` hazard prior |

The earthquake set is a prior, not a classifier. Magnitude is not predictable from
position, and a model claiming otherwise is fitting noise.

### Baselines, and the one task actually worth training

Plain logistic regression, about a second each on CPU:

| Set | Val accuracy | Val F1 | Verdict |
|---|---|---|---|
| `landslide` | **1.0000** | 1.0000 | trivially separable — do not train anything heavier |
| `cyclone` | **0.9975** | 0.9975 | trivially separable |
| `drought_anomaly` | 0.9656 | 0.9123 | learnable, largely solved |
| `flood_season` | 0.7919 | **0.0426** | **not solved. This is the real problem.** |

Three findings, each of which changes what is worth doing:

**The cyclone dataset leaks its own target.** `Pre_existing_Disturbance` is identical to
the `Cyclone` label in all 2,000 rows, so a one-column model scores 100% and learns
nothing. It is excluded, and `check_leakage()` now refuses to build any feature set with
a column correlated above r = 0.99 with the target.

**Two of the sets are synthetic and trivially separable.** A shuffled-label control
collapses to 0.50 and 0.53, so the separation is in the data rather than in a broken
measurement. Spending a night of GPU time on them buys nothing.

**`flood_season` is where a trained model would earn its keep.** 79% accuracy sounds
respectable and is worthless: the base rate is 79.3% and recall is 0.022, so the baseline
is simply predicting "not extreme" for everything. The tool prints a warning saying so
rather than reporting the flattering number. Predicting a severe monsoon from its first
half, on real IMD data going back to 1901, is both unsolved here and exactly what PS-1
asks for.

---

## 7. API

Eight endpoints, all computed by the engine rather than replayed from fixtures.

| Endpoint | Returns |
|---|---|
| `GET /api/zones?disaster=` | 30 wards with `X`, `U`, `G`, `zone`, `zone_reason`, depth, assets |
| `GET /api/detections?disaster=` | per-ward sensor hits with fused `p_alive` |
| `GET /api/priority?disaster=` | ranked dispatch with team, ETA, equipment protocol |
| `GET /api/evacuation?disaster=` | flows, shelter utilisation, overflow, prohibited routes |
| `GET /api/alert/summary?disaster=` | zone counts, population at risk, alert level |
| `GET /api/model/metrics` | model cards, read from measured validation output |
| `POST /api/simulate` | re-runs the engine at operator-set storm conditions |
| `GET /api/export/incident_action_plan` | the four mandates as an NDRF operational order |

Every response joins on `cell_id`, the `"{lat:.4f}_{lon:.4f}"` centroid string, and
carries `ward_name` so the UI never has to join.

`POST /api/simulate` takes `river_stage_m`, `rainfall_scale` and `sar_staleness_h`. It is
a real recomputation: every `X`, `U`, `G` and zone is derived again from those inputs.

---

## 8. What is real and what is not

Stated plainly, because a demo that overclaims loses the room in Q&A.

| Component | Status |
|---|---|
| IMD gridded rainfall | **real**, decoded and orientation-verified |
| Earthquake catalogue, GLDAS, subdivision rainfall | **real**, 17.8k / 473k / 4.3k rows |
| Hazard formulas, zone engine, uncertainty, hysteresis | **real**, implemented from the spec and audited |
| rPPG pulse extraction | **real** POS signal processing; refuses a BPM below 3 dB SNR |
| Thermal hotspot detection | **real** adaptive-percentile blob detection |
| Priority, dispatch, routing, evacuation | **real** algorithms |
| Ward terrain, elevation, population | **modelled** for Coimbatore, labelled in every record |
| Per-ward detection records | **synthetic** pending live feeds; the fused `p_alive` is real |
| YOLO and YAMNet detections | wrappers ready, weights train separately |
| `GET /api/radar/heartbeat` | **simulated.** FINDER-class radar is hardware we do not have |

`configs/disasters/flood.yaml` needs river discharge. With no CWC gauge export, both the
rating coefficient and bankfull `q_max` would be invented, and inventing them put every
real IMD cell in the wrong tier. `configs/disasters/flood_rainfall.yaml` drops discharge
and anchors `X` to IMD's own published rainfall categories, so every number traces to a
documented threshold.

---

## 9. Verification

```bash
python -m src.hazard.test_hazard         # 10 checks
python -m src.detect.test_detect         # 14 checks
python -m src.test_pipeline              # 31 checks
python -m src.test_frontend_contract     # 59 checks
python -m src.audit_math                 # 104 formula checks
python data/mock/check_mocks.py          # fixtures obey the contracts
```

**218 checks, no framework and no fixtures.**

`src.audit_math` is separate from the tests on purpose. The tests check behaviour, so
they pass even when a constant is quietly wrong; the audit recomputes every formula from
the spec text by an independent route and compares — all seven hazard scores, the six
physics models, noisy-OR, the zone rule on its exact boundaries, the log-odds table
against the spec's own printed log values, survivability, expected lives, the BPR edge
cost, the POS projection matrix, and Horn slope against an analytic plane.

`src.test_frontend_contract` locks the field names the dashboard reads. Renaming one
breaks a panel silently — the page still renders and the value shows as undefined — so
these fail first instead.

Two things the audit records rather than smooths over. The landslide formula is not
self-limiting, so the evaluator clamps it. And spec §5.3's claim that thermal + rPPG +
audio "compounds to P > 0.95" reaches **0.938** at our 0.05 prior; the arithmetic is
right and the spec's figure is loose. It holds from a prior near 0.06, and §5.3 defines
the prior per cell rather than as a constant.

---

## 10. Layout

```
configs/
  global.yaml              grid, thresholds, zone rule, uncertainty, fusion table, routing
  disasters/*.yaml         eight configs, one shared engine; only the formula for X changes
data/
  scenarios/               Coimbatore ward model: terrain, population, assets, shelters
  raw/kaggle/              six committed datasets
  mock/                    frozen API contracts
src/
  grid.py                  cell_id conventions
  scenario.py              ward model, river distance, inundation depth
  features.py              cleaning, standardisation, splits, leakage guard, baselines
  audit_math.py            104 formula checks against the spec
  hazard/                  formulas · physics · zones · 10 checks
  ingest/                  sources · imd_grid · level1-3 · sync · kaggle_sets
  detect/                  rppg · thermal · rgb_yolo · audio_yamnet · fusion · loader
  priority/                survivability · expected_lives · dispatch
  routing/                 graph · evacuation
  report/generate.py       incident brief, Claude API with a deterministic fallback
backend/main.py            FastAPI, the eight endpoints
frontend/                  dashboard, served at / when present
run_demo.py                starts the server
```

---

## 11. Running the modules

Every module runs standalone and fails loudly on missing data rather than substituting
zeros. A zero hazard score and an unmeasured cell are opposite claims.

```bash
# the Coimbatore flood path, end to end
python -m src.scenario --build coimbatore_flood -o data/cache/cell_inputs.cbe.json
python -m src.hazard.zones --config configs/disasters/flood_rainfall.yaml \
    --cells data/cache/cell_inputs.cbe.json -o data/cache/zones.cbe.json

# real IMD rainfall
python -m src.ingest.imd_grid --grd data/raw/imd/rain_ind0.25_26_09_12.grd --cells --top 40 \
    -o data/cache/cell_inputs.imd.json

python -m src.hazard.physics               # governing physics, worked examples
python -m src.detect.rppg --demo           # synthetic pulse, no camera
python -m src.detect.rppg --webcam         # live, needs opencv-python
python -m src.detect.fusion --demo         # the p_alive ladder
python -m src.priority.expected_lives --demo
python -m src.routing.graph --demo
python -m src.report.generate --offline
```

---

## 12. Configuration

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

Two blocks there are calibration knobs rather than findings, and both say so in place.
The GMPE attenuation coefficients need refitting per region before any absolute PGA is
trustworthy, and the §4.4 exposure weights have no values in the spec, so they sit at
equal thirds.

---

## 13. Status

- [x] Coimbatore ward scenario, 30 wards on the Noyyal
- [x] Hazard `X`, config-driven across eight disaster configs
- [x] Zone engine: noisy-OR `U`, green suitability `G`, hysteresis, exposure score
- [x] Real IMD gridded rainfall, decoded and orientation-verified
- [x] Ingest: rainfall accumulation, terrain, infrastructure, layer ageing
- [x] Preprocessing: five feature sets, leakage guard, baselines
- [x] Detection: POS rPPG, thermal blobs, RGB-T merge, Bayesian log-odds fusion
- [x] Priority: survivability decay, expected lives, greedy dispatch
- [x] Routing: risk-weighted Dijkstra, capacity-aware evacuation
- [x] Incident Action Plan export, all four mandates
- [x] FastAPI backend, eight endpoints, live simulation
- [x] 218 verification checks
- [ ] Trained model for `flood_season`, the one task the baseline does not solve
- [ ] YOLO and YAMNet weights
- [ ] Real Bhuvan, Sentinel-1 and WorldPop exports replacing the modelled ward terrain
