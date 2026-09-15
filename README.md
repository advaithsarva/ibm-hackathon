# The Sentinel Grid — Disaster Response Intelligence Pipeline

**IBM BOB National Hackathon 2026 · Problem Statement 1: AI-Powered Disaster Early
Warning & Rescue Intelligence Platform**

A district faces a monsoon over the next 24–48 hours. Heavy rainfall, urban flooding,
river overflow, landslides, roads and hospitals cut off. We built a platform that turned
the heterogeneous data emergency teams already receive — rainfall grids, river gauges,
satellite imagery, terrain, population, infrastructure — into four answers a district
operations director can act on.

The working scenario is **Coimbatore, Tamil Nadu**: 30 municipal wards along the
**Noyyal**, at a river stage of 4.6 m against a 3.2 m bankfull. Seven hazard types run
over the same ward model.

---

## 1. What PS-1 asked, and where we answered it

| The question | How we answered it | Code |
|---|---|---|
| **Where is the disaster likely to occur?** | hazard `X` per ward from rainfall and terrain retention, then RED / BLUE / GREEN zoning that carries uncertainty | `src/hazard/` |
| **Who and what is likely to be affected?** | population per ward, weighted critical assets, isolation factor, submerged chokepoints | `src/ingest/level3_infra.py`, `src/scenario.py` |
| **Which locations should teams respond to first?** | expected lives saved, `Π = N · P_alive · S(t + ETA) · κ`, then a greedy dispatch under a team-hour budget | `src/priority/` |
| **What is the safest and fastest evacuation route?** | risk-weighted Dijkstra, then capacity-aware assignment to high-ground relief centres | `src/routing/`, `backend/main.py` |

All four are exported as one NDRF-format operational order at
`GET /api/export/incident_action_plan`.

---

## 2. Quick start

```bash
git clone https://github.com/advaithsarva/ibm-hackathon
cd ibm-hackathon
pip install -r requirements.txt
python run_demo.py                    # API + dashboard on http://localhost:8000
```

The engine needed three dependencies: PyYAML, numpy and networkx. Nothing reaches the
network after install. The demo was built to run with the cable out, and the dashboard
vendors its own copy of Leaflet for the same reason.

The dashboard lives in `frontend/`. `backend/main.py` mounts that directory
automatically, so the UI appears at `/` with no configuration and no CORS.

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

BLUE is the contribution. Most risk maps bury uncertainty in a confidence interval nobody
acts on; we made it a zone class with a queue position. `U` is a noisy-OR over
three failure modes, so any one of them is enough to raise it, and together they compound:

```
U = 1 − (1 − u_model)(1 − u_stale)(1 − u_cover)
```

A ward turned blue when the models disagreed, when the satellite pass was eleven hours
old, or when the sensors only covered part of it, and the API reports which, per ward,
so the dashboard shows the operator *why*. Wards change class only after crossing a
threshold by 0.05 and holding two cycles, so a ward on a boundary does not flicker.

### What the Coimbatore scenario produced

At the current storm state: **23 RED, 6 BLUE, 1 GREEN.** Only Annur — highest, furthest
out, reporting from the ground — was confidently safe. Six high-ground wards came out
BLUE because their only observation was an 11-hour-old satellite pass.

Moving the sliders re-ran the model rather than replaying a recording:

| River stage | Rainfall | Result |
|---|---|---|
| 2.0 m (receding) | ×0.4 | 7 RED, 16 BLUE, 7 GREEN, no evacuation needed |
| 4.6 m (current) | ×1.0 | 23 RED, 6 BLUE, 1 GREEN, 30,100 evacuated |
| 5.8 m (worsening) | ×1.4 | 24 RED, 6 BLUE, 0 GREEN, shelters saturated, 6,260 unplaced |

### Seven hazards over one ward model

Each hazard derives its own inputs from the same terrain, using the physics already in
`src/hazard/physics.py`. There is no second set of constants.

| Hazard | Driving inputs | Result |
|---|---|---|
| Flood | rainfall × terrain retention + inundation depth | 23 RED, 6 BLUE, 1 GREEN |
| Earthquake | GMPE attenuation from an M6.4 event, vs30 amplification | 27 RED, 3 BLUE |
| Cyclone | Holland wind field from a 962 hPa centre | 26 RED, 4 BLUE |
| Landslide | 3-day rainfall + measured slope on the Ghats margin | 29 RED, 1 BLUE |
| Drought | soil moisture index against drainage | 26 RED, 4 BLUE |
| Wildfire | fire weather index, rising where water does not linger | 12 RED, 18 BLUE |
| Lightning | CAPE nowcast | 30 RED |

Drought and lightning rank by **exposure**, not expected lives saved. Neither has a
survivability constant in the spec, and that is not an oversight to paper over: drought
unfolds over months and lightning is instantaneous, so there is no decay curve to put an
ETA inside. The API reports which ranking it used.

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

### Two decisions a judge will probe

**`ETA` sits inside the survivability decay term of `Π`, not outside it.** The ranking
accounts for who is still alive when the team arrives, not only how many are there now.
How much that changes the order depends on the hazard: at flood's τ of 8 hours a larger
distant cluster still wins, while at wildfire's τ of 2 hours the same 90-minute ETA flips
it. `python -m src.priority.expected_lives --demo` prints both cases. The claim is
conditional, and the code says so.

**Rainfall is near-uniform across a district; flooding is not.** What separates wards is
how much rain *stays*. A ward 20 m above the floodplain sheds it; a ward 1.5 m above
retains it. Our first version fed raw rainfall into the hazard formula, scored every ward
identically, and painted the map uniformly red. That is the same result as having no
model at all.

---

## 5. Data

### Real data in the repository

| Source | What it is | Status |
|---|---|---|
| **IMD 0.25° gridded daily rainfall** | the NWDP product PS-1 names first | decoded, orientation-verified |
| **Kaggle earthquake catalogue** | 17,805 events: time, lat, lon, depth, magnitude | ran end to end |
| **GLDAS soil moisture** | 473,256 district-month readings, 2018–2023 | feature set built and trained |
| **IMD subdivision rainfall** | monsoon totals by subdivision back to 1901 | feature set built and trained |
| Kaggle cyclone / landslide | 2,000-row classifier feature tables | feature sets built and trained |
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
nothing in the file said whether our reshape was transposed, and a transposed read still
produces plausible-looking rainfall in entirely the wrong districts.
`verify_orientation()` checks the land mask against geography instead: no land below 8°N,
widest across 20–28°N. A transposed read raises there and never reaches the map.

### Column mapping

Nothing hardcodes a schema. Each disaster config carries a `dataset.columns` block of
candidate names, and the loader resolves the real header against them: exact match
first, then whole-token, so `Magnitude (Mw)` and `Focal Depth (km)` resolved without
anyone editing code.

Matching is never on bare substrings. Against these files the alias `y` matches `YEAR`,
`x` matches `Proximity_to_Water`, and `lon` matches `Cyclone`; each would have silently
bound a column of unrelated numbers to a coordinate and placed hazard where nothing
happened.

---

## 6. The models we trained

```bash
python -m src.features --build-all             # model-ready arrays for every dataset
python -m src.features --baseline flood_season # measured, held-out
```

Standardisation was fitted on the training rows only, one-hot columns were left unscaled,
and every split was stratified. Each build wrote a `metadata.json` with feature names,
scaler statistics, class balance and each feature's correlation with the target.

### Trained models and measured validation results

Logistic regression on standardised features, held-out stratified validation:

| Model | Train rows | Val rows | Accuracy | ROC-AUC | F1 | Brier |
|---|---|---|---|---|---|---|
| Landslide susceptibility | 1,600 | 400 | **1.0000** | 1.0000 | 1.0000 | 0.0001 |
| Cyclone formation | 1,600 | 400 | **0.9975** | 1.0000 | 0.9975 | 0.0020 |
| Drought anomaly | 378,605 | 94,651 | **0.9656** | 0.9924 | 0.9123 | 0.0385 |
| Extreme monsoon season | 3,457 | 865 | 0.7919 | 0.6620 | **0.0426** | 0.1590 |

Plus a spatial prior over **1,715 cells** from the earthquake catalogue: event count,
maximum observed magnitude and median depth per 0.5° cell. That one is a prior, not a
classifier: magnitude is not predictable from position, and a model claiming otherwise
would be fitting noise.

Feature importance came out of the trained weights, not from an assumption. On
standardised inputs the coefficients are directly comparable, which is why the scaler
runs first:

- **Extreme monsoon**: July rainfall 55%, early-season total 24%, June rainfall 16%
- **Drought anomaly**: within-district z-score 82%, absolute soil moisture 11%
- **Cyclone**: ocean depth 18%, atmospheric pressure 16%, sea-surface temperature 15%
- **Landslide**: soil saturation 15%, vegetation cover 15%, proximity to water 15%

### Three findings that changed what we built

**The cyclone dataset leaked its own target.** `Pre_existing_Disturbance` was identical
to the `Cyclone` label in all 2,000 rows, so a one-column model scored 100% and learned
nothing. We excluded it and added `check_leakage()`, which now refuses to build any
feature set containing a column correlated above r = 0.99 with its target.

**Two of the datasets were synthetic and trivially separable.** A shuffled-label control
collapsed to 0.50 and 0.53, so the separation is real and the measurement is sound. We
stopped there. Spending a night of GPU time to prove it twice would buy nothing.

**Extreme-monsoon prediction is the one task still open.** 79% accuracy sounds
respectable and is worthless: the base rate is 79.3% and recall is 0.022, so the model is
predicting "not extreme" for everything. ROC-AUC of 0.662 is the honest measure. The
tooling leads with that warning, and the dashboard's model card shows the same sentence. Predicting a severe monsoon from its first half, on
real IMD records going back to 1901, is exactly what PS-1 asks for and is where a
gradient-boosted model would earn its keep.

---

## 7. API

Eight endpoints, all computed by the engine on every request.

| Endpoint | Returns |
|---|---|
| `GET /api/zones?disaster=` | 30 wards with `X`, `U`, `G`, `zone`, `zone_reason`, depth, TWI, assets |
| `GET /api/detections?disaster=` | per-ward sensor hits with fused `p_alive` |
| `GET /api/priority?disaster=` | ranked dispatch with team, ETA, equipment protocol, ranking basis |
| `GET /api/evacuation?disaster=` | flows, shelter utilisation, overflow, prohibited routes |
| `GET /api/alert/summary?disaster=` | zone counts, population at risk, alert level |
| `GET /api/model/metrics` | model cards built from measured validation output |
| `POST /api/simulate` | re-runs the engine at operator-set storm conditions |
| `GET /api/export/incident_action_plan` | the four mandates as an NDRF operational order |

`disaster` accepts all seven hazards. Every response joins on `cell_id`, the
`"{lat:.4f}_{lon:.4f}"` centroid string, and carries `ward_name` so the UI never has to
join.

`POST /api/simulate` takes `river_stage_m`, `rainfall_scale` and `sar_staleness_h` and
recomputes every `X`, `U`, `G` and zone from them.

**Every value the dashboard displays is backed by an API field.** The model card reports
measured accuracy, ROC-AUC, F1, Brier score, a confusion matrix and weight-derived
feature importance. Where a model has no such metric the API returns null and the panel
shows a dash; nothing is filled in with a plausible-looking placeholder.

---

## 8. What is real and what is not

Stated plainly, because a demo that overclaims loses the room in Q&A.

| Component | Status |
|---|---|
| IMD gridded rainfall | **real**, decoded and orientation-verified |
| Earthquake catalogue, GLDAS, subdivision rainfall | **real**: 17.8k / 473k / 4.3k rows |
| Hazard formulas, zone engine, uncertainty, hysteresis | **real**, implemented from the spec and audited |
| Four trained classifiers | **real**, held-out validation, figures in §6 |
| rPPG pulse extraction | **real** POS signal processing; refuses a BPM below 3 dB SNR |
| Thermal hotspot detection | **real** adaptive-percentile blob detection |
| Priority, dispatch, routing, evacuation | **real** algorithms |
| Ward terrain, elevation, population | **modelled** for Coimbatore, labelled in every record |
| Per-ward detection records | **synthetic** pending live feeds; the fused `p_alive` is real |
| YOLO11n person detector | **real**, fine-tuned on PennFudanPed (real photos), see §14 |
| YAMNet acoustic classifier | **real** Google checkpoint; only its small transfer head trains on synthetic audio, see §14 |
| Heart-rate (PPG→BPM) model | **real**, trained on PPG-DaLiA; MAE ≈ 28.7 bpm held-out — a real, imperfect number, see §14 |
| Bio-Radar pulse chart (dashboard) | **real model output** driving the chart parameters; the chart itself is still a visualization, not hardware |
| `GET /api/radar/heartbeat` | **simulated.** FINDER-class radar is hardware we do not have |

`configs/disasters/flood.yaml` needs river discharge. With no CWC gauge export, both the
rating coefficient and bankfull `q_max` would have been invented, and inventing them put
every real IMD cell in the wrong tier. `configs/disasters/flood_rainfall.yaml` drops
discharge and anchors `X` to IMD's own published rainfall categories, so every number
traces to a documented threshold.

---

## 9. Verification

```bash
python -m src.hazard.test_hazard         #  10 checks
python -m src.detect.test_detect         #  14 checks
python -m src.test_pipeline              #  31 checks
python -m src.test_frontend_contract     # 155 checks
python -m src.audit_math                 # 104 formula checks
python data/mock/check_mocks.py          # fixtures obey the contracts
```

**314 checks, no framework and no fixtures.**

`src.audit_math` is separate from the tests on purpose. The tests check behaviour, so
they pass even when a constant is quietly wrong; the audit recomputes every formula from
the spec text by an independent route and compares: all seven hazard scores, the six
physics models, noisy-OR, the zone rule on its exact boundaries, the log-odds table
against the spec's own printed log values, survivability, expected lives, the BPR edge
cost, the POS projection matrix, and Horn slope against an analytic plane.

`src.test_frontend_contract` locks the field names the dashboard reads. Renaming one
breaks a panel silently: the page still renders and the value shows as undefined. These
fail first instead.

Two things the audit records in place. The landslide formula is not
self-limiting, so the evaluator clamps it. And spec §5.3's claim that thermal + rPPG +
audio "compounds to P > 0.95" reaches **0.938** at our 0.05 prior; the arithmetic is
right and the spec's figure is loose. It holds from a prior near 0.06, and §5.3 defines
the prior per cell.

### Modelling errors we caught and fixed

Each produced plausible-looking output, which is why they were hard to spot:

- The Noyyal was drawn through ward centroids, putting riverside wards at 0.00 km from
  the river and making the strongest driver of inundation meaningless.
- Uniform rainfall scored every ward identically and painted the map entirely red.
- The two uncertainty sources were measured on different scales, so `u_model` saturated
  everywhere — disagreement invented by the code rather than found in the data.
- Green suitability penalised wards for being far from the river, which is precisely what
  makes them good shelters.
- Slope was derived as 1.6° per metre of relief, reading a 22 m ward as a 37° cliff.
  Over a 2 km ward it is under 1°, and only the Ghats margin has real slope.

---

## 10. Layout

```
configs/
  global.yaml              grid, thresholds, zone rule, uncertainty, fusion table, routing
  disasters/*.yaml         eight configs, one shared engine; only the formula for X changes
data/
  scenarios/               Coimbatore ward model: terrain, population, assets, shelters, events
  raw/kaggle/              six committed datasets
  mock/                    frozen API contracts
src/
  grid.py                  cell_id conventions
  scenario.py              ward model, river distance, inundation depth, per-hazard inputs
  features.py              cleaning, standardisation, splits, leakage guard, trained baselines
  audit_math.py            104 formula checks against the spec
  hazard/                  formulas · physics · zones · 10 checks
  ingest/                  sources · imd_grid · level1-3 · sync · kaggle_sets
  detect/                  rppg · thermal · rgb_yolo · audio_yamnet · fusion · loader
  priority/                survivability · expected_lives · dispatch
  routing/                 graph · evacuation
  report/generate.py       incident brief, Claude API with a deterministic fallback
backend/main.py            FastAPI, the eight endpoints
frontend/                  Sentinel Grid dashboard, served at /
run_demo.py                starts the server
```

---

## 11. Running the modules

Every module runs standalone and fails loudly on missing data rather than substituting
zeros. A zero hazard score and an unmeasured cell are opposite claims.

```bash
# the Coimbatore path, end to end
python -m src.scenario --build coimbatore_flood -o data/cache/cell_inputs.cbe.json
python -m src.hazard.zones --config configs/disasters/flood_rainfall.yaml \
    --cells data/cache/cell_inputs.cbe.json -o data/cache/zones.cbe.json

python -m src.ingest.imd_grid --grd data/raw/imd/rain_ind0.25_26_09_12.grd --top 10
python -m src.hazard.physics               # governing physics, worked examples
python -m src.detect.rppg --demo           # synthetic pulse, no camera
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

Two blocks there are calibration knobs, and both say so in place.
The GMPE attenuation coefficients need refitting per region before any absolute PGA is
trustworthy, and the §4.4 exposure weights have no values in the spec, so they sit at
equal thirds.

---

## 13. Status

- [x] Coimbatore ward scenario, 30 wards on the Noyyal
- [x] Seven hazards over one ward model, each with its own derived inputs
- [x] Zone engine: noisy-OR `U`, green suitability `G`, hysteresis, exposure score
- [x] Real IMD gridded rainfall, decoded and orientation-verified
- [x] Ingest: rainfall accumulation, terrain, infrastructure, layer ageing
- [x] Preprocessing: five feature sets, leakage guard, four trained classifiers
- [x] Detection: POS rPPG, thermal blobs, RGB-T merge, Bayesian log-odds fusion
- [x] Priority: survivability decay, expected lives, greedy dispatch, exposure fallback
- [x] Routing: risk-weighted Dijkstra, capacity-aware evacuation
- [x] Incident Action Plan export, all four mandates
- [x] FastAPI backend, eight endpoints, live simulation
- [x] Sentinel Grid dashboard wired to every endpoint
- [x] 314 verification checks
- [x] Aerial person detector (YOLO11n) fine-tuned on real photos, see §14
- [x] Acoustic classifier (real YAMNet) with a transfer head, see §14
- [x] Heart-rate estimation model, trained on real PPG data, see §14
- [x] Gradient-boosted model for extreme-monsoon prediction, see §14 — beats the logistic baseline's recall but is not a solved task
- [ ] Flood segmentation model (SegFormer/U-Net) — still not built, no code or weights
- [ ] Real Bhuvan, Sentinel-1 and WorldPop exports replacing the modelled ward terrain
- [ ] Thermal YOLO (FLIR ADAS/LLVIP) — blocked, both datasets are registration-gated

---

## 14. GPU-trained models (this fork)

This fork adds real GPU training on top of the base pipeline above. Everything in §1–§13
is unchanged; this section documents what got added and where it runs.

The base repo's own trained models (§6) are plain CPU logistic regression — deliberately,
as a yardstick to check whether a task is separable before spending GPU time on it. This
fork adds a second, GPU-trained layer for the tasks worth going further on, using an
NVIDIA RTX 3050 (CUDA), and — for the detection models the base repo ships with no
weights or training data for — finds and trains on real, no-login datasets instead of
leaving them stubbed.

| Model | Dataset | Result | Honest read |
|---|---|---|---|
| Landslide / Cyclone / Drought — PyTorch MLP | same Kaggle sets as §6 | matches the logistic baseline (≥97% acc) | confirms these tasks are already linearly separable; the MLP's extra capacity doesn't buy more |
| Extreme monsoon — PyTorch MLP (class-weighted) | same flood_season set as §6 | recall 2% → 63% | class-weighting the loss — not available in the base logistic CLI — is most of the fix |
| Extreme monsoon — XGBoost (GPU) | same flood_season set | recall 46%, acc 68% | the gradient-boosted model §6 names as the open task, actually built; not a solved task, but a real improvement |
| YOLO11n person detector | PennFudanPed (170 real photos, direct download) | precision 0.988, recall 0.958, mAP50 0.992 | fine-tuned from Ultralytics' COCO-pretrained weights, not trained from scratch |
| YAMNet + transfer head | real Google AudioSet checkpoint (TF-Hub) + a synthetic proxy corpus | real inference confirmed (correctly IDs a synthetic siren-like tone); transfer head reports 100% on a 20-sample synthetic set | **no licensed distress-audio dataset was available without registration** — the YAMNet backbone is real and unmodified, the small head on top is not evaluated on real screams |
| Heart-rate (PPG → BPM) | PPG-DaLiA (real wrist PPG + accelerometer + ECG-derived ground truth, 15 subjects, via a Zenodo mirror) | MAE 28.7 bpm, RMSE 34.9 bpm, held-out | a real, honestly weak number — PPG-DaLiA is deliberately hard (real motion artifacts), and this is a simple FFT-plus-accelerometer baseline, not a proper motion-artifact-cancellation model |

**Where this shows up in the dashboard:**
- **ML Models tab** → a "GPU-Trained Models" card section lists all of the above with their real measured metrics, alongside the base repo's own two model cards.
- **Bio-Radar tab** → the pulse chart's frequency and displayed BPM are now driven by a real PPG-DaLiA test window run through the trained heart-rate model (`GET /api/model/heartbeat_samples`), instead of `Math.random()`. Same chart, real parameters underneath.

**Two bugs hit and fixed in the GPU training tool itself** (outside this repo, not in the code above): `torchvision` installed CPU-only against a CUDA `torch` build, and a Windows-specific quirk where a detached parent process's captured subprocess stdout came back empty despite success — fixed by having each job write its result to a file instead.

**Not done, and not claimed to be:** thermal YOLO (FLIR ADAS/LLVIP are both registration-gated), flood segmentation (no code exists for it at all), and the YAMNet transfer head is not validated against real distress audio — see the table above.

Training code for this section lives outside this repo, in a separate `pyos.py` GPU
queue with its own live dashboard; `trained_models/` here holds the metrics this
section reports (model weight files are git-ignored — see `.gitignore`).
