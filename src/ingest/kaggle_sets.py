"""Loader for the seven Kaggle disaster datasets, one per hazard config.

The column names in these files are not known ahead of time and differ between them, so
nothing here hardcodes a schema. Each disaster config declares a `dataset:` block listing
candidate column names, and this module resolves the real header against those candidates.
Adapting to the actual file is a YAML edit, not a code change.

Workflow when a dataset lands:

    python -m src.ingest.kaggle_sets --list
    python -m src.ingest.kaggle_sets --inspect flood_rainfall   # what columns does it really have
    python -m src.ingest.kaggle_sets --load flood_rainfall -o data/cache/cell_inputs.rain.json

--inspect prints the header, says which config aliases matched, and names the ones that
did not, so the fix is obvious rather than a guess.
"""
import argparse
import csv
import json
import re
import pathlib
import sys

from src.grid import cell_id
from src.hazard.formulas import REPO_ROOT, load_config

RAW = REPO_ROOT / "data" / "raw" / "kaggle"

# disaster config name -> Kaggle slug and the file we expect inside the zip
# What each file actually turned out to contain, checked against the download rather
# than assumed from the title. `griddable` is the one that matters: a dataset with no
# coordinates cannot place hazard on a map, however good it is for training a classifier.
DATASETS = {
    "earthquake": {
        "slug": "ankitd7752/indian-subcontinent-earthquake-data-2000-to-2024",
        "kind": "event_catalog",
        "griddable": True,
        "what": "USGS-format catalogue: time, latitude, longitude, depth, mag",
        "note": "Works end to end. PGA is derived from magnitude and depth.",
    },
    "drought": {
        "slug": "kevinmathewsgeorge/india-drought-analysis-data-2000-2023",
        "kind": "observation",
        "griddable": False,
        "what": "GLDAS groundwater by district: date, ADM2_NAME, mean, source",
        "note": "Named districts, no coordinates. Join to a district centroid table first.",
    },
    "flood_rainfall": {
        "slug": "aksahaha/rainfall-india",
        "kind": "observation",
        "griddable": False,
        "what": "Seasonal totals by subdivision: subdivision, YEAR, JUN, JUL, AUG, SEP",
        "note": ("Monsoon-season totals, not daily, and no coordinates. The IMD 0.25 "
                 "degree .grd read by src/ingest/imd_grid.py is the better source for "
                 "the same job and is already wired."),
    },
    "cyclone": {
        "slug": "rajumavinmar/cyclone-dataset",
        "kind": "feature_table",
        "griddable": False,
        "what": "Formation features with a binary Cyclone label; Latitude but no longitude",
        "note": "A classifier training set, not tracks. Cannot be placed on the grid.",
    },
    "landslide": {
        "slug": "rajumavinmar/landslide-dataset",
        "kind": "feature_table",
        "griddable": False,
        "what": "Rainfall_mm, Slope_Angle, Soil_Saturation and a binary Landslide label",
        "note": ("No coordinates and no dates. Good for training a susceptibility model, "
                 "which is exactly what the spec's XGBoost layer would use."),
    },
    "wildfire": {
        "slug": "jaynadkarni/indian-forest-fires-dataset",
        "kind": "aggregate",
        "griddable": False,
        "what": "State-level annual fire counts for 2008-09 through 2010-11",
        "note": "Three years of state totals. Base rates only; NASA FIRMS is the live feed.",
    },
    "lightning": {
        "slug": "shhhantanu/lightning-event-records-for-india-20192022",
        "kind": "event_catalog",
        "griddable": None,
        "what": "Parquet strike records, 2.8 GB unpacked",
        "note": ("Too large for the repo and stored as parquet, which this loader does "
                 "not read. Kept out of git; download it on the machine that needs it."),
    },
}

CSV_LIMIT = 10 ** 7


# Some datasets carry the physical measurements but not the quantity the hazard formula
# wants. An earthquake catalogue lists magnitude and depth, never PGA. Rather than
# demanding a column that does not exist, derive it with the physics already in
# src/hazard/physics.py.
def _derive_pga(record):
    """PGA at the surface above the hypocentre, from magnitude and focal depth.

    Epicentral distance is taken as zero: we are scoring the cell the event happened in,
    so the hypocentral distance is the focal depth. That is the strongest shaking the
    event produces, which is the right number for a worst-case impact zone and the wrong
    one for anywhere further out. Scoring a whole region means running the GMPE per cell
    with that cell's distance, which is what src/hazard/physics.py exposes.
    """
    from src.hazard.physics import amplify_vs30, gmpe_pga
    depth = record.get("depth_km")
    magnitude = record.get("magnitude_mw")
    if magnitude is None or depth is None or depth <= 0:
        return None
    pga = gmpe_pga(magnitude, epicentral_distance_km=0.0, focal_depth_km=depth)
    vs30 = record.get("vs30")
    return amplify_vs30(pga, vs30) if vs30 else pga


DERIVATIONS = {
    ("earthquake", "pga_ms2"): _derive_pga,
}


def _tokens(name):
    """Column name -> lowercase word tokens. "Focal Depth (km)" -> focal, depth, km."""
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def dataset_dir(name):
    return RAW / name


def find_csv(name):
    """The largest CSV under the dataset folder. Kaggle zips often carry several."""
    d = dataset_dir(name)
    if not d.exists():
        raise FileNotFoundError(
            f"{name} not downloaded. Run:\n"
            f"    kaggle datasets download -d {DATASETS[name]['slug']} -p {d} --unzip")
    candidates = sorted(d.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no CSV under {d}")
    return candidates[0]


def read_header(path):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        return next(reader, [])


def resolve_columns(header, wanted):
    """{field: [aliases]} against a real header -> ({field: column}, [unresolved]).

    Exact case-insensitive match first, then a whole-token match, so "Magnitude (Mw)"
    resolves to magnitude and "Focal Depth (km)" to depth.

    Matching is on whole tokens, never bare substrings. A substring rule looks harmless
    and is not: against these seven files the alias "y" matches YEAR, "x" matches
    Proximity_to_Water, and "lon" matches Cyclone. Each silently binds a column of
    unrelated numbers to a coordinate and puts the hazard somewhere it never happened,
    which is far worse than reporting the column as unresolved.
    """
    lowered = {h.lower().strip(): h for h in header}
    tokenised = [(h, set(_tokens(h))) for h in header]
    resolved, unresolved = {}, []

    for field, aliases in wanted.items():
        match = next((lowered[a.lower()] for a in aliases if a.lower() in lowered), None)
        if match is None:
            for alias in aliases:
                alias_tokens = set(_tokens(alias))
                if not alias_tokens:
                    continue
                match = next((h for h, toks in tokenised if alias_tokens <= toks), None)
                if match:
                    break
        if match:
            resolved[field] = match
        else:
            unresolved.append(field)

    return resolved, unresolved


def dataset_config(disaster_cfg):
    """The `dataset:` block from a disaster config, or a clear error."""
    block = disaster_cfg.get("dataset")
    if not block or "columns" not in block:
        raise ValueError(
            f"{disaster_cfg['name']}: no `dataset.columns` block in its config. Add one "
            f"mapping each hazard input to candidate column names, then rerun.")
    return block


def load_records(name, limit=None):
    """CSV -> [{field: value}] using the aliases in the disaster config.

    Rows missing a required hazard input are skipped and counted, never defaulted to
    zero: an unmeasured cell and a zero-hazard cell are opposite claims.
    """
    cfg = load_config(f"configs/disasters/{name}.yaml")
    block = dataset_config(cfg)
    path = find_csv(name)

    header = read_header(path)
    resolved, unresolved = resolve_columns(header, block["columns"])
    required = set(cfg["hazard"]["inputs"])
    derivable = {f for f in required if (name, f) in DERIVATIONS}
    missing_required = required - set(resolved) - derivable
    if missing_required:
        raise ValueError(
            f"{path.name}: could not find a column for {sorted(missing_required)}.\n"
            f"Header is: {header}\n"
            f"Add the real column name to dataset.columns in "
            f"configs/disasters/{name}.yaml")

    csv.field_size_limit(CSV_LIMIT)
    records, skipped = [], 0
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            record = {}
            ok = True
            for field, column in resolved.items():
                raw = (row.get(column) or "").strip()
                if raw == "" or raw.upper() in ("NA", "N/A", "NAN", "NULL", "-"):
                    if field in required and (name, field) not in DERIVATIONS:
                        ok = False
                        break
                    continue
                try:
                    record[field] = float(raw)
                except ValueError:
                    record[field] = raw
            if not ok:
                skipped += 1
                continue

            for (dataset_name, field), derive in DERIVATIONS.items():
                if dataset_name == name and field not in record:
                    value = derive(record)
                    if value is not None:
                        record[field] = value
            if not required <= set(record):
                skipped += 1
                continue

            records.append(record)
            if limit and len(records) >= limit:
                break

    return records, {"path": path, "header": header, "resolved": resolved,
                     "unresolved": unresolved, "skipped": skipped}


def to_cell_inputs(name, records, top_n=40, default_sensed=0.95, age_hours=3.0):
    """Records with lat/lon -> cell inputs for the zone engine.

    Only the hazard inputs are real. Terrain, shelter capacity and population are
    stand-ins and are labelled as such in every record, so nobody reads a modelled
    number as a measurement.
    """
    cfg = load_config(f"configs/disasters/{name}.yaml")
    required = cfg["hazard"]["inputs"]

    placed = [r for r in records if "lat" in r and "lon" in r]
    if not placed:
        raise ValueError(
            f"{name}: no rows carry both lat and lon, so nothing can be put on the grid. "
            f"Add their column names to dataset.columns, or join this dataset to a "
            f"district centroid table first.")

    # Prefer the most recent events. Ranking by hazard instead selects only the largest
    # ones, and for an earthquake catalogue every large event saturates X at 1.0 directly
    # above its hypocentre, so the map comes out uniformly red and says nothing.
    dated = [r for r in placed if isinstance(r.get("date"), str) and r["date"].strip()]
    if len(dated) >= top_n:
        ranked = sorted(dated, key=lambda r: r["date"], reverse=True)[:top_n]
    else:
        ranked = sorted(placed, key=lambda r: -max(
            (v for k, v in r.items() if k in required and isinstance(v, (int, float))),
            default=0.0))[:top_n]

    cells, seen = [], set()
    for r in ranked:
        lat, lon = float(r["lat"]), float(r["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        cid = cell_id(lat, lon)
        if cid in seen:
            continue                       # one cell, one record
        seen.add(cid)

        cells.append({
            "cell_id": cid,
            "centroid": [round(lat, 4), round(lon, 4)],
            "hazard_inputs": {k: r[k] for k in required if k in r},
            "sensors": {"sigma_ens": 0.0, "data_age_hours": age_hours,
                        "area_sensed_fraction": default_sensed},
            "green_inputs": {"forecast_max_x": 0.5, "elevation_norm": 0.5,
                             "shelter_capacity": 1200, "pop_inflow": 2000,
                             "access": 0.8, "travel_distance_norm": 0.3},
            "pop": int(r.get("pop", 0)),
            "assets": [],
            "reachable": True,
            "components": {
                "source": f"kaggle:{DATASETS[name]['slug']}",
                "measured": [k for k in required if k in r],
                "modelled": ["green_inputs", "pop", "sensors"],
            },
        })
    return cells


def _cli():
    ap = argparse.ArgumentParser(description="Load the Kaggle disaster datasets.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", choices=sorted(DATASETS))
    ap.add_argument("--load", choices=sorted(DATASETS))
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    if args.list or not (args.inspect or args.load):
        print(f"{'disaster':<12} {'status':<14} {'kind':<14} kaggle slug")
        for name, meta in sorted(DATASETS.items()):
            present = dataset_dir(name).exists()
            print(f"{name:<12} {'downloaded' if present else 'MISSING':<14} "
                  f"{meta['kind']:<14} {meta['slug']}")
        print("\nDownload all seven:")
        for name, meta in sorted(DATASETS.items()):
            print(f"  kaggle datasets download -d {meta['slug']} "
                  f"-p data/raw/kaggle/{name} --unzip")
        return

    if args.inspect:
        name = args.inspect
        path = find_csv(name)
        header = read_header(path)
        cfg = load_config(f"configs/disasters/{name}.yaml")
        print(f"{path.relative_to(REPO_ROOT)}\n")
        print(f"header ({len(header)} columns):")
        for h in header:
            print(f"  {h}")

        try:
            block = dataset_config(cfg)
        except ValueError as e:
            print(f"\n{e}")
            return
        resolved, unresolved = resolve_columns(header, block["columns"])
        print(f"\nmatched:")
        for field, column in sorted(resolved.items()):
            print(f"  {field:<16} <- {column}")
        if unresolved:
            print(f"\nNOT matched (add the real name to dataset.columns):")
            for field in unresolved:
                print(f"  {field:<16} tried {block['columns'][field]}")
        required = set(cfg["hazard"]["inputs"])
        derivable = {f for f in required if (name, f) in DERIVATIONS}
        blocking = required - set(resolved) - derivable
        print(f"\nhazard inputs required: {sorted(required)}")
        if derivable:
            print(f"derived from other columns: {sorted(derivable)}")
        print("ready to load" if not blocking else f"BLOCKED on {sorted(blocking)}")
        return

    name = args.load
    records, info = load_records(name)
    print(f"{info['path'].name}: {len(records)} rows, {info['skipped']} skipped")
    if info["unresolved"]:
        print(f"  unmatched columns: {info['unresolved']}")

    cells = to_cell_inputs(name, records, args.top)
    text = json.dumps(cells, indent=2)
    if args.out:
        out = pathlib.Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"{out.relative_to(REPO_ROOT)}: {len(cells)} cells")
        print(f"\nNext:  python -m src.hazard.zones "
              f"--config configs/disasters/{name}.yaml --cells {args.out} "
              f"-o data/cache/zones.{name}.json")
    else:
        print(text[:2000])


if __name__ == "__main__":
    sys.exit(_cli())
