"""Dataset registry and offline-first fetcher (spec section 9, PS-1 dataset stack).

Every dataset PS-1 names lives here with its URL, licence note and local path. The rule
the whole demo depends on: fetch during the download phase, read from disk on stage.
`load()` never touches the network unless you pass allow_download=True, so a forgotten
call cannot turn into a live request in front of judges.

    python -m src.ingest.sources --list
    python -m src.ingest.sources --fetch imd_rainfall usgs_quakes --download
"""
import argparse
import hashlib
import json
import pathlib
import urllib.error
import urllib.request

from src.hazard.formulas import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw"
CACHE = REPO_ROOT / "data" / "cache"

# auth: "none" downloads unattended. "register" needs a manual login first, so those are
# listed for the slides and fetched by hand, not by this script (spec section 9.2).
SOURCES = {
    "imd_rainfall": {
        "url": "https://indiawris.gov.in/wris/#/rainfall",
        "local": "imd/district_rainfall_daily.csv",
        "auth": "none",
        "level": 1,
        "what": "IMD district-wise daily rainfall via the National Water Data Portal",
        "note": "PS-1 primary rainfall source. Portal export, not a direct file URL.",
    },
    "cwc_gauges": {
        "url": "https://indiawris.gov.in/wris/#/riverMonitoring",
        "local": "cwc/river_gauge_levels.csv",
        "auth": "none",
        "level": 1,
        "what": "CWC river gauge levels, for river-overflow risk",
    },
    "glofas": {
        "url": "https://global-flood.emergency.copernicus.eu/",
        "local": "glofas/discharge_forecast.nc",
        "auth": "register",
        "level": 1,
        "what": "Copernicus GloFAS river discharge forecasts, 24-48 h horizon",
    },
    "bhuvan_flood_vulnerability": {
        "url": "https://bhuvan-app1.nrsc.gov.in/disaster/disaster.php",
        "local": "bhuvan/flood_vulnerability_index.geojson",
        "auth": "register",
        "level": 2,
        "what": "ISRO Bhuvan National Flood Vulnerability Index",
    },
    "srtm_dem": {
        "url": "https://portal.opentopography.org/apidocs/",
        "local": "dem/srtm_30m.tif",
        "auth": "none",
        "level": 2,
        "what": "SRTM 30 m elevation, source of slope and relative height",
    },
    "sentinel1_sar": {
        "url": "https://dataspace.copernicus.eu/",
        "local": "sentinel1/vv_backscatter.tif",
        "auth": "register",
        "level": 2,
        "what": "Sentinel-1 SAR VV. Water reads below -18 dB and sees through monsoon cloud",
    },
    "copernicus_ems": {
        "url": "https://emergency.copernicus.eu/mapping/list-of-activations-rapid",
        "local": "copernicus/ems_flood_delineation.geojson",
        "auth": "none",
        "level": 2,
        "what": "Copernicus EMS rapid-mapping flood delineations, for validation",
    },
    "osm_infrastructure": {
        "url": "https://overpass-api.de/api/interpreter",
        "local": "osm/infrastructure.geojson",
        "auth": "none",
        "level": 3,
        "what": "Roads, bridges, hospitals, schools, shelters. Fetched via OSMnx",
    },
    "worldpop": {
        "url": "https://hub.worldpop.org/geodata/listing?id=29",
        "local": "worldpop/population_100m.tif",
        "auth": "none",
        "level": 3,
        "what": "100 m population density, the N in the priority score",
    },
    "nasa_gdis": {
        "url": "https://data.nasa.gov/",
        "local": "gdis/geocoded_disasters.csv",
        "auth": "none",
        "level": 3,
        "what": "NASA Geocoded Disasters, historical events for validation",
    },
    "usgs_quakes": {
        "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson",
        "local": "usgs/all_month.geojson",
        "auth": "none",
        "level": 1,
        "what": "USGS earthquake feed. Auth-free and instant, so it is the smoke test",
    },
}


def local_path(key):
    if key not in SOURCES:
        raise KeyError(f"unknown source {key!r}. Known: {', '.join(sorted(SOURCES))}")
    return RAW / SOURCES[key]["local"]


def fetch(key, allow_download=False, timeout=60):
    """Return the local path, downloading only when explicitly permitted."""
    src = SOURCES[key]
    path = local_path(key)

    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"{key} not present at {path}. Run the download phase first:\n"
            f"    python -m src.ingest.sources --fetch {key} --download"
        )
    if src["auth"] != "none":
        raise PermissionError(
            f"{key} needs a manual login at {src['url']}. Download it by hand and save it "
            f"to {path}."
        )
    if not src["url"].endswith((".geojson", ".csv", ".json", ".tif", ".zip", ".nc")):
        raise ValueError(
            f"{key} is a portal page, not a direct file: {src['url']}\n"
            f"Export the data by hand and save it to {path}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(src["url"], timeout=timeout) as response:
            path.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as e:
        raise ConnectionError(f"could not download {key} from {src['url']}: {e}")
    return path


def checksum(path):
    """SHA-256 of a local file, so four laptops can confirm they hold the same data."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(write=False):
    """What is present, what is missing, and the hash of everything present."""
    rows = {}
    for key in sorted(SOURCES):
        path = local_path(key)
        rows[key] = {
            "present": path.exists(),
            "path": str(path.relative_to(REPO_ROOT)),
            "level": SOURCES[key]["level"],
            "auth": SOURCES[key]["auth"],
            "sha256": checksum(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    if write:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def _cli():
    ap = argparse.ArgumentParser(description="Dataset registry and offline-first fetcher.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch", nargs="*", help="source keys, or all auth-free ones if empty")
    ap.add_argument("--download", action="store_true", help="permit network access")
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args()

    if args.list:
        for level in (1, 2, 3):
            print(f"\nLEVEL {level}")
            for key, s in sorted(SOURCES.items()):
                if s["level"] != level:
                    continue
                mark = "present" if local_path(key).exists() else "MISSING"
                print(f"  {key:<28} {mark:<8} {s['auth']:<9} {s['what']}")
        return

    if args.manifest:
        rows = manifest(write=True)
        have = sum(1 for r in rows.values() if r["present"])
        print(f"{have}/{len(rows)} present. Written to data/cache/manifest.json")
        for key, r in rows.items():
            if not r["present"]:
                print(f"  missing: {key} -> {r['path']}")
        return

    if args.fetch is not None:
        keys = args.fetch or [k for k, s in SOURCES.items() if s["auth"] == "none"]
        for key in keys:
            try:
                path = fetch(key, allow_download=args.download)
                print(f"  ok       {key} -> {path.relative_to(REPO_ROOT)}")
            except (FileNotFoundError, PermissionError, ValueError, ConnectionError) as e:
                print(f"  SKIPPED  {key}: {e}")
        return

    ap.print_help()


if __name__ == "__main__":
    _cli()
