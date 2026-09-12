"""Level 1: atmospheric and meteorological ingest (spec section 1.1).

PS-1's scenario is a monsoon over the next 24-48 hours, so this level carries the
variables the flood and landslide formulas need: 24 h rainfall, multi-day cumulative
rainfall for soil saturation, and river gauge level against bankfull.

Reads IMD district daily rainfall as exported from the National Water Data Portal.
Column names differ between exports, so the reader matches on a set of known aliases
and raises with the header it actually saw rather than guessing.

    python -m src.ingest.level1_meteo --rainfall data/raw/imd/district_rainfall_daily.csv
"""
import argparse
import csv
import datetime as dt
import pathlib

# NWDP and IMD exports have used all of these for the same three fields.
ALIASES = {
    "district": ["district", "district_name", "dist", "districtname"],
    "date": ["date", "obs_date", "observation_date", "day"],
    "rainfall_mm": ["rainfall_mm", "rainfall", "rain_mm", "actual_rainfall", "value"],
}
HEAVY_RAIN_MM = 64.5      # IMD's heavy-rainfall warning threshold for a 24 h total


def _resolve(header):
    lowered = {h.lower().strip(): h for h in header}
    resolved = {}
    for field, names in ALIASES.items():
        match = next((lowered[n] for n in names if n in lowered), None)
        if match is None:
            raise ValueError(
                f"rainfall CSV has no column for {field!r}. Looked for {names}, "
                f"header is {list(header)}"
            )
        resolved[field] = match
    return resolved


def read_district_rainfall(path):
    """CSV -> {district: [(date, mm), ...]} sorted oldest first."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"rainfall file not found: {p}. Export it from the National Water Data "
            f"Portal during the download phase (see src/ingest/sources.py)."
        )

    series = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{p} is empty")
        cols = _resolve(reader.fieldnames)
        for row in reader:
            district = (row[cols["district"]] or "").strip()
            raw_date = (row[cols["date"]] or "").strip()
            raw_mm = (row[cols["rainfall_mm"]] or "").strip()
            if not district or not raw_date or raw_mm == "":
                continue
            try:
                mm = float(raw_mm)
            except ValueError:
                continue                       # portal exports use "NA" for no reading
            if mm < 0:
                continue                       # sentinel for missing, not negative rain
            series.setdefault(district, []).append((parse_date(raw_date), mm))

    if not series:
        raise ValueError(f"{p} parsed to zero usable rows; check the export format")
    for rows in series.values():
        rows.sort(key=lambda r: r[0])
    return series


def parse_date(text):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date format: {text!r}")


def accumulate(series, days, as_of=None):
    """Rolling total over the last `days` observations ending at `as_of`.

    Returns {district: mm}. Districts with no reading in the window are absent rather
    than zero: no rain and no gauge are different claims (spec section 12.9).
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    out = {}
    for district, rows in series.items():
        window = [(d, mm) for d, mm in rows if as_of is None or d <= as_of]
        if not window:
            continue
        end = window[-1][0]
        start = end - dt.timedelta(days=days - 1)
        total = sum(mm for d, mm in window if start <= d <= end)
        out[district] = round(total, 2)
    return out


def flood_inputs(series, as_of=None, q_max=None, gauges=None):
    """Everything configs/disasters/flood.yaml declares, per district.

    q_max is bankfull discharge. Without a gauge reading the discharge term is absent,
    and the caller decides whether to run on rainfall alone rather than this module
    silently inventing a zero.
    """
    r24 = accumulate(series, 1, as_of)
    r3d = accumulate(series, 3, as_of)
    r5d = accumulate(series, 5, as_of)

    out = {}
    for district in r24:
        row = {
            "r24_mm": r24[district],
            "precip_3d_cum": r3d.get(district),
            "precip_5d_cum": r5d.get(district),
            "heavy_rain_warning": r24[district] > HEAVY_RAIN_MM,
        }
        if gauges and district in gauges:
            row["river_gauge_m"] = gauges[district]
            if q_max:
                row["q_cumecs"] = gauges[district]
                row["q_max"] = q_max
        out[district] = row
    return out


def landslide_inputs(series, slope_by_district, as_of=None):
    """configs/disasters/landslide.yaml needs 3-day rainfall and slope."""
    r3d = accumulate(series, 3, as_of)
    return {
        d: {"r3d_mm": r3d[d], "slope_deg": slope_by_district[d]}
        for d in r3d if d in slope_by_district
    }


def _cli():
    ap = argparse.ArgumentParser(description="Level 1 meteorological ingest.")
    ap.add_argument("--rainfall", required=True)
    ap.add_argument("--as-of", help="YYYY-MM-DD, defaults to the latest reading")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    series = read_district_rainfall(args.rainfall)
    as_of = parse_date(args.as_of) if args.as_of else None
    rows = flood_inputs(series, as_of)

    print(f"{len(series)} districts, {sum(len(v) for v in series.values())} observations")
    ranked = sorted(rows.items(), key=lambda kv: kv[1]["r24_mm"], reverse=True)
    print(f"\n{'district':<28} {'24h':>8} {'3d':>8} {'5d':>8}  warning")
    for district, r in ranked[:args.top]:
        warn = "HEAVY" if r["heavy_rain_warning"] else ""
        print(f"{district:<28} {r['r24_mm']:>8.1f} "
              f"{r['precip_3d_cum'] or 0:>8.1f} {r['precip_5d_cum'] or 0:>8.1f}  {warn}")


if __name__ == "__main__":
    _cli()
