"""Preprocessing and feature engineering (spec section 1.1, the stage before inference).

Turns a raw dataset into arrays a model can load in two lines:

    import numpy as np
    d = np.load("data/cache/features/landslide/train.npz")
    X, y = d["X"], d["y"]

Three things this does that a bare pandas read does not:

  Standardisation is fitted on train only. Fitting on the full set before splitting
  leaks validation statistics into training and quietly inflates every score you then
  report.

  One-hot columns are left unscaled. Standardising a 0/1 indicator destroys the thing
  that makes it readable and gains nothing.

  The split is stratified, so both classes keep their proportions. On a balanced set that
  barely matters; on the imbalanced ones it is the difference between a validation fold
  that measures something and one that does not.

    python -m src.features --profile landslide
    python -m src.features --build landslide
    python -m src.features --build-all
"""
import argparse
import csv
import json
import math
import pathlib

import numpy as np

from src.hazard.formulas import REPO_ROOT

OUT = REPO_ROOT / "data" / "cache" / "features"

# Column roles per dataset, checked against the real files rather than assumed.
# continuous -> standardised. binary -> passed through as 0/1. target -> y.
SCHEMAS = {
    "landslide": {
        "csv": "data/raw/kaggle/landslide/landslide_dataset.csv",
        "target": "Landslide",
        "continuous": ["Rainfall_mm", "Slope_Angle", "Soil_Saturation",
                       "Vegetation_Cover", "Earthquake_Activity", "Proximity_to_Water"],
        "binary": ["Soil_Type_Gravel", "Soil_Type_Sand", "Soil_Type_Silt"],
        "note": ("478 of 2000 rows have all three soil indicators at zero. That is a "
                 "dropped reference category, not missing data: those rows are a fourth "
                 "soil type the file does not name. Do not impute them and do not add a "
                 "fourth column, or you reintroduce the dummy trap."),
        "feeds": "configs/disasters/landslide.yaml susceptibility layer over the FS formula",
    },
    "cyclone": {
        "csv": "data/raw/kaggle/cyclone/cyclone_dataset.csv",
        "target": "Cyclone",
        "continuous": ["Sea_Surface_Temperature", "Atmospheric_Pressure", "Humidity",
                       "Wind_Shear", "Vorticity", "Latitude", "Ocean_Depth",
                       "Proximity_to_Coastline"],
        "binary": [],
        "excluded": {
            "Pre_existing_Disturbance": (
                "LEAKED. Identical to the target Cyclone in all 2000 rows, r = 1.000. A "
                "model using this column alone scores 100% and learns nothing. The "
                "remaining eight features are the real problem."),
        },
        "note": ("Formation likelihood, not track or intensity. It answers 'will a "
                 "cyclone form in these conditions', which is not the same question as "
                 "the Holland wind field, and it carries no longitude so it cannot be "
                 "placed on the grid."),
        "feeds": "configs/disasters/cyclone.yaml formation prior, not the hazard score",
    },
}


def read_csv(path):
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it first:\n"
            f"    python -m src.ingest.kaggle_sets --list")
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def clean(rows, schema):
    """Coerce to float, drop rows with any missing or non-finite value, drop duplicates.

    Returns (X, y, feature_names, report). Rows are dropped rather than imputed: on a
    2000-row set with no missingness, an imputer is machinery that only hides a problem
    if one later appears.
    """
    features = schema["continuous"] + schema["binary"]
    target = schema["target"]

    kept, dropped_missing, dropped_bad = [], 0, 0
    for row in rows:
        values = []
        ok = True
        for column in features + [target]:
            raw = (row.get(column) or "").strip()
            if raw == "" or raw.upper() in ("NA", "N/A", "NAN", "NULL"):
                ok = False
                break
            try:
                value = float(raw)
            except ValueError:
                ok = False
                break
            if not math.isfinite(value):
                ok = False
                break
            values.append(value)
        if ok:
            kept.append(values)
        else:
            dropped_missing += 1

    if not kept:
        raise ValueError("every row was dropped; check the column names in SCHEMAS")

    array = np.asarray(kept, dtype=float)
    before = len(array)
    _unique, index = np.unique(array, axis=0, return_index=True)
    array = array[np.sort(index)]                 # dedupe, original order preserved
    dropped_bad = before - len(array)

    X, y = array[:, :-1], array[:, -1]
    labels = set(np.unique(y).tolist())
    if not labels <= {0.0, 1.0}:
        raise ValueError(f"{target} is not binary; found {sorted(labels)}")

    return X, y.astype(int), features, {
        "rows_in": len(rows),
        "rows_kept": len(array),
        "dropped_missing": dropped_missing,
        "dropped_duplicate": dropped_bad,
    }


def stratified_split(y, val_fraction=0.2, seed=0):
    """Indices for train and val, keeping each class's proportion in both."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * (1 - val_fraction)))
        train.extend(idx[:cut].tolist())
        val.extend(idx[cut:].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    return np.array(train), np.array(val)


def fit_scaler(X, continuous_count):
    """Mean and std over the continuous columns of the TRAINING rows only.

    A zero-variance column would divide by zero, so its scale is forced to 1 and it
    passes through unchanged rather than becoming NaN partway through training.
    """
    mean = X[:, :continuous_count].mean(axis=0)
    std = X[:, :continuous_count].std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def apply_scaler(X, mean, std, continuous_count):
    out = X.copy()
    out[:, :continuous_count] = (out[:, :continuous_count] - mean) / std
    return out


def point_biserial(x, y):
    """Correlation between a continuous feature and a binary target.

    Which features actually carry signal, before anyone spends a night training on them.
    """
    g1, g0 = x[y == 1], x[y == 0]
    if len(g1) == 0 or len(g0) == 0:
        return 0.0
    sd = x.std()
    if sd < 1e-12:
        return 0.0
    n = len(x)
    return (g1.mean() - g0.mean()) / sd * math.sqrt(len(g1) * len(g0) / (n * n))


LEAKAGE_R = 0.99


def check_leakage(X, y, features, threshold=LEAKAGE_R):
    """Any feature almost perfectly correlated with the target is leakage, not signal.

    The cyclone file shipped with Pre_existing_Disturbance as a verbatim copy of its own
    target: r = 1.000, and a one-column model scores 100%. That is what this catches, and
    it is worth catching automatically because it looks like a triumph right up until the
    model meets real data.
    """
    return [(f, round(point_biserial(X[:, i], y), 4))
            for i, f in enumerate(features)
            if abs(point_biserial(X[:, i], y)) >= threshold]


def build(name, val_fraction=0.2, seed=0, out_dir=OUT, allow_leakage=False):
    schema = SCHEMAS[name]
    X, y, features, report = clean(read_csv(schema["csv"]), schema)
    n_cont = len(schema["continuous"])

    leaked = check_leakage(X, y, features)
    if leaked and not allow_leakage:
        detail = ", ".join(f"{f} (r={r})" for f, r in leaked)
        raise ValueError(
            f"{name}: target leakage in {detail}. Training on this gives a model that "
            f"scores perfectly and generalises to nothing. Move the column into the "
            f"schema's `excluded` block, or pass allow_leakage=True if you mean it.")

    train_idx, val_idx = stratified_split(y, val_fraction, seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Fit on train only. Fitting before the split leaks validation statistics.
    mean, std = fit_scaler(X_train, n_cont)
    X_train = apply_scaler(X_train, mean, std, n_cont)
    X_val = apply_scaler(X_val, mean, std, n_cont)

    target_dir = pathlib.Path(out_dir) / name
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(target_dir / "val.npz", X=X_val, y=y_val)

    correlations = {f: round(point_biserial(X[:, i], y), 4)
                    for i, f in enumerate(features)}
    metadata = {
        "dataset": name,
        "source_csv": schema["csv"],
        "target": schema["target"],
        "feature_names": features,
        "n_continuous": n_cont,
        "n_binary": len(schema["binary"]),
        "scaler": {"mean": mean.tolist(), "std": std.tolist(),
                   "applies_to": features[:n_cont],
                   "fitted_on": "training rows only"},
        "split": {"val_fraction": val_fraction, "seed": seed, "stratified": True,
                  "n_train": int(len(y_train)), "n_val": int(len(y_val))},
        "class_balance": {"train": {"0": int((y_train == 0).sum()),
                                    "1": int((y_train == 1).sum())},
                          "val": {"0": int((y_val == 0).sum()),
                                  "1": int((y_val == 1).sum())}},
        "point_biserial_with_target": correlations,
        "cleaning": report,
        "excluded_columns": schema.get("excluded", {}),
        "leakage_check": {"threshold_r": LEAKAGE_R, "flagged": leaked},
        "note": schema["note"],
        "feeds": schema["feeds"],
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def profile(name):
    """What is in the file, before committing a night of GPU time to it."""
    schema = SCHEMAS[name]
    X, y, features, report = clean(read_csv(schema["csv"]), schema)
    n_cont = len(schema["continuous"])

    print(f"{name}: {report['rows_kept']} usable rows of {report['rows_in']}, "
          f"{len(features)} features")
    if report["dropped_missing"] or report["dropped_duplicate"]:
        print(f"  dropped {report['dropped_missing']} incomplete, "
              f"{report['dropped_duplicate']} duplicate")

    ones = int((y == 1).sum())
    ratio = ones / len(y)
    print(f"  target {schema['target']}: {ones} positive / {len(y) - ones} negative "
          f"({ratio:.1%})")
    if not 0.35 <= ratio <= 0.65:
        print(f"  imbalanced: weight the loss or the majority class wins by default")

    print(f"\n  {'feature':<28} {'kind':<11} {'min':>10} {'max':>10} {'mean':>10} {'r':>7}")
    for i, f in enumerate(features):
        col = X[:, i]
        kind = "continuous" if i < n_cont else "binary"
        r = point_biserial(col, y)
        print(f"  {f:<28} {kind:<11} {col.min():>10.4g} {col.max():>10.4g} "
              f"{col.mean():>10.4g} {r:>7.3f}")

    strong = [f for i, f in enumerate(features) if abs(point_biserial(X[:, i], y)) > 0.2]
    print(f"\n  carrying signal (|r| > 0.2): {', '.join(strong) if strong else 'none'}")
    for column, why in schema.get("excluded", {}).items():
        print(f"\n  EXCLUDED {column}: {why}")
    print(f"\n  {schema['note']}")
    print(f"  feeds: {schema['feeds']}")


def logistic_baseline(name, epochs=400, lr=0.5, l2=1e-4, out_dir=OUT):
    """Plain logistic regression on the built features. Pure numpy, a second to run.

    Not a deliverable, a yardstick. If this scores 0.95 then the dataset is separable and
    a night of GPU time on something heavier buys almost nothing; if it scores 0.6 the
    features are weak and no architecture rescues that. Either answer is worth knowing
    before anyone starts training.
    """
    d = pathlib.Path(out_dir) / name
    if not (d / "train.npz").exists():
        raise FileNotFoundError(f"{d} not built. Run: python -m src.features --build {name}")
    train, val = np.load(d / "train.npz"), np.load(d / "val.npz")
    Xtr, ytr, Xva, yva = train["X"], train["y"], val["X"], val["y"]

    # Bias column, then batch gradient descent on the log-loss.
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xva = np.hstack([Xva, np.ones((len(Xva), 1))])
    w = np.zeros(Xtr.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Xtr @ w, -30, 30)))
        grad = Xtr.T @ (p - ytr) / len(ytr) + l2 * w
        w -= lr * grad

    def metrics(X, y):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"accuracy": float((pred == y).mean()), "precision": precision,
                "recall": recall, "f1": f1}

    return {"train": metrics(Xtr, ytr), "val": metrics(Xva, yva),
            "weights": dict(zip(json.loads((d / "metadata.json").read_text())
                                ["feature_names"] + ["bias"], w.round(4).tolist()))}


def _cli():
    ap = argparse.ArgumentParser(description="Clean and preprocess into model-ready arrays.")
    ap.add_argument("--profile", choices=sorted(SCHEMAS))
    ap.add_argument("--build", choices=sorted(SCHEMAS))
    ap.add_argument("--build-all", action="store_true")
    ap.add_argument("--baseline", choices=sorted(SCHEMAS))
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.profile:
        profile(args.profile)
        return
    if args.baseline:
        r = logistic_baseline(args.baseline)
        print(f"logistic regression baseline, {args.baseline}")
        for split in ("train", "val"):
            m = r[split]
            print(f"  {split:<6} accuracy {m['accuracy']:.4f}  precision {m['precision']:.4f}"
                  f"  recall {m['recall']:.4f}  f1 {m['f1']:.4f}")
        gap = r["train"]["accuracy"] - r["val"]["accuracy"]
        print(f"  train-val gap {gap:+.4f}" +
              ("  (overfitting)" if gap > 0.05 else "  (no overfitting)"))
        print("\n  weights, largest first:")
        for k, v in sorted(r["weights"].items(), key=lambda kv: -abs(kv[1]))[:6]:
            print(f"    {k:<28} {v:+.4f}")
        print("\n  Beat this with the trained model, or the trained model is not earning "
              "its GPU time.")
        return
    names = sorted(SCHEMAS) if args.build_all else ([args.build] if args.build else [])
    if not names:
        ap.print_help()
        return

    for name in names:
        m = build(name, args.val_fraction, args.seed)
        print(f"{name}: {m['split']['n_train']} train / {m['split']['n_val']} val, "
              f"{len(m['feature_names'])} features "
              f"-> data/cache/features/{name}/")
    print("\nLoad in a training script with:")
    print("    d = np.load('data/cache/features/<name>/train.npz'); X, y = d['X'], d['y']")


if __name__ == "__main__":
    _cli()
