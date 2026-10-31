"""
EdgeFit HAR — offline training & benchmarking pipeline.

Downloads the public UCI HAR dataset, trains all candidate models on subjects 1–24,
evaluates them once on the unseen subjects 25–30, benchmarks size and CPU latency,
and writes every artifact the Streamlit app needs into ./artifacts.

Usage:
    python train.py                      # downloads the dataset from UCI
    python train.py --data-dir "path/to/UCI HAR Dataset"   # use a local copy
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import re
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SUBJECTS = [25, 26, 27, 28, 29, 30]
UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
FEATURE_COUNTS = [5, 10, 15, 25, 50, 100, 200, 561]   # for the edge trade-off curve
EDGE_K = 25                                            # feature budget of the edge model
ART = Path(__file__).parent / "artifacts"


# ----------------------------------------------------------------------------- data
def download_dataset(cache: Path) -> Path:
    """Download and extract the official UCI archive (handles the nested zip)."""
    cache.mkdir(parents=True, exist_ok=True)
    outer = cache / "uci_har.zip"
    if not outer.exists():
        print("Downloading UCI HAR dataset ...")
        urllib.request.urlretrieve(UCI_URL, outer)
    extract = cache / "extracted"
    with zipfile.ZipFile(outer) as zf:
        zf.extractall(extract)
    for nested in extract.rglob("*.zip"):
        target = nested.parent / nested.stem
        if not target.exists():
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(target)
    found = list(extract.rglob("features.txt"))
    if not found:
        raise FileNotFoundError("features.txt not found in the downloaded archive")
    return found[0].parent


def unique(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}__dup{seen[n]}")
    return out


def clean(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]+", "_", name)).strip("_")


def load(root: Path):
    raw = pd.read_csv(root / "features.txt", sep=r"\s+", header=None)[1].tolist()
    names = unique([clean(n) for n in unique(raw)])
    labels = pd.read_csv(root / "activity_labels.txt", sep=r"\s+", header=None)
    activities = labels[1].tolist()                       # index 0..5 -> class 1..6

    def part(s):
        X = pd.read_csv(root / s / f"X_{s}.txt", sep=r"\s+", header=None).to_numpy(np.float32)
        y = pd.read_csv(root / s / f"y_{s}.txt", header=None)[0].to_numpy() - 1   # 0-based
        g = pd.read_csv(root / s / f"subject_{s}.txt", header=None)[0].to_numpy()
        return X, y, g

    Xa, ya, ga = part("train")
    Xb, yb, gb = part("test")
    return np.vstack([Xa, Xb]), np.concatenate([ya, yb]), np.concatenate([ga, gb]), names, activities


# ----------------------------------------------------------------------------- helpers
def lr_pipeline():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=2500, random_state=RANDOM_STATE))])


def lgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          random_state=RANDOM_STATE, verbose=-1, n_jobs=1)


def rf(n=300):
    return RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)


def size_mb(model) -> float:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.tell() / 1024**2


def latency_ms(model, X_single, X_batch, n_single=300, n_batch=30):
    """Median CPU latency (ms) for one window and for a batch, after warm-up."""
    for _ in range(20):
        model.predict(X_single)
    model.predict(X_batch)
    single = []
    for _ in range(n_single):
        t0 = time.perf_counter_ns(); model.predict(X_single); single.append((time.perf_counter_ns() - t0) / 1e6)
    batch = []
    for _ in range(n_batch):
        t0 = time.perf_counter_ns(); model.predict(X_batch); batch.append((time.perf_counter_ns() - t0) / 1e6)
    return float(np.median(single)), float(np.median(batch))


def native_lr_latency_us(pipe, x, n=20000):
    """Latency of the same Logistic Regression as plain NumPy maths (scale -> dot -> argmax),
    i.e. what the model costs without scikit-learn's per-call overhead."""
    sc, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    mean, scale = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
    W, b = clf.coef_.T.astype(np.float32), clf.intercept_.astype(np.float32)
    x = x.ravel()
    for _ in range(1000):
        int(np.argmax(((x - mean) / scale) @ W + b))
    t = []
    for _ in range(n):
        t0 = time.perf_counter_ns(); int(np.argmax(((x - mean) / scale) @ W + b)); t.append(time.perf_counter_ns() - t0)
    return float(np.median(t) / 1e3)


def grouped_cv_f1(make_model, X, y, g, cv):
    scores = []
    for tr, va in cv.split(X, y, g):
        m = make_model().fit(X[tr], y[tr])
        scores.append(f1_score(y[va], m.predict(X[va]), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None, help='local "UCI HAR Dataset" folder')
    args = ap.parse_args()

    root = args.data_dir or download_dataset(Path(__file__).parent / ".data")
    X, y, g, names, activities = load(root)
    print(f"Loaded {X.shape[0]} windows x {X.shape[1]} features from {len(np.unique(g))} subjects")

    test = np.isin(g, TEST_SUBJECTS)
    Xtr, ytr, gtr, Xte, yte, gte = X[~test], y[~test], g[~test], X[test], y[test], g[test]
    assert set(gtr).isdisjoint(set(gte))
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    # Feature ranking learned on the training subjects only
    t0 = time.time()
    rf_full = rf().fit(Xtr, ytr)
    rf_time = time.time() - t0
    order = np.argsort(rf_full.feature_importances_)[::-1]

    # Edge trade-off curve: grouped-CV Macro F1 of Logistic Regression vs. feature budget.
    # NOTE: the ranking is computed once on all training subjects (slightly optimistic for CV).
    curve = []
    for k in FEATURE_COUNTS:
        idx = order[:k]
        m, s = grouped_cv_f1(lr_pipeline, Xtr[:, idx], ytr, gtr, cv)
        curve.append({"features": k, "cv_macro_f1_mean": m, "cv_macro_f1_std": s})
        print(f"  LR top-{k:>3}: grouped-CV Macro F1 {m:.4f} ± {s:.4f}")

    top_k = order[:EDGE_K]
    specs = {
        "Logistic Regression": (lr_pipeline, None),
        f"Logistic Regression (top {EDGE_K})": (lr_pipeline, top_k),
        "LightGBM": (lgbm, None),
        f"LightGBM (top {EDGE_K})": (lgbm, top_k),
        "Random Forest": (None, None),
        f"Random Forest (top {EDGE_K})": (rf, top_k),
    }

    results, fitted, confusion, reports = [], {}, {}, {}
    for name, (make, idx) in specs.items():
        cols = slice(None) if idx is None else idx
        if name == "Random Forest":
            model, train_time = rf_full, rf_time
        else:
            t0 = time.time(); model = make().fit(Xtr[:, cols], ytr); train_time = time.time() - t0
        pred = model.predict(Xte[:, cols])
        single, batch = latency_ms(model, Xte[:1, cols], Xte[:1000, cols])
        results.append({
            "model": name, "family": name.split(" (")[0], "features": int(Xtr[:, cols].shape[1]),
            "test_macro_f1": float(f1_score(yte, pred, average="macro")),
            "test_accuracy": float(accuracy_score(yte, pred)),
            "train_time_s": float(train_time), "size_mb": size_mb(model),
            "latency_single_ms": single, "latency_batch1000_ms": batch,
        })
        fitted[name] = (model, idx)
        confusion[name] = confusion_matrix(yte, pred, labels=range(6)).tolist()
        reports[name] = classification_report(yte, pred, labels=range(6), target_names=activities,
                                              output_dict=True, zero_division=0)
        print(f"  {name:<32} F1 {results[-1]['test_macro_f1']:.4f}  acc {results[-1]['test_accuracy']:.4f}  "
              f"{results[-1]['size_mb']:.3f} MB  {single:.3f} ms")

    # Grouped CV estimate for the recommended model (validation before the single test look)
    lr_cv_mean, lr_cv_std = grouped_cv_f1(lr_pipeline, Xtr, ytr, gtr, cv)

    lr = fitted["Logistic Regression"][0]
    pred_lr = lr.predict(Xte)
    per_subject = [{"subject": int(s), "windows": int((gte == s).sum()),
                    "accuracy": float(accuracy_score(yte[gte == s], pred_lr[gte == s])),
                    "macro_f1": float(f1_score(yte[gte == s], pred_lr[gte == s], average="macro"))}
                   for s in TEST_SUBJECTS]

    # Persist artifacts for the app (small models only; Random Forests are ~20 MB and not needed)
    ART.mkdir(exist_ok=True)
    (ART / "models").mkdir(exist_ok=True)
    for key, fname in [("Logistic Regression", "lr_full.joblib"),
                       (f"Logistic Regression (top {EDGE_K})", f"lr_top{EDGE_K}.joblib"),
                       ("LightGBM", "lgbm_full.joblib")]:
        joblib.dump(fitted[key][0], ART / "models" / fname, compress=3)
    np.savez_compressed(ART / "test_windows.npz", X=Xte.astype(np.float32), y=yte, subject=gte)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": {"name": "UCI Human Activity Recognition Using Smartphones", "doi": "10.24432/C54S4K",
                    "windows": int(X.shape[0]), "features": int(X.shape[1]), "subjects": int(len(np.unique(g))),
                    "train_windows": int(len(ytr)), "test_windows": int(len(yte)),
                    "test_subjects": TEST_SUBJECTS, "window_seconds": 2.56, "overlap": 0.5,
                    "sampling_hz": 50, "class_counts": {a: int((y == i).sum()) for i, a in enumerate(activities)}},
        "activities": activities,
        "edge_k": EDGE_K,
        "top_features": [{"feature": names[i], "importance": float(rf_full.feature_importances_[i])}
                         for i in order[:EDGE_K]],
        "top_k_indices": [int(i) for i in top_k],
        "results": results,
        "lr_grouped_cv": {"mean": lr_cv_mean, "std": lr_cv_std, "folds": 4},
        "feature_curve": curve,
        "native_lr_latency_us": native_lr_latency_us(lr, Xte[:1]),
        "lr_ops_per_window": int(Xtr.shape[1] * 6 * 2 + Xtr.shape[1] * 2),
        "confusion": confusion,
        "reports": reports,
        "per_subject": per_subject,
        "environment": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "lightgbm": lightgbm.__version__},
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nArtifacts written to {ART.resolve()}")


if __name__ == "__main__":
    main()
