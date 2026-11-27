"""
Package the trained models and a compact copy of the 15% test split into
backend/artifacts so the API can run without the 100 MB raw CSV.

Usage:
    python scripts/build_artifacts.py /path/to/ChemicalE

Re-run this whenever you retrain a model in the ChemicalE repo.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

WARMUP_ROWS = 1440  # 24 h of validation data so the 1440-min rolling UA feature is exact

MODELS = {
    "pca_t2_q": ("pca_t2_q_checkpoint.npz", "pca_t2_q_scores.json"),
    "lstm_autoencoder": ("lstm_autoencoder_checkpoint.npz", "lstm_autoencoder_scores.json"),
    "local_outlier_factor": ("lof_checkpoint.npz", "local_outlier_factor_scores.json"),
    "isolation_forest": ("isolation_forest_checkpoint.npz", "isolation_forest_scores.json"),
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        ts, rows = [], []
        for row in reader:
            ts.append(row[0])
            rows.append([float(x) if x != "" else np.nan for x in row[1:]])
    return header[1:], ts, np.array(rows, dtype=np.float64)


def main(repo: Path) -> None:
    training = repo / "training"
    out = Path(__file__).resolve().parent.parent / "backend" / "artifacts"
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    for key, (ckpt, scores) in MODELS.items():
        src = training / "models" / key
        shutil.copy2(src / ckpt, out / "models" / f"{key}.npz")
        shutil.copy2(src / scores, out / "results" / f"{key}_scores.json")

    for name in ["model_comparison.csv", "evaluation_summary.json"]:
        shutil.copy2(training / "results" / name, out / "results" / name)
    shutil.copy2(training / "models" / "checkpoints_manifest.json", out / "results" / "checkpoints_manifest.json")
    shutil.copy2(training / "splits" / "split_metadata.json", out / "results" / "split_metadata.json")
    shutil.copy2(
        training / "models" / "lstm_autoencoder" / "gradient_descent_comparison.json",
        out / "results" / "gradient_descent_comparison.json",
    )

    tags, ts_val, X_val = read_csv(training / "splits" / "val_15.csv")
    tags_t, ts_test, X_test = read_csv(training / "splits" / "test_15.csv")
    assert tags == tags_t

    np.savez_compressed(
        out / "test_window.npz",
        tag_names=np.array(tags),
        warmup_timestamps=np.array(ts_val[-WARMUP_ROWS:]),
        warmup_values=X_val[-WARMUP_ROWS:],
        timestamps=np.array(ts_test),
        values=X_test,
    )

    # Reference scores from the original pipeline, used by the test suite
    for key in MODELS:
        src = training / "models" / key / f"{key}_test_scores.csv"
        with src.open(newline="") as f:
            scores = [float(r["anomaly_score"]) for r in csv.DictReader(f)]
        np.save(out / "results" / f"{key}_reference_test_scores.npy", np.array(scores))

    print(json.dumps({"artifacts": str(out), "test_rows": len(ts_test), "tags": len(tags)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve())
