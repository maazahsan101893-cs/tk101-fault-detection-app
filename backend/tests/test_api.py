import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, engine

client = TestClient(app)

# Consolidated test-split event counts reported in the ChemicalE README
EXPECTED_EVENTS = {
    "pca_t2_q": 11,
    "lstm_autoencoder": 12,
    "local_outlier_factor": 19,
    "isolation_forest": 26,
}


@pytest.mark.parametrize("key", list(EXPECTED_EVENTS))
def test_scores_match_training_pipeline(key):
    det = engine.detectors[key]
    full = np.vstack([engine.warmup_values, engine.values])
    scores = det.score(engine._scaled(det, full), len(engine.warmup_values))
    ref = np.load(engine.artifacts / "results" / f"{key}_reference_test_scores.npy")
    # LSTM rows 0-8 legitimately differ: we use real history instead of padding
    assert np.allclose(scores[9:], ref[9:], atol=1e-2)


@pytest.mark.parametrize("key,count", EXPECTED_EVENTS.items())
def test_full_test_split_event_counts(key, count):
    r = client.post("/api/analyze/sample", json={"model": key, "offset_hours": 0, "hours": 324})
    assert r.status_code == 200
    assert r.json()["results"][0]["summary"]["events"] == count


def test_overview_and_models():
    o = client.get("/api/overview").json()
    assert o["best_model"] == "pca_t2_q"
    assert len(o["leaderboard"]) == 4
    assert len(client.get("/api/models").json()) == 4


def test_all_models_window():
    r = client.post("/api/analyze/sample", json={"model": "all", "offset_hours": 120, "hours": 24})
    res = r.json()["results"]
    assert [x["model"] for x in res] == list(EXPECTED_EVENTS)
    assert all(len(x["series"]) <= 1500 for x in res)


def test_upload_roundtrip():
    csv_text = client.get("/api/sample.csv").text
    r = client.post(
        "/api/analyze/upload",
        files={"file": ("s.csv", csv_text.encode(), "text/csv")},
        data={"model": "pca_t2_q"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"][0]["summary"]["rows"] == 1440
    assert body["results"][0]["events"]


def test_upload_rejects_missing_columns():
    r = client.post(
        "/api/analyze/upload",
        files={"file": ("bad.csv", b"timestamp,FOO\n2024-01-01 00:00:00,1\n", "text/csv")},
        data={"model": "pca_t2_q"},
    )
    assert r.status_code == 422
    assert "Missing" in r.json()["detail"]


def test_bad_window():
    r = client.post("/api/analyze/sample", json={"model": "pca_t2_q", "offset_hours": 500, "hours": 24})
    assert r.status_code == 422
