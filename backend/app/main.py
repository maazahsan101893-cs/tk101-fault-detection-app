import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .service import Engine, InputError

ModelKey = Literal["pca_t2_q", "lstm_autoencoder", "local_outlier_factor", "isolation_forest", "all"]
# Vercel caps request bodies at 4.5 MB; Docker/local allow more
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "4.4" if os.environ.get("VERCEL") else "25"))

app = FastAPI(
    title="TK-101 Fault Detection API",
    description="Historian-based fault detection and root-cause analysis for the TK-101 steam-heated hold tank.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Engine()


def _keys(model: str) -> list[str]:
    return list(engine.detectors) if model == "all" else [model]


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": list(engine.detectors)}


@app.get("/api/overview")
def overview():
    """Leaderboard, dataset splits and gradient-descent study from the training repo."""
    leaderboard = []
    for entry in engine.results["manifest"]["checkpoints"]:
        key = entry["model_key"]
        scores = engine.results["models"][key]
        det = engine.detectors[key]
        leaderboard.append(
            {
                **entry,
                "display_name": det.name,
                "family": det.family,
                "test_precision": scores["testing_split_15pct"]["precision"],
                "test_recall": scores["testing_split_15pct"]["recall"],
                "test_accuracy": scores["testing_split_15pct"]["accuracy"],
                "test_events": scores["testing_split_15pct"]["candidate_fault_events"],
                "test_snr": scores["testing_split_15pct"]["signal_to_noise_ratio"],
            }
        )
    return {
        "best_model": engine.results["manifest"]["best_model_key"],
        "leaderboard": leaderboard,
        "splits": engine.results["splits"],
        "gradient_descent": engine.results["gradient_descent"],
        "features": engine.detectors["pca_t2_q"].features,
        "test_window": engine.window_info(),
    }


@app.get("/api/models")
def models():
    return [
        {
            "key": k,
            "name": d.name,
            "family": d.family,
            "threshold": d.threshold,
            "contribution_method": d.contribution_method,
            "config": d.config(),
            "metrics": engine.results["models"][k],
        }
        for k, d in engine.detectors.items()
    ]


class SampleRequest(BaseModel):
    model: ModelKey = "pca_t2_q"
    offset_hours: float = Field(0, ge=0, description="Hours after the start of the test split")
    hours: float = Field(72, gt=0, le=324, description="Window length in hours")


@app.post("/api/analyze/sample")
def analyze_sample(req: SampleRequest):
    """Score a window of the held-out 15% test split (2024-05-16 → 2024-05-30)."""
    try:
        return engine.analyze_sample(req.offset_hours, req.hours, tuple(_keys(req.model)))
    except InputError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...), model: ModelKey = Form("pca_t2_q")):
    """Score your own historian CSV (same tag names as historian_60s.csv, 1-minute rows)."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB:g} MB.")
    try:
        X, ts, warnings = engine.parse_upload(content)
        return engine.analyze(X, ts, 0, _keys(model), warnings)
    except InputError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/sample.csv", response_class=PlainTextResponse)
def sample_csv(offset_hours: float = Query(126, ge=0, le=320), hours: float = Query(24, gt=0, le=48)):
    """A slice of real test data you can download and re-upload to try the upload flow."""
    return PlainTextResponse(
        engine.sample_csv(offset_hours, hours),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tk101_sample.csv"},
    )


@app.get("/api/template.csv", response_class=PlainTextResponse)
def template_csv():
    return PlainTextResponse(
        engine.template_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tk101_template.csv"},
    )


def _frontend_dir() -> str | None:
    """Built React app: STATIC_DIR in Docker, frontend/dist on Vercel or locally."""
    if os.environ.get("STATIC_DIR"):
        return os.environ["STATIC_DIR"] if Path(os.environ["STATIC_DIR"]).is_dir() else None
    if Path("frontend/dist").is_dir():  # Vercel runs from the project root
        return "frontend/dist"
    repo_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    return str(repo_dist) if repo_dist.is_dir() else None


# Serve the React build for every non-/api path. API routes always take priority.
# On Vercel these files are promoted to the CDN at build time.
_frontend = _frontend_dir()
if _frontend:
    app.frontend("/", directory=_frontend)
