# TK-101 Fault Detection & Root Cause Analysis: Web App

A full-stack web app for the [ChemicalE](https://github.com/Hashwar-11/ChemicalE) ML pipeline project
(historian-based fault detection for the TK-101 steam-heated hydrolysis hold tank).

- **Backend:** FastAPI + NumPy. Loads the four trained checkpoints (`.npz`) and reproduces the exact scoring from the
  training scripts: PCA T²+Q, LSTM Autoencoder, Local Outlier Factor and Isolation Forest. No scikit-learn or PyTorch needed.
- **Frontend:** React + TypeScript + Vite + Recharts. Includes a model leaderboard, an interactive fault-detection view,
  root-cause tag contributions and per-model metrics.
- **Deploy:** Vercel (FastAPI framework preset: one Python function, React build served from the CDN) **or** a single Docker container.

The backend's test suite checks that live scoring reproduces the training pipeline: the full test split yields
**11 / 12 / 19 / 26** fault events for PCA / LSTM-AE / LOF / IF, matching the project README.

---

## What the app does

| Page | What you get |
| --- | --- |
| **Overview** | Best-model KPIs, full leaderboard (RMSE, ROC-AUC, precision, recall, F1, composite), 70/15/15 split, gradient-descent study (BGD vs SGD vs Mini-batch), the 40 model features |
| **Fault Detection** | Pick a window of the held-out test split (6 h → full 13.5 days) **or upload your own historian CSV**. Then pick a detector or *Compare all*. Shows the anomaly-score timeline with the 99% limit, consolidated fault events, and root-cause tags for each event (click a row) |
| **Models** | Loss / cost / RMSE / classification metrics per split, best epoch, and configuration for each detector |
| **API docs** | Interactive Swagger UI at `/api/docs` |

Fault events use the same rules as training: alarms of 2 min or less are dropped, and gaps of 15 min or less are merged.

---

## Project structure

```
tk101-app/
├── pyproject.toml            # Vercel: Python deps, entrypoint, frontend build script
├── vercel.json               # Vercel: function duration + excluded files
├── Dockerfile                # single container: API + built frontend
├── docker-compose.yml
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI routes
│   │   ├── service.py        # analysis engine, event consolidation, CSV parsing
│   │   ├── detectors.py      # numpy inference for the 4 models
│   │   └── features.py       # physics feature engineering (ported 1:1)
│   ├── artifacts/            # checkpoints, metrics JSON, compact test split (8 MB)
│   ├── tests/test_api.py
│   └── requirements.txt
├── frontend/                 # React app (src/pages, src/components)
└── scripts/build_artifacts.py  # regenerate backend/artifacts from the ChemicalE repo
```

---

## Run locally (development)

```bash
# 1. Backend  → http://localhost:8000/api/docs
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 2. Frontend → http://localhost:5173  (proxies /api to :8000)
cd frontend
npm install
npm run dev
```

Run the tests: `cd backend && pip install pytest httpx && pytest -q`

## Run with Docker

```bash
docker compose up --build
# open http://localhost:8000
```

The same image runs on Railway, Render, Fly.io or Google Cloud Run. Point the host at the root `Dockerfile`; it listens on
`$PORT` (default 8000). Docker allows uploads up to 25 MB (`MAX_UPLOAD_MB`).

## Deploy to Vercel

Vercel detects this repo as a **FastAPI** project and deploys it as a single Vercel Function. `pyproject.toml` sets up everything:
- `tool.vercel.entrypoint`: the app lives at `backend/app/main.py`.
- `tool.vercel.scripts.build`: builds the React dashboard into `frontend/dist`.
- `app.frontend("/")` in `main.py`: Vercel serves that build from its CDN. `/api/*` goes to the function.

Vercel does not run the Dockerfile; that is for local use and container hosts.

**Option A: GitHub import (recommended)**
1. Push this repo to GitHub.
2. On [vercel.com/new](https://vercel.com/new), import it. Framework Preset should show **FastAPI** (select it if not).
3. Leave Root Directory as the repo root. Leave the Build, Output and Install Command overrides **off**.
4. Click **Deploy**. Every push to `main` redeploys automatically.

**Option B: Vercel CLI**
```bash
npm i -g vercel
vercel          # first run links the project (preview deploy)
vercel --prod   # production deploy
```

**Limits on Vercel**
- Request bodies are capped at **4.5 MB**, so CSV uploads are limited to about 4–5 days of 1-minute data.
  The bundled test split has no limit, since it runs server-side. For larger files, use the Docker deployment.
- `maxDuration` is set to 60 s. A full 13.5-day *Compare all* run takes about 2 s.
- Python 3.12 (set by `requires-python` in `pyproject.toml`).

Optional environment variables:

| Variable | Where | Default | Purpose |
| --- | --- | --- | --- |
| `VITE_API_URL` | frontend build | *(same origin)* | Point the UI at a backend on another domain (e.g. Vercel frontend + Railway backend) |
| `VITE_MAX_UPLOAD_MB` | frontend build | `4` | Client-side upload size check |
| `MAX_UPLOAD_MB` | backend | `25` (Docker) / `4.4` (on Vercel, auto) | Server-side upload limit |
| `CORS_ORIGINS` | backend | `*` | Comma-separated allowed origins |

---

## Uploading your own data

CSV with a `timestamp` column plus the historian tag columns from `historian_60s.csv`, one row per minute.
- Download `/api/sample.csv` (24 h of real test data) to see the format, or `/api/template.csv` for headers only.
- Only the TK-101 tags are required. Utility-skid tags (`CW-*`, `N2-*`, `SD-*`, `AIR-*`) are optional.
- Empty cells are filled with training medians, exactly as in training.
- With less than 24 h of data, the coil-fouling feature `UA_ROLLING_DIFF` uses a shorter baseline, and the app warns you.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/health` | Liveness + loaded models |
| GET | `/api/overview` | Leaderboard, splits, gradient-descent study, features |
| GET | `/api/models` | Per-model config and metrics |
| POST | `/api/analyze/sample` | `{"model": "pca_t2_q" \| ... \| "all", "offset_hours": 0, "hours": 72}` |
| POST | `/api/analyze/upload` | multipart: `file` (CSV), `model` |
| GET | `/api/sample.csv`, `/api/template.csv` | Example data / blank template |

## After retraining

If you retrain models in the ChemicalE repo, refresh the app's artifacts:

```bash
python scripts/build_artifacts.py /path/to/ChemicalE
cd backend && pytest -q    # update EXPECTED_EVENTS in tests if event counts change
```
