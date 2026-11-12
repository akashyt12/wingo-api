# Wingo 1M Prediction API

FastAPI service that returns **BIG/SMALL** and **COLOUR** predictions for the next
1-minute Wingo draw, using a trained `WingoNet` checkpoint with a pure-NumPy
inference fallback (torch is optional).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info + endpoint list |
| GET | `/health` | Liveness + model backend |
| GET | `/predict` | Current prediction (cached; `?fresh=true` refetches) |
| GET | `/history?limit=50` | Last N normalised draws (1–500) |
| GET | `/ui` | Minimal auto-refreshing dashboard |
| GET | `/docs` | Swagger UI |

### Example response (`/predict`)

```json
{
  "status": "ok",
  "period": "2026111210140",
  "prediction": {
    "size": "SMALL",
    "big_small": "SMALL",
    "colour": "GREEN",
    "colours": ["GREEN"],
    "violet_alert": false
  },
  "probabilities": {
    "big": 0.4921, "small": 0.5079,
    "red": 0.411, "green": 0.523, "violet": 0.181,
    "size_z": -0.99
  },
  "latest_result": {"period": "2026111210139", "number": 8, "size": "BIG", "colour": "RED"},
  "rule_engine": {"pred": null, "confidence": 0.0, "votes": []},
  "model": {"backend": "numpy-native (no torch required)", "features": 93},
  "upstream_records": 1000,
  "generated_at": "2026-11-12T10:13:00+00:00"
}
```

## Run locally

```bash
cd wingo-api
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:
- Dashboard: http://localhost:8000/ui
- Swagger: http://localhost:8000/docs
- Prediction: http://localhost:8000/predict

## Deploy from GitHub

### Render (free, easiest)

1. Push this folder to a new GitHub repo (see below).
2. Render → **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add env var `WINGO_FORCE_NUMPY=1` (keeps RAM low, no torch).
6. Deploy. Or simply import the included `render.yaml` as a Blueprint.

### Railway / Fly / any Docker host

`Procfile` and `Dockerfile` are included. On Railway just connect the repo; it
detects the `Procfile`. For Docker: `docker build -t wingo-api . && docker run -p 8000:8000 wingo-api`.

### Push to GitHub

```bash
cd wingo-api
git init
git add .
git commit -m "Wingo 1M prediction API"
git branch -M main
git remote add origin https://github.com/<you>/wingo-api.git
git push -u origin main
```

`models/best.pt` is ~19 MB, under GitHub's 100 MB file limit, so it commits
normally. If you swap in a larger model, use Git LFS or download it at build time.

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `WINGO_FORCE_NUMPY` | `0` | `1` = never import torch (lightweight) |
| `MODEL_PATH` | `models/best.pt` | Checkpoint path |
| `WINGO_API_URL` | built-in 1M history URL | Upstream draw source |
| `CACHE_TTL` | `3` | Seconds to cache predictions |
| `PORT` | `8000` | Server port (set by host) |

## How it works

1. Fetch the latest 1,000 draws from the upstream history API.
2. Build the exact 128-token window + 93 causal features used in training.
3. Run `WingoNet` (NumPy forward; torch used automatically if installed).
4. Optional structural rule-engine consensus (BIG/SMALL) is attached as `rule_engine`.

## Accuracy note

Out-of-sample test accuracy of the shipped model is **50.88%** on BIG/SMALL.
Wingo's 1.96x payout break-even is **51.02%**, so the model is close to — not
reliably above — break-even. Treat predictions as experimental; this is not
financial advice or a guaranteed winning system.
