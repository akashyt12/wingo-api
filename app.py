#!/usr/bin/env python3
"""Wingo 1M Prediction API (FastAPI).

Serves BIG/SMALL + COLOUR predictions for the upcoming 1-minute Wingo draw.

Endpoints
---------
GET /            -> service info + endpoint list
GET /health      -> liveness + model backend
GET /predict     -> current prediction (cached, ?fresh=true to force)
GET /history     -> last N normalised draws (?limit=50)
GET /ui          -> minimal auto-refreshing dashboard
GET /docs        -> Swagger UI (auto-generated)

Deploy: any host that runs `uvicorn app:app`. NumPy inference means torch is
optional, so it fits small free-tier containers.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Force the lightweight pure-NumPy backend when requested (no torch import).
if os.getenv("WINGO_FORCE_NUMPY", "0") == "1":
    sys.modules["torch"] = None  # type: ignore[assignment]

import predictor as P  # noqa: E402

P.RULE_ENGINE_DIR = BASE
RULE_STATUS = P.init_rule_engine()

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE / "models" / "best.pt")))
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")

ENGINE = P.WingoPredictorEngine(MODEL_PATH)
UPSTREAM = os.getenv("WINGO_API_URL", P.DEFAULT_API_URL)
CACHE_TTL = float(os.getenv("CACHE_TTL", "3"))

app = FastAPI(
    title="Wingo 1M Prediction API",
    version="1.0.0",
    description="BIG/SMALL + COLOUR predictions for the next Wingo 1-minute draw.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()
_state: dict[str, Any] = {"ts": 0.0, "data": None}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


def _build_payload(records: list[dict]) -> dict[str, Any]:
    tok, feat, meta = P.prepare_live_inputs(records)
    pred = ENGINE.predict(tok, feat)
    rule = P.rule_predict(records)
    return {
        "status": "ok",
        "period": meta["next_issue"],
        "prediction": {
            "size": pred["pred_size"],
            "big_small": pred["pred_size"],
            "colour": pred["primary_colour"],
            "colours": pred["multilabel_colours"].split("+"),
            "violet_alert": pred["violet_alert"],
        },
        "probabilities": {
            "big": round(pred["p_big"], 6),
            "small": round(pred["p_small"], 6),
            "red": round(pred["p_red"], 6),
            "green": round(pred["p_green"], 6),
            "violet": round(pred["p_violet"], 6),
            "size_z": round(pred["z_size"], 4),
        },
        "latest_result": {
            "period": meta["latest_issue"],
            "number": meta["latest_number"],
            "size": meta["latest_size"],
            "colour": meta["latest_colour"],
        },
        "rule_engine": rule,
        "rule_engine_status": RULE_STATUS,
        "model": {"backend": ENGINE.backend, "features": ENGINE.n_feat},
        "upstream_records": meta["n_records"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_prediction(fresh: bool = False) -> tuple[dict[str, Any], bool, float]:
    now = time.time()
    with _lock:
        cached = _state["data"]
        age = now - _state["ts"]
        if not fresh and cached is not None and age < CACHE_TTL:
            return cached, True, age
        try:
            records = P.fetch_live_records(UPSTREAM)
            payload = _jsonable(_build_payload(records))
        except Exception as exc:  # noqa: BLE001
            if cached is not None:
                return cached, True, age
            raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {exc}")
        _state["data"] = payload
        _state["ts"] = time.time()
        return payload, False, 0.0


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Wingo 1M Prediction API",
        "version": "1.0.0",
        "model_backend": ENGINE.backend,
        "endpoints": ["/health", "/predict", "/history?limit=50", "/ui", "/docs"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_backend": ENGINE.backend,
        "features": ENGINE.n_feat,
        "model_path": str(MODEL_PATH.name),
        "rule_engine": RULE_STATUS,
        "cache_ttl_sec": CACHE_TTL,
    }


@app.get("/predict")
def predict(fresh: bool = Query(False, description="Bypass cache and refetch upstream")) -> JSONResponse:
    payload, cached, age = _get_prediction(fresh=fresh)
    return JSONResponse(content=payload, headers={
        "X-Cache": "HIT" if cached else "MISS",
        "X-Cache-Age": f"{age:.2f}",
    })


@app.get("/history")
def history(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    try:
        records = P.fetch_live_records(UPSTREAM)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {exc}")
    items = []
    for r in records[-limit:]:
        cols = [c for c, f in (("RED", r["red"]), ("GREEN", r["green"]), ("VIOLET", r["violet"])) if f]
        items.append({
            "period": r["issueNumber"],
            "number": r["number"],
            "size": "BIG" if r["size"] == 1 else "SMALL",
            "colours": cols,
        })
    return {"count": len(items), "draws": items}


_UI_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>1MIN WINGO</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{margin:0;background:#0e1117;color:#e6e6e6;font-family:Consolas,Menlo,monospace;
       display:flex;justify-content:center;align-items:center;min-height:100vh}
  .card{width:340px;border:1px solid #2a2f3a;border-radius:14px;background:#161b22;
        padding:18px 20px;box-shadow:0 8px 30px rgba(0,0,0,.5)}
  h1{font-size:18px;letter-spacing:3px;text-align:center;margin:0 0 12px;color:#8ab4ff}
  .row{display:flex;justify-content:space-between;padding:7px 0;font-size:14px}
  .sep{border-top:1px dashed #2a2f3a;margin:8px 0}
  .label{color:#8b949e}
  .big{color:#3fb950;font-weight:700}.small{color:#f0883e;font-weight:700}
  .red{color:#f85149}.green{color:#3fb950}.violet{color:#bc8cff}
  .issue{font-size:20px;font-weight:700;letter-spacing:1px}
  .muted{color:#6e7681;font-size:11px;text-align:center;margin-top:12px}
  .err{color:#f85149;text-align:center;font-size:12px}
</style></head><body>
<div class="card">
  <h1>1MIN WINGO</h1>
  <div class="row"><span class="label">Latest</span><span id="latest">--</span></div>
  <div class="sep"></div>
  <div class="row"><span class="label">NEXT PREDICTION</span><span></span></div>
  <div class="row"><span class="issue" id="period">-----</span><span id="size" class="big">--</span></div>
  <div class="row"><span class="label">Colour</span><span id="colour">--</span></div>
  <div class="sep"></div>
  <div class="row"><span class="label">P(BIG)</span><span id="pbig">--</span></div>
  <div class="muted" id="foot">loading...</div>
</div>
<script>
const $=id=>document.getElementById(id);
function cls(v){return v==='BIG'?'big':'small';}
function ccls(v){return v==='RED'?'red':v==='GREEN'?'green':'violet';}
async function tick(){
  try{
    const r=await fetch('/predict?fresh=true',{cache:'no-store'});
    if(!r.ok) throw new Error('HTTP '+r.status);
    const d=await r.json();
    $('latest').textContent=`${d.latest_result.period.slice(-5)} ${d.latest_result.number} ${d.latest_result.size}`;
    $('period').textContent=d.period.slice(-5);
    $('size').textContent=d.prediction.size;
    $('size').className=cls(d.prediction.size);
    $('colour').textContent=d.prediction.colour;
    $('colour').className=ccls(d.prediction.colour);
    $('pbig').textContent=(d.probabilities.big*100).toFixed(2)+'%';
    $('foot').textContent=`${d.model.backend} | ${new Date(d.generated_at).toLocaleTimeString()}`;
  }catch(e){$('foot').innerHTML='<span class="err">'+e.message+'</span>';}
}
tick(); setInterval(tick, 3000);
</script></body></html>"""


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return _UI_HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
