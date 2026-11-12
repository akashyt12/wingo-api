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

# --- Win/Loss session tracking (in-memory; resets on cold start) ---
_pending: dict[str, dict[str, Any]] = {}          # period -> pick
_session: dict[str, Any] = {
    "size_wins": 0, "size_losses": 0,
    "col_wins": 0, "col_losses": 0,
    "total": 0, "recent": [],
}
_MAX_RESOLVED = 50


def _resolve_pending(records: list[dict]) -> None:
    """Score any pending picks whose period has now completed."""
    by_period = {r["issueNumber"]: r for r in records}
    now = time.time()
    for period in list(_pending):
        pick = _pending[period]
        rec = by_period.get(period)
        if rec is None:
            if now - pick.get("ts", now) > 400:
                _pending.pop(period, None)
            continue
        actual_size = "BIG" if rec["size"] == 1 else "SMALL"
        actual_cols = [
            c for c, f in (
                ("RED", rec["red"]), ("GREEN", rec["green"]), ("VIOLET", rec["violet"])
            ) if f
        ]
        size_hit = pick["size"] == actual_size
        col_hit = pick["colour"] in actual_cols
        _session["size_wins"] += int(size_hit)
        _session["size_losses"] += int(not size_hit)
        _session["col_wins"] += int(col_hit)
        _session["col_losses"] += int(not col_hit)
        _session["total"] += 1
        _session["recent"].insert(0, {
            "period": period,
            "size": pick["size"],
            "colour": pick["colour"],
            "size_hit": size_hit,
            "col_hit": col_hit,
            "actual_number": rec["number"],
            "actual_size": actual_size,
            "actual_colour": "+".join(actual_cols),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        })
        _pending.pop(period, None)
    _session["recent"] = _session["recent"][:_MAX_RESOLVED]


def _session_block() -> dict[str, Any]:
    total = _session["total"]
    return {
        "total": total,
        "size_wins": _session["size_wins"],
        "size_losses": _session["size_losses"],
        "col_wins": _session["col_wins"],
        "col_losses": _session["col_losses"],
        "size_acc": round(_session["size_wins"] / total * 100, 1) if total else None,
        "col_acc": round(_session["col_wins"] / total * 100, 1) if total else None,
        "recent": _session["recent"][:10],
    }


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

    # Score completed picks, then register the new one.
    _resolve_pending(records)
    next_issue = meta["next_issue"]
    _pending[next_issue] = {
        "period": next_issue,
        "size": pred["pred_size"],
        "colour": pred["primary_colour"],
        "ts": time.time(),
    }

    last = _session["recent"][0] if _session["recent"] else None
    last_pick = None
    if last:
        last_pick = {
            "period": last["period"],
            "size": last["size"],
            "colour": last["colour"],
            "size_hit": last["size_hit"],
            "col_hit": last["col_hit"],
            "size_mark": "\u2713" if last["size_hit"] else "\u2717",
            "col_mark": "\u2713" if last["col_hit"] else "\u2717",
            "actual_number": last["actual_number"],
            "actual_size": last["actual_size"],
            "actual_colour": last["actual_colour"],
        }

    return {
        "status": "ok",
        "period": next_issue,
        "prediction": {
            "size": pred["pred_size"],
            "big_small": pred["pred_size"],
            "colour": pred["primary_colour"],
            "colours": pred["multilabel_colours"].split("+"),
            "violet_alert": pred["violet_alert"],
        },
        "last_pick": last_pick,
        "session": _session_block(),
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
  .card{width:360px;border:1px solid #2a2f3a;border-radius:14px;background:#161b22;
        padding:18px 20px;box-shadow:0 8px 30px rgba(0,0,0,.5)}
  h1{font-size:18px;letter-spacing:3px;text-align:center;margin:0 0 12px;color:#8ab4ff}
  .row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;font-size:14px}
  .sep{border-top:1px dashed #2a2f3a;margin:8px 0}
  .label{color:#8b949e}
  .big{color:#3fb950;font-weight:700}.small{color:#f0883e;font-weight:700}
  .red{color:#f85149}.green{color:#3fb950}.violet{color:#bc8cff}
  .win{color:#3fb950;font-weight:700}.loss{color:#f85149;font-weight:700}
  .issue{font-size:20px;font-weight:700;letter-spacing:1px}
  .muted{color:#6e7681;font-size:11px;text-align:center;margin-top:10px}
  .err{color:#f85149;text-align:center;font-size:12px}
  .marks{letter-spacing:2px;font-size:15px}
</style></head><body>
<div class="card">
  <h1>1MIN WINGO</h1>
  <div class="row"><span class="label">Latest</span><span id="latest">--</span></div>
  <div class="sep"></div>
  <div class="row"><span class="label">LAST PICK</span>
    <span><span id="lp">--</span> <span id="lsm"></span>
    &nbsp;&nbsp;Colour: <span id="lc">--</span> <span id="lcm"></span></span></div>
  <div class="row"><span class="label">NEXT PREDICTION</span><span></span></div>
  <div class="row"><span class="issue" id="period">-----</span><span id="size" class="big">--</span></div>
  <div class="row"><span class="label">Colour</span><span id="colour">--</span></div>
  <div class="sep"></div>
  <div class="row"><span class="label">Session</span><span id="sess">--</span></div>
  <div class="row"><span class="label">Recent</span><span class="marks" id="recent"></span></div>
  <div class="muted" id="foot">loading...</div>
</div>
<script>
const $=id=>document.getElementById(id);
const cls=v=>v==='BIG'?'big':'small';
const ccls=v=>v==='RED'?'red':v==='GREEN'?'green':'violet';
async function tick(){
  try{
    const r=await fetch('/predict?fresh=true',{cache:'no-store'});
    if(!r.ok) throw new Error('HTTP '+r.status);
    const d=await r.json();
    $('latest').textContent=`${d.latest_result.period.slice(-5)} ${d.latest_result.number} ${d.latest_result.size}`;
    if(d.last_pick){
      $('lp').textContent=`${d.last_pick.period.slice(-5)} - ${d.last_pick.size}`;
      $('lsm').textContent=d.last_pick.size_mark;
      $('lsm').className=d.last_pick.size_hit?'win':'loss';
      $('lc').textContent=d.last_pick.colour;
      $('lcm').textContent=d.last_pick.col_mark;
      $('lcm').className=d.last_pick.col_hit?'win':'loss';
    }
    $('period').textContent=d.period.slice(-5);
    $('size').textContent=d.prediction.size;
    $('size').className=cls(d.prediction.size);
    $('colour').textContent=d.prediction.colour;
    $('colour').className=ccls(d.prediction.colour);
    const s=d.session;
    if(s.total>0){
      $('sess').textContent=`Size ${s.size_wins}/${s.total} (${s.size_acc}%) | Colour ${s.col_wins}/${s.total} (${s.col_acc}%)`;
      $('recent').innerHTML=s.recent.map(x=>`<span class="${x.size_hit?'win':'loss'}">${x.size_hit?'\\u2713':'\\u2717'}</span>`).join('');
    }
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
