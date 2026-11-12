#!/usr/bin/env python3
"""Live 1-Minute Wingo AI Predictor (Full Standalone Runner).

Fetches the latest 1,000 Wingo 1M results from:
  https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000

Builds the exact 128-token window + 93 causal features used in training,
runs the trained WingoNet checkpoint (best.pt / final.pt) via PyTorch (GPU/CPU)
or Pure-NumPy fallback, outputs full SIZE (BIG/SMALL) + COLOUR (RED/GREEN/VIOLET)
predictions for the upcoming 1-minute draw, and tracks live Win/Loss accuracy
automatically every minute as new results arrive.

Usage:
  python live_wingo_predictor.py                        # Live 1-min continuous loop
  python live_wingo_predictor.py --once                 # Single live prediction
  python live_wingo_predictor.py --model best.pt        # Specify checkpoint path
  python live_wingo_predictor.py --test-local           # Offline self-test
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import pickle
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

# ==============================================================================
# Configuration & Constants
# ==============================================================================
DEFAULT_API_URL = (
    "https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000"
)
SEQ_LEN = 128
N_FEATURES = 93

# K2-style last-4 pattern lookup table (exact match with prepare_data.py)
PATTERN4 = {
    (1, 1, 1, 1): 1,  (1, 1, 1, 0): -1, (1, 1, 0, 1): 1,  (1, 1, 0, 0): -1,
    (1, 0, 1, 1): 1,  (1, 0, 1, 0): 1,  (1, 0, 0, 1): -1, (1, 0, 0, 0): -1,
    (0, 1, 1, 1): 1,  (0, 1, 1, 0): -1, (0, 1, 0, 1): 1,  (0, 1, 0, 0): -1,
    (0, 0, 1, 1): 1,  (0, 0, 1, 0): -1, (0, 0, 0, 1): -1, (0, 0, 0, 0): 1,
}

# Exact dragon_p and zig_p tables produced by prepare_data.py on the training split
TRAINED_DRAGON_P = {k: 1.0 for k in range(1, 20)}
TRAINED_ZIG_P = {k: 1.0 for k in range(1, 21)}

# Train-only smoothed P(BIG | last-4 size pattern) from prepare_data.py (data93/meta.json).
# LSB = oldest draw; index = SB[i-4]*1 + SB[i-3]*2 + SB[i-2]*4 + SB[i-1]*8.
TRAINED_PMAP = {
    0: 0.50112252534186, 1: 0.4990138067061144, 2: 0.4922931314223905,
    3: 0.5039652437762913, 4: 0.5095971884293052, 5: 0.4803502482073911,
    6: 0.5103964513446078, 7: 0.4959674686546933, 8: 0.49398082024076717,
    9: 0.4958190541466758, 10: 0.5231956986282484, 11: 0.48324478953821004,
    12: 0.5059667517417397, 13: 0.5052093973442288, 14: 0.5088444595052525,
    15: 0.5054670003952049,
}

# ==============================================================================
# 0. Rule-engine layer (BIG/SMALL structural rules mined from the dataset)
#    Optional: if the engine dir is missing, predictions still work.
# ==============================================================================
RULE_ENGINE_DIR = Path(__file__).resolve().parent
_RULE: dict[str, Any] = {"mod": None, "weights": None, "ids": None, "min_agree": 2}


def init_rule_engine() -> str:
    """Import wingo_engine + load calibrated weights/consensus config."""
    try:
        if str(RULE_ENGINE_DIR) not in sys.path:
            sys.path.insert(0, str(RULE_ENGINE_DIR))
        import wingo_engine as we  # type: ignore

        wj = json.loads((RULE_ENGINE_DIR / "engine_rules.json").read_text(encoding="utf-8"))
        _RULE["mod"] = we
        _RULE["weights"] = wj.get("weights")
        # Best consensus config found in self-test: structural rules, >=2 agree.
        _RULE["ids"] = set(wj.get("universes", {}).get("strong_struct", []))
        _RULE["min_agree"] = 2
        return f"ready (consensus: >={_RULE['min_agree']} of {len(_RULE['ids'])} structural rules)"
    except Exception as exc:  # noqa: BLE001
        _RULE["mod"] = None
        _RULE["weights"] = None
        _RULE["ids"] = None
        return f"disabled ({exc})"


def rule_predict(records: list[dict]) -> dict[str, Any] | None:
    """Run the structural Big/Small consensus engine on the live history."""
    we = _RULE["mod"]
    if we is None or not records or not _RULE["ids"]:
        return None
    try:
        D = np.array([int(r["number"]) for r in records], dtype=np.int16)
        MIN = np.array([int(r["minute"]) for r in records], dtype=np.int32)
        next_min = (int(records[-1]["minute"]) + 1) % 1440
        return we.predict_next_consensus(
            D, MIN, _RULE["ids"], _RULE["min_agree"], next_min
        )
    except Exception:  # noqa: BLE001
        return None


# ==============================================================================
# 1. Resilient Live API Fetcher & Record Parser
# ==============================================================================
def derive_wingo_colours(digit: int, raw_colour: str = "") -> tuple[int, int, int]:
    """Return (RED, GREEN, VIOLET) binary flags for a Wingo digit 0..9."""
    c_up = str(raw_colour).upper()
    if any(tok in c_up for tok in ("RED", "GREEN", "VIOLET")):
        r = 1 if "RED" in c_up else 0
        g = 1 if "GREEN" in c_up else 0
        v = 1 if "VIOLET" in c_up else 0
        return r, g, v
    # Deterministic Wingo colour rule from digit 0..9
    r = 1 if digit in (0, 2, 4, 6, 8) else 0
    g = 1 if digit in (1, 3, 5, 7, 9) else 0
    v = 1 if digit in (0, 5) else 0
    return r, g, v


def parse_issue_time(issue_str: str) -> tuple[int, int]:
    """Extract (minute_of_day 0..1439, zeller_dow 0..6) matching prepare_data.py."""
    s = "".join(ch for ch in str(issue_str).strip() if ch.isdigit())
    if len(s) >= 12:
        mnt = (int(s[-4:]) - 1) % 1440
        y, m, dd = int(s[0:4]), int(s[4:6]), int(s[6:8])
        a = (14 - m) // 12
        yy = y - a
        mm = m + 12 * a - 2
        doy = (dd + (31 * mm) // 12 + yy + yy // 4 - yy // 100 + yy // 400) % 7
        return mnt, doy
    now = dt.datetime.now()
    mnt = now.hour * 60 + now.minute
    y, m, dd = now.year, now.month, now.day
    a = (14 - m) // 12
    yy = y - a
    mm = m + 12 * a - 2
    doy = (dd + (31 * mm) // 12 + yy + yy // 4 - yy // 100 + yy // 400) % 7
    return mnt, doy


def compute_next_issue_and_time(last_issue: str, last_mnt: int, last_dow: int) -> tuple[str, int, int]:
    """Compute the upcoming 1-minute period issueNumber, minute_of_day, and dow."""
    s = "".join(ch for ch in str(last_issue).strip() if ch.isdigit())
    next_mnt = (last_mnt + 1) % 1440
    next_dow = (last_dow + 1) % 7 if next_mnt == 0 else last_dow

    if len(s) >= 12:
        seq = int(s[-4:])
        prefix = s[:-4]
        if seq < 1440:
            next_issue = f"{prefix}{seq + 1:04d}"
        else:
            # Roll over date YYYYMMDD by +1 day and reset sequence to 0001
            try:
                d_obj = dt.date(int(s[0:4]), int(s[4:6]), int(s[6:8])) + dt.timedelta(days=1)
                mid = s[8:-4]
                next_issue = f"{d_obj.strftime('%Y%m%d')}{mid}0001"
            except ValueError:
                next_issue = str(int(s) + 1)
        return next_issue, next_mnt, next_dow

    if s.isdigit():
        return str(int(s) + 1), next_mnt, next_dow
    return f"{last_issue}_NEXT", next_mnt, next_dow


def extract_records_list(payload: Any) -> list[dict]:
    """Extract the list of draw dicts from any API JSON wrapper format."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "data", "list", "results", "history", "items", "draws"):
            val = payload.get(key)
            if isinstance(val, list) and len(val) > 0:
                return val
            if isinstance(val, dict):
                for subkey in ("records", "list", "data", "items", "results"):
                    subval = val.get(subkey)
                    if isinstance(subval, list) and len(subval) > 0:
                        return subval
    raise ValueError(f"Unrecognized JSON structure from API: {type(payload)}")


def normalize_records(raw_records: list[dict]) -> list[dict]:
    """Normalize records and sort chronologically (oldest first, newest last)."""
    cleaned = []
    for idx, r in enumerate(raw_records):
        if not isinstance(r, dict):
            continue
        # Issue number
        issue = (
            r.get("issueNumber")
            or r.get("issue")
            or r.get("period")
            or r.get("expect")
            or r.get("qishu")
            or r.get("id")
            or str(idx)
        )
        issue_str = str(issue).strip()

        # Digit 0..9
        num_val = None
        for nk in ("number", "num", "digit", "result", "openNum", "winning_number", "premium"):
            if nk in r and r[nk] is not None:
                num_val = r[nk]
                break
        if num_val is None:
            continue
        num_digits = [ch for ch in str(num_val).strip() if ch.isdigit()]
        if not num_digits:
            continue
        digit = int(num_digits[-1])

        # Colour
        raw_col = r.get("colour") or r.get("color") or ""
        red, green, violet = derive_wingo_colours(digit, str(raw_col))
        mnt, dow = parse_issue_time(issue_str)

        cleaned.append(
            {
                "issueNumber": issue_str,
                "number": digit,
                "size": 1 if digit >= 5 else 0,
                "red": red,
                "green": green,
                "violet": violet,
                "minute": mnt,
                "dow": dow,
            }
        )

    if len(cleaned) < 10:
        raise ValueError(f"Too few valid records parsed from API ({len(cleaned)} found).")

    # Sort chronologically: oldest -> newest
    first_issue = "".join(ch for ch in cleaned[0]["issueNumber"] if ch.isdigit())
    last_issue = "".join(ch for ch in cleaned[-1]["issueNumber"] if ch.isdigit())
    if first_issue and last_issue and first_issue != last_issue:
        cleaned.sort(key=lambda x: "".join(ch for ch in x["issueNumber"] if ch.isdigit()))
    else:
        # Standard Wingo APIs return newest first at index 0
        cleaned.reverse()

    return cleaned


def fetch_live_records(api_url: str, timeout: float = 10.0) -> list[dict]:
    """Fetch and normalize the latest 1,000 Wingo 1M results from the API."""
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WingoAI/1.0",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_bytes = resp.read()
    payload = json.loads(raw_bytes.decode("utf-8"))
    raw_list = extract_records_list(payload)
    return normalize_records(raw_list)


# ==============================================================================
# 2. Exact 89-Feature + 128-Token Builder (Causal for Upcoming Draw i = n)
# ==============================================================================
def roll_ratio(flag: np.ndarray, i: int, w: int) -> float:
    a = max(0, i - w)
    seg = flag[a:i]
    return float(seg.mean()) if len(seg) else 0.5


def roll_sum(x: np.ndarray, i: int, w: int) -> float:
    a = max(0, i - w)
    seg = x[a:i]
    return float(seg.sum()) if len(seg) else 0.0


def lookup(d: dict, k: int, default: float) -> float:
    if k in d:
        return d[k]
    near = [kk for kk in d if abs(kk - k) <= 1]
    if near:
        return float(np.mean([d[kk] for kk in near]))
    return default


def build_features(
    i: int,
    D: np.ndarray,
    SB: np.ndarray,
    R: np.ndarray,
    G: np.ndarray,
    V: np.ndarray,
    MIN: np.ndarray,
    DOW: np.ndarray,
    dragon_p: dict = TRAINED_DRAGON_P,
    zig_p: dict = TRAINED_ZIG_P,
    pmap: dict = TRAINED_PMAP,
) -> np.ndarray:
    """Exact 93-feature builder matching prepare_data.py for position i (uses < i)."""
    f = np.zeros(N_FEATURES, dtype=np.float32)
    k = 0

    # --- streaks (5) ---
    run_len, run_val = 0, -1
    for j in range(i - 1, max(0, i - 60) - 1, -1):
        if run_len == 0:
            run_val, run_len = int(SB[j]), 1
        elif SB[j] == run_val:
            run_len += 1
        else:
            break
    f[k] = (1 if run_val == 1 else -1) * run_len; k += 1
    f[k] = run_len / 10.0; k += 1

    lb = ls = cur = curv = 0
    for j in range(max(1, i - 50), i):
        if SB[j] == curv:
            cur += 1
        else:
            curv, cur = int(SB[j]), 1
        lb = max(lb, cur) if curv == 1 else lb
        ls = max(ls, cur) if curv == 0 else ls
    f[k] = lb / 10.0; k += 1
    f[k] = ls / 10.0; k += 1

    ccol = int(R[i - 1]) * 1 + int(G[i - 1]) * 2 + int(V[i - 1]) * 4
    cs = 1
    for j in range(i - 2, max(0, i - 30) - 1, -1):
        cj = int(R[j]) * 1 + int(G[j]) * 2 + int(V[j]) * 4
        if cj == ccol:
            cs += 1
        else:
            break
    f[k] = cs / 10.0; k += 1

    # --- zigzag (4) ---
    zlen = 1
    for j in range(i - 1, max(1, i - 30) - 1, -1):
        if SB[j] != SB[j - 1]:
            zlen += 1
        else:
            break
    zactive = 1 if (i >= 3 and SB[i - 1] != SB[i - 2] and SB[i - 2] != SB[i - 3]) else 0
    f[k] = float(zactive); k += 1
    f[k] = zlen / 8.0; k += 1
    f[k] = min(1.0, zlen / 8.0); k += 1
    f[k] = lookup(zig_p, zlen, 0.5); k += 1

    # --- dragon (4) ---
    dactive = 1 if run_len >= 3 else 0
    f[k] = float(dactive); k += 1
    f[k] = run_len / 10.0; k += 1
    f[k] = min(1.0, run_len / 10.0); k += 1
    f[k] = lookup(dragon_p, run_len, 0.5); k += 1

    # --- size ratios (5) ---
    for w in (10, 20, 50, 100, 200):
        f[k] = roll_ratio(SB, i, w); k += 1

    # --- colour ratios (6) ---
    for flag in (G, R, V):
        f[k] = roll_ratio(flag, i, 50); k += 1
    for flag in (G, R, V):
        f[k] = roll_ratio(flag, i, 200); k += 1

    # --- size lags (5) ---
    for L in range(1, 6):
        f[k] = float(SB[i - L]) if i - L >= 0 else 0.5; k += 1

    # --- digit lags (5) ---
    for L in range(1, 6):
        f[k] = (float(D[i - L]) / 9.0) if i - L >= 0 else 0.5; k += 1

    # --- colour multi-label lags (6) ---
    for L in (1, 2):
        if i - L >= 0:
            f[k] = R[i - L]; k += 1
            f[k] = G[i - L]; k += 1
            f[k] = V[i - L]; k += 1
        else:
            k += 3

    # --- digit freq last 200 (10) ---
    seg = D[max(0, i - 200) : i]
    if len(seg):
        cnt = np.bincount(seg, minlength=10).astype(np.float32) / len(seg)
    else:
        cnt = np.full(10, 0.1, dtype=np.float32)
    f[k : k + 10] = cnt; k += 10

    # --- stats (8) ---
    ent = 0.0
    for p in cnt:
        if p > 0:
            ent -= float(p) * math.log(float(p) + 1e-9)
    f[k] = ent / math.log(10.0); k += 1

    seg2 = SB[max(1, i - 200) : i].astype(np.float32)
    if len(seg2) > 10:
        x, y = seg2[:-1], seg2[1:]
        if x.std() > 0 and y.std() > 0:
            f[k] = float(np.corrcoef(x, y)[0, 1])
        k += 1
    else:
        k += 1

    seg3 = SB[max(0, i - 100) : i]
    if len(seg3) > 2:
        runs = 1 + int(np.sum(seg3[1:] != seg3[:-1]))
        p = float(seg3.mean())
        exp = 1 + 2 * len(seg3) * p * (1 - p)
        var = max(1e-6, (exp - 1) * (exp - 2) / max(1, len(seg3) - 1))
        f[k] = float(np.clip((runs - exp) / math.sqrt(var) / 5.0, -1, 1))
        k += 1
    else:
        k += 1

    p_sz = roll_ratio(SB, i, 500)
    exp_streak = 1.0 / max(0.05, p_sz if run_val == 1 else 1 - p_sz)
    f[k] = float(np.clip((run_len - exp_streak) / 5.0, -2, 2)); k += 1
    f[k] = roll_sum(D, i, 5) / 45.0; k += 1
    a = max(0, i - 10)
    f[k] = (float(D[a:i].mean()) / 9.0) if i > a else 0.5; k += 1
    even = (
        np.asarray([1.0 if d % 2 == 0 else 0.0 for d in D[max(0, i - 50) : i]]).mean()
        if i > 0
        else 0.5
    )
    f[k] = float(even); k += 1

    if i > 0:
        hist = D[max(0, i - 500) : i]
        last = {}
        for idx2, dval in enumerate(hist):
            last[int(dval)] = len(hist) - 1 - idx2
        ov = [last.get(dv, 50) for dv in range(10)]
        f[k] = float(np.mean(ov)) / 50.0
        k += 1
    else:
        k += 1

    # --- transitions (6) ---
    a = max(1, i - 500)
    seg_s = SB[a:i]
    if len(seg_s) > 50:
        prev, cur2 = seg_s[:-1], seg_s[1:]
        bb = np.sum((prev == 1) & (cur2 == 1)); nb = np.sum(prev == 1)
        bs = np.sum((prev == 0) & (cur2 == 1)); ns = np.sum(prev == 0)
        f[k] = bb / max(1, nb); k += 1
        f[k] = bs / max(1, ns); k += 1
        pr, cg = R[a:i], G[a:i]
        rg = np.sum((pr[:-1] == 1) & (cg[1:] == 1)); nr = np.sum(pr[:-1] == 1)
        gr = np.sum((cg[:-1] == 1) & (pr[1:] == 1)); ng = np.sum(cg[:-1] == 1)
        f[k] = rg / max(1, nr); k += 1
        f[k] = gr / max(1, ng); k += 1
        f[k] = float(V[a:i].mean()); k += 1
        p = float(seg_s.mean())
        pe = 0.0
        if 0 < p < 1:
            pe = -(p * math.log(p + 1e-9) + (1 - p) * math.log(1 - p + 1e-9))
        f[k] = pe / math.log(2.0); k += 1
    else:
        k += 6

    # --- patterns (12) ---
    p4 = tuple(int(SB[i - 4 + t]) for t in range(4)) if i >= 4 else (0, 0, 0, 0)
    f[k] = (sum(v << t for t, v in enumerate(p4))) / 15.0; k += 1
    last8 = SB[max(0, i - 8) : i]
    rle = 1 + int(np.sum(last8[1:] != last8[:-1])) if len(last8) > 1 else 1
    f[k] = rle / 8.0; k += 1
    f[k] = 1.0 if (i >= 4 and len(set(p4)) == 1) else 0.0; k += 1
    f[k] = 1.0 if p4 == (1, 0, 1, 0) and i >= 4 else 0.0; k += 1
    f[k] = 1.0 if p4 in ((1, 1, 0, 0), (0, 0, 1, 1)) else 0.0; k += 1
    p3 = tuple(int(SB[i - 3 + t]) for t in range(3)) if i >= 3 else (0, 0, 0)
    f[k] = sum(v << t for t, v in enumerate(p3)) / 7.0; k += 1
    f[k] = float(PATTERN4.get(p4, 0)) * 0.5 + 0.5; k += 1
    cp = tuple(1 if R[i - 4 + t] else 0 for t in range(4)) if i >= 4 else (0, 0, 0, 0)
    f[k] = (sum(v << t for t, v in enumerate(cp))) / 15.0; k += 1
    a = max(0, i - 10)
    f[k] = float(R[a:i].mean()); k += 1
    f[k] = float(G[a:i].mean()); k += 1
    f[k] = float(V[a:i].mean()); k += 1
    f[k] = float(V[max(0, i - 20) : i].mean()); k += 1

    # --- time (9) ---
    mnt = int(MIN[i]) if i < len(MIN) else 0
    ang = 2 * math.pi * mnt / 1440.0
    f[k] = math.sin(ang); k += 1
    f[k] = math.cos(ang); k += 1
    h = mnt / 60.0
    f[k] = math.sin(2 * math.pi * h / 24.0); k += 1
    f[k] = math.cos(2 * math.pi * h / 24.0); k += 1
    dval = int(DOW[i]) if i < len(DOW) else 0
    f[k] = math.sin(2 * math.pi * dval / 7.0); k += 1
    f[k] = math.cos(2 * math.pi * dval / 7.0); k += 1
    f[k] = 1.0 if dval >= 5 else 0.0; k += 1
    f[k] = mnt / 1439.0; k += 1
    f[k] = 1.0 if h < 6 else 0.0; k += 1

    # --- overdue (4) ---
    def since(flag: np.ndarray, cap: int = 30) -> float:
        for j in range(i - 1, max(0, i - cap) - 1, -1):
            if flag[j]:
                return (i - j) / cap
        return 1.0

    f[k] = since(V); k += 1
    f[k] = since(SB, 10); k += 1
    f[k] = since(G); k += 1
    f[k] = since(R); k += 1

    # --- ALT-4 rule block (4) — matches prepare_data.py feat[89..92] ---
    pat_int = sum(int(SB[i - 4 + t]) << t for t in range(4)) if i >= 4 else 0
    alt4 = 1.0 if pat_int in (5, 10) else 0.0
    f[k] = alt4; k += 1
    f[k] = float(SB[i - 1]) if (alt4 and i >= 1) else 0.5; k += 1
    f[k] = float(pmap.get(pat_int, 0.5)) if i >= 4 else 0.5; k += 1
    f[k] = 1.0 if run_len >= 8 else 0.0; k += 1

    assert k == N_FEATURES, f"Feature count mismatch: {k} != {N_FEATURES}"
    return f


def prepare_live_inputs(records: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build [1, 128] tok and [1, 93] feat for the next upcoming period (i = len(records))."""
    n = len(records)
    D = np.array([r["number"] for r in records], dtype=np.int16)
    SB = np.array([r["size"] for r in records], dtype=np.int8)
    R = np.array([r["red"] for r in records], dtype=np.int8)
    G = np.array([r["green"] for r in records], dtype=np.int8)
    V = np.array([r["violet"] for r in records], dtype=np.int8)

    last_rec = records[-1]
    next_issue, next_mnt, next_dow = compute_next_issue_and_time(
        last_rec["issueNumber"], last_rec["minute"], last_rec["dow"]
    )

    MIN = np.array([r["minute"] for r in records] + [next_mnt], dtype=np.int32)
    DOW = np.array([r["dow"] for r in records] + [next_dow], dtype=np.int32)

    feat_vec = build_features(n, D, SB, R, G, V, MIN, DOW)
    tok_vec = np.full((1, SEQ_LEN), 10, dtype=np.uint8)
    seq = D[max(0, n - SEQ_LEN) : n]
    tok_vec[0, SEQ_LEN - len(seq) :] = seq.astype(np.uint8)

    meta_info = {
        "n_records": n,
        "latest_issue": last_rec["issueNumber"],
        "latest_number": last_rec["number"],
        "latest_size": "BIG" if last_rec["size"] == 1 else "SMALL",
        "latest_colour": "+".join(
            [
                c
                for c, flag in (
                    ("RED", last_rec["red"]),
                    ("GREEN", last_rec["green"]),
                    ("VIOLET", last_rec["violet"]),
                )
                if flag
            ]
        ),
        "next_issue": next_issue,
        "next_minute": next_mnt,
        "signed_streak": int(feat_vec[0]),
        "zigzag_len": int(round(feat_vec[6] * 8.0)),
        "dragon_active": bool(feat_vec[9] > 0.5),
        "big_ratio_50": float(feat_vec[15]),
        "last_10_digits": D[-10:].tolist(),
    }
    return tok_vec, feat_vec[None, :], meta_info


# ==============================================================================
# 3. Model Loader & Inference Engine (PyTorch + Pure-NumPy Fallback)
# ==============================================================================
def load_pt_weights_numpy(pt_path: str | Path) -> dict[str, np.ndarray]:
    """Load PyTorch .pt zip archive directly into NumPy arrays without requiring torch."""
    with zipfile.ZipFile(pt_path, "r") as zf:
        namelist = zf.namelist()
        pkl_name = [n for n in namelist if n.endswith("data.pkl")][0]
        prefix = pkl_name[: -len("data.pkl")]
        storages: dict[str, Any] = {}

        class DummyStorage:
            def __init__(self, dtype: Any, key: str, size: int) -> None:
                raw = zf.read(f"{prefix}data/{key}")
                self.arr = np.frombuffer(raw, dtype=dtype)

        def rebuild_tensor_v2(
            storage: DummyStorage,
            storage_offset: int,
            size: tuple[int, ...],
            stride: Any,
            requires_grad: bool,
            backward_hooks: Any,
            metadata: Any = None,
        ) -> np.ndarray:
            count = int(np.prod(size))
            return storage.arr[storage_offset : storage_offset + count].reshape(size)

        class TorchUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return rebuild_tensor_v2
                if module == "torch" and "Storage" in name:
                    dtype_map = {
                        "FloatStorage": np.float32,
                        "HalfStorage": np.float16,
                        "LongStorage": np.int64,
                        "IntStorage": np.int32,
                        "ByteStorage": np.uint8,
                        "DoubleStorage": np.float64,
                        "BoolStorage": np.bool_,
                    }
                    return dtype_map.get(name, np.float32)
                return super().find_class(module, name)

            def persistent_load(self, pid: Any) -> DummyStorage:
                _, storage_type, key, _location, size = pid
                if key not in storages:
                    storages[key] = DummyStorage(storage_type, key, size)
                return storages[key]

        ckpt = TorchUnpickler(io.BytesIO(zf.read(pkl_name))).load()
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        fixed_sd = {}
        for k, v in sd.items():
            k_new = "reg" if k == "registers" else k
            k_new = k_new.replace(".attn.q_norm.", ".attn.qn.").replace(
                ".attn.k_norm.", ".attn.kn."
            )
            fixed_sd[k_new] = v
        return fixed_sd


class WingoPredictorEngine:
    """Unified WingoNet inference engine supporting PyTorch (CUDA/CPU) and NumPy."""

    def __init__(self, ckpt_path: str | Path) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.backend = "numpy"
        self.device = "cpu"
        self.torch_model = None
        self.np_weights: dict[str, np.ndarray] = {}

        try:
            import torch
            from wingo_net import load_checkpoint

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch_model, _ = load_checkpoint(self.ckpt_path, device=self.device)
            self.backend = f"pytorch ({self.device})"
        except Exception:
            self.np_weights = load_pt_weights_numpy(self.ckpt_path)
            self.backend = "numpy-native (no torch required)"

        # Detect the feature width the checkpoint was trained with (89 = legacy, 93 = ALT-4).
        if self.torch_model is not None:
            self.n_feat = int(self.torch_model.feat_mlp[0].weight.shape[1])
        else:
            self.n_feat = int(self.np_weights["feat_mlp.0.weight"].shape[1])

        # Precompute ALiBi slopes for 8 heads
        start = 2 ** (-math.pow(2, -(math.log2(8) - 3)))
        self.alibi = np.array([start * (start**i) for i in range(8)], dtype=np.float32)

    def predict(self, tok: np.ndarray, feat: np.ndarray) -> dict[str, Any]:
        """Run forward pass on tok [1, 128] and feat [1, N_FEATURES] -> predictions."""
        feat = feat[:, : self.n_feat]
        if self.torch_model is not None:
            import torch

            with torch.no_grad():
                t_ten = torch.from_numpy(tok.astype(np.int64)).to(self.device)
                f_ten = torch.from_numpy(feat.astype(np.float32)).to(self.device)
                sz_logit_t, col_logit_t = self.torch_model(t_ten, f_ten)
                sz_logit = float(sz_logit_t.cpu().item())
                col_logits = col_logit_t.cpu().numpy()[0]
        else:
            sz_logit_arr, col_logits_arr = self._forward_numpy(tok, feat)
            sz_logit = float(sz_logit_arr[0])
            col_logits = col_logits_arr[0]

        # Probabilities via sigmoid
        p_big = 1.0 / (1.0 + math.exp(-sz_logit))
        p_small = 1.0 - p_big

        # Decision line is always 0.5000 (same as training/eval). Strength is just
        # how far p_big sits from that line in units of the test-set std. The sign
        # therefore always agrees with the raw pick (no more SMALL/BIG contradiction).
        Z_STD = 0.007963
        z_size = (p_big - 0.5) / Z_STD
        pred_size = "BIG" if p_big > 0.5 else "SMALL"
        pred_size_cal = pred_size
        size_conf_pct = max(p_big, p_small) * 100.0
        cal_conf_pct = min(99.0, 50.0 + abs(z_size) * 12.5)

        p_red = 1.0 / (1.0 + math.exp(-float(col_logits[0])))
        p_green = 1.0 / (1.0 + math.exp(-float(col_logits[1])))
        p_violet = 1.0 / (1.0 + math.exp(-float(col_logits[2])))

        # Primary colour = direct argmax (same rule the evaluator used). The old
        # mean-shift "calibration" could disagree with the raw pick, so it was
        # removed. Violet alerts only on a real >0.5 prediction, not the base rate.
        primary_colour = "RED" if p_red >= p_green else "GREEN"
        calibrated_colour = primary_colour
        violet_alert = p_violet > 0.5

        active_multilabel = []
        if p_red > 0.5:
            active_multilabel.append("RED")
        if p_green > 0.5:
            active_multilabel.append("GREEN")
        if p_violet > 0.5:
            active_multilabel.append("VIOLET")
        if not active_multilabel:
            active_multilabel.append(primary_colour)

        return {
            "size_logit": sz_logit,
            "p_big": p_big,
            "p_small": p_small,
            "pred_size": pred_size,
            "pred_size_calibrated": pred_size_cal,
            "size_conf_pct": size_conf_pct,
            "calibrated_conf_pct": cal_conf_pct,
            "z_size": z_size,
            "p_red": p_red,
            "p_green": p_green,
            "p_violet": p_violet,
            "primary_colour": primary_colour,
            "calibrated_colour": calibrated_colour,
            "multilabel_colours": "+".join(active_multilabel),
            "violet_alert": violet_alert,
        }

    def _forward_numpy(self, tok: np.ndarray, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = self.np_weights

        def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
            rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
            return (x / rms) * weight

        def silu(x: np.ndarray) -> np.ndarray:
            return x / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

        B, L = tok.shape
        feat = feat[:, : int(w["feat_mlp.0.weight"].shape[1])]
        x = w["embed.weight"][tok]
        reg = np.broadcast_to(w["reg"][None, :, :], (B, 4, 256))
        x = np.concatenate([reg, x], axis=1)
        Ltot = x.shape[1]
        pos = np.arange(Ltot)
        dist = -np.abs(pos[None, :] - pos[:, None]).astype(np.float32)
        mask = np.triu(np.ones((Ltot, Ltot), dtype=bool), 1)

        for b in range(5):
            p = f"blocks.{b}."
            h = rmsnorm(x, w[p + "an.weight"])
            q = rmsnorm(
                (h @ w[p + "attn.Wq.weight"].T).reshape(B, Ltot, 8, 32),
                w[p + "attn.qn.weight"],
            )
            k = rmsnorm(
                (h @ w[p + "attn.Wk.weight"].T).reshape(B, Ltot, 2, 32),
                w[p + "attn.kn.weight"],
            )
            v = (h @ w[p + "attn.Wv.weight"].T).reshape(B, Ltot, 2, 32)
            k = np.repeat(k, 4, axis=2)
            v = np.repeat(v, 4, axis=2)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(32.0)
            scores = scores + self.alibi[:, None, None] * dist
            scores = np.where(mask, -1e9, scores)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_s = np.exp(scores - scores_max)
            attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            out = (
                (attn @ v).transpose(0, 2, 1, 3).reshape(B, Ltot, 256)
                @ w[p + "attn.Wo.weight"].T
            )
            x = x + out

            h2 = rmsnorm(x, w[p + "fn.weight"])
            ff = (
                silu(h2 @ w[p + "ffn.w1.weight"].T) * (h2 @ w[p + "ffn.w2.weight"].T)
            ) @ w[p + "ffn.w3.weight"].T
            x = x + ff

        # Training forward does NOT apply self.final before pooling (it stays at init
        # and gets no gradient). Mirror that exactly so predictions match the model.
        x_tok = x[:, 4:, :]
        pooled = x_tok.mean(axis=1)
        f = (
            silu(feat @ w["feat_mlp.0.weight"].T + w["feat_mlp.0.bias"])
            @ w["feat_mlp.2.weight"].T
            + w["feat_mlp.2.bias"]
        )
        h_final = np.concatenate([pooled, f], axis=-1)
        sz_logit = (h_final @ w["size_head.weight"].T).squeeze(-1) + w["size_head.bias"]
        col_logit = (h_final @ w["colour_head.weight"].T) + w["colour_head.bias"]
        return sz_logit, col_logit


def find_checkpoint(explicit_path: str | None = None) -> Path:
    """Locate best.pt or final.pt across common working directories."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint not found at: {explicit_path}")

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path("best.pt"),
        Path("final.pt"),
        Path("results/wingo/best.pt"),
        Path("results/wingo/final.pt"),
        Path("extracted/results/wingo/best.pt"),
        Path("extracted/results/wingo/final.pt"),
        script_dir / "best.pt",
        script_dir / "final.pt",
        script_dir / "results/wingo/best.pt",
        script_dir.parent / "results/wingo/best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find best.pt or final.pt! Place best.pt in the same folder or pass --model path/to/best.pt"
    )


# ==============================================================================
# 4. Console Formatter & Live 1-Minute Loop
# ==============================================================================
def _issue5(issue: Any) -> str:
    s = "".join(ch for ch in str(issue) if ch.isdigit())
    return s[-5:] if len(s) >= 5 else s


def _mark(win: bool) -> str:
    sym = "\u2713" if win else "\u2717"
    try:
        sym.encode(sys.stdout.encoding or "utf-8")
        return sym
    except Exception:
        return "OK" if win else "XX"


def print_prediction_card(
    meta: dict[str, Any],
    pred: dict[str, Any],
    stats: dict[str, int],
    last_eval_msg: str | None = None,
    rule_pred: dict[str, Any] | None = None,
    prev: dict[str, Any] | None = None,
) -> None:
    W = 60
    bar = "=" * W
    thin = "-" * W
    print("\n" + bar)
    print("1MIN WINGO".center(W))
    print(bar)
    print(
        f"  Latest Result : {_issue5(meta['latest_issue'])} - {meta['latest_size']}"
        f"   (digit {meta['latest_number']}, {meta['latest_colour']})"
    )
    print(thin)
    if prev is not None:
        print(
            f"  LAST PICK     : {_issue5(prev['issue'])} - {prev['size']:<5s} {_mark(prev['hit'])}"
            f"     Colour: {prev['colour']:<6s} {_mark(prev['col_hit'])}"
        )
    else:
        print("  LAST PICK     : --")
    print("                 NEXT PREDICTION")
    print(
        f"  {_issue5(meta['next_issue'])} - {pred['pred_size']:<5s}"
        f"                 Colour: {pred['primary_colour']}"
    )
    print(thin)
    total = stats.get("total", 0)
    if total > 0:
        sw = stats.get("size_wins", 0)
        cw = stats.get("col_wins", 0)
        line = (
            f"  Session : Size {sw}/{total} ({sw / total * 100:.0f}%)"
            f" | Colour {cw}/{total} ({cw / total * 100:.0f}%)"
        )
        rt = stats.get("rule_total", 0)
        if rt > 0:
            rw = stats.get("rule_wins", 0)
            line += f" | Rule {rw}/{rt}"
        print(line)
    print(bar, flush=True)


def run_offline_self_test(engine: WingoPredictorEngine) -> None:
    """Verify full pipeline on 1,000 synthetic/local records when running --test-local."""
    print(f"[INFO] Loaded checkpoint: {engine.ckpt_path} via {engine.backend}")
    test_tok_path = Path("extracted/wingo_ai/wingo_ai/data/test_tok.npy")
    if test_tok_path.exists():
        test_tok = np.load(test_tok_path)
        digits = test_tok[1:1001, -1].tolist()
    else:
        rng = np.random.default_rng(42)
        digits = rng.integers(0, 10, size=1000).tolist()

    base_issue = 2026110710000100
    fake_api_records = []
    for i, d in enumerate(digits):
        r, g, v = derive_wingo_colours(int(d))
        cols = [name for name, flg in (("RED", r), ("GREEN", g), ("VIOLET", v)) if flg]
        fake_api_records.append(
            {
                "issueNumber": str(base_issue + i + 1),
                "number": int(d),
                "colour": ",".join(cols),
            }
        )
    # Reverse to mimic API returning newest-first
    fake_api_records.reverse()
    norm = normalize_records(fake_api_records)
    tok, feat, meta = prepare_live_inputs(norm)
    pred = engine.predict(tok, feat)
    print_prediction_card(meta, pred, {"total": 0, "size_wins": 0, "col_wins": 0})


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Live 1-Minute Wingo AI Predictor")
    parser.add_argument("--api", default=DEFAULT_API_URL, help="Live 1,000-history API URL")
    parser.add_argument("--model", default=None, help="Path to best.pt or final.pt")
    parser.add_argument("--interval", type=float, default=3.0, help="API poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run a single prediction and exit")
    parser.add_argument("--test-local", action="store_true", help="Run offline self-test")
    args = parser.parse_args()

    ckpt_path = find_checkpoint(args.model)
    engine = WingoPredictorEngine(ckpt_path)

    if args.test_local:
        run_offline_self_test(engine)
        return

    rule_status = init_rule_engine()
    print(f"[INIT] Model loaded from : {ckpt_path}")
    print(f"[INIT] Inference Backend : {engine.backend}")
    print(f"[INIT] Rule Engine       : {rule_status}")
    print(f"[INIT] Live API Endpoint : {args.api}")
    print("[INIT] Waiting for live 1M draws... (Press Ctrl+C to stop)")

    last_seen_issue: str | None = None
    pending_pred: dict[str, Any] | None = None
    pending_issue: str | None = None
    pending_rule: dict[str, Any] | None = None
    pending_rule_issue: str | None = None
    stats = {"total": 0, "size_wins": 0, "col_wins": 0,
             "rule_total": 0, "rule_wins": 0}

    while True:
        try:
            records = fetch_live_records(args.api)
            latest = records[-1]
            cur_issue = latest["issueNumber"]

            if cur_issue != last_seen_issue:
                eval_msg = None
                prev_info = None
                if pending_pred is not None and pending_issue == cur_issue:
                    actual_size = "BIG" if latest["size"] == 1 else "SMALL"
                    actual_cols = [
                        c
                        for c, flg in (
                            ("RED", latest["red"]),
                            ("GREEN", latest["green"]),
                            ("VIOLET", latest["violet"]),
                        )
                        if flg
                    ]
                    size_hit = pending_pred["pred_size"] == actual_size
                    col_hit = pending_pred["primary_colour"] in actual_cols
                    stats["total"] += 1
                    stats["size_wins"] += int(size_hit)
                    stats["col_wins"] += int(col_hit)
                    prev_info = {
                        "issue": cur_issue,
                        "size": pending_pred["pred_size"],
                        "hit": size_hit,
                        "colour": pending_pred["primary_colour"],
                        "col_hit": col_hit,
                    }
                    rule_part = ""
                    if (pending_rule is not None and pending_rule_issue == cur_issue
                            and pending_rule.get("pred")):
                        rule_hit = pending_rule["pred"] == actual_size
                        stats["rule_total"] += 1
                        stats["rule_wins"] += int(rule_hit)
                        rule_part = (
                            f" | Rule={pending_rule['pred']} "
                            f"[{'WIN' if rule_hit else 'LOSS'}]"
                        )
                    eval_msg = (
                        f"Issue {cur_issue}: Actual={latest['number']} ({actual_size}, {'+'.join(actual_cols)}) | "
                        f"Pred Size={pending_pred['pred_size']} [{'WIN' if size_hit else 'LOSS'}] | "
                        f"Pred Colour={pending_pred['primary_colour']} [{'WIN' if col_hit else 'LOSS'}]"
                        f"{rule_part}"
                    )

                rpred = rule_predict(records)
                tok, feat, meta = prepare_live_inputs(records)
                pred = engine.predict(tok, feat)
                print_prediction_card(meta, pred, stats, eval_msg, rpred, prev_info)

                last_seen_issue = cur_issue
                pending_issue = meta["next_issue"]
                pending_pred = pred
                pending_rule = rpred
                pending_rule_issue = meta["next_issue"]

                if args.once:
                    break

        except KeyboardInterrupt:
            print("\n[EXIT] Stopped by user.")
            break
        except urllib.error.URLError as e:
            print(f"[WARN] API network error: {e} — retrying in {args.interval:.0f}s...", flush=True)
        except Exception as e:
            print(f"[WARN] Error processing live feed: {e} — retrying in {args.interval:.0f}s...", flush=True)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()