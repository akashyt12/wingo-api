#!/usr/bin/env python3
"""Unified BIG/SMALL rule engine for Wingo 1M.

Modes
-----
  single_zig3 : ONLY the strongest rule (zigzag>=3 -> repeat last size)
  structural  : structural rules only (zigzag/repeat + long-run + patterns)
  all         : every rule incl. weak time / always-repeat (diluted baseline)

Rules' directions & weights are calibrated on TRAIN only during self-test;
on live data the calibrated weights are loaded from engine_rules.json.

Self-test on the dataset prints per-rule and combined accuracy on
VAL / TEST (honest out-of-sample).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\Welcome\Downloads\Telegram Desktop\big_dataset (2).json")
RULES_JSON = Path(r"C:\Users\Welcome\ai-research\wingo_ai\engine_rules.json")

PAYOUT = 1.96
BREAKEVEN = 1.0 / PAYOUT


# ---------------------------------------------------------------- data
def load_records(src=SRC):
    d = json.load(open(src, encoding="utf-8"))
    recs = list(reversed(d["records"]))
    n = len(recs)
    D = np.array([r["number"] for r in recs], dtype=np.int16)
    MIN = np.zeros(n, np.int32)
    for i, r in enumerate(recs):
        MIN[i] = (int(r["issueNumber"][-4:]) - 1) % 1440
    return D, MIN


def build_ctx(D, MIN):
    SB = (D >= 5).astype(np.int8)
    n = len(SB)
    run = np.ones(n, np.int16)
    zig = np.zeros(n, np.int16)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if SB[i] == SB[i - 1] else 1
        zig[i] = zig[i - 1] + 1 if SB[i] != SB[i - 1] else 0
    pats = {}
    for k in (4, 5, 6):
        p = np.zeros(n, np.int64)
        for t in range(k):
            p[k:] += (SB[t: n - k + t].astype(np.int64) << (k - 1 - t))
        pats[k] = p
    return {"SB": SB, "run": run, "zig": zig, "pat": pats,
            "hour": (MIN // 60).astype(np.int16)}


# ---------------------------------------------------------------- rules
# direction: 'repeat' = dynamic (same as last size); int 0/1 = fixed.
# tier: STRONG / MEDIUM / WEAK
RULES = [
    {"id": "zig3_repeat",  "kind": "zig",   "param": 3, "dir": "repeat", "tier": "STRONG"},
    {"id": "zig2_repeat",  "kind": "zig",   "param": 2, "dir": "repeat", "tier": "STRONG"},
    {"id": "run8_repeat",  "kind": "run",   "param": 8, "dir": "repeat", "tier": "MEDIUM"},
    {"id": "run6_repeat",  "kind": "run",   "param": 6, "dir": "repeat", "tier": "MEDIUM"},
    {"id": "pat4_0101",    "kind": "pat",   "bits": "0101",   "dir": 1, "tier": "STRONG"},
    {"id": "pat4_1010",    "kind": "pat",   "bits": "1010",   "dir": 0, "tier": "STRONG"},
    {"id": "pat5_00101",   "kind": "pat",   "bits": "00101",  "dir": 1, "tier": "STRONG"},
    {"id": "pat5_11010",   "kind": "pat",   "bits": "11010",  "dir": 0, "tier": "STRONG"},
    {"id": "pat6_000101",  "kind": "pat",   "bits": "000101", "dir": 1, "tier": "MEDIUM"},
    {"id": "pat6_110101",  "kind": "pat",   "bits": "110101", "dir": 1, "tier": "MEDIUM"},
    {"id": "pat6_000000",  "kind": "pat",   "bits": "000000", "dir": 0, "tier": "MEDIUM"},
    {"id": "pat6_111111",  "kind": "pat",   "bits": "111111", "dir": 1, "tier": "MEDIUM"},
    {"id": "night_big",    "kind": "hour",  "lo": 0, "hi": 5, "dir": 1, "tier": "WEAK"},
    {"id": "repeat_last",  "kind": "always","dir": "repeat", "tier": "WEAK"},
]
_MODE = {
    "single_zig3": {"zig3_repeat"},
    "structural": {"zig3_repeat", "zig2_repeat", "run8_repeat", "run6_repeat",
                   "pat4_0101", "pat4_1010", "pat5_00101", "pat5_11010",
                   "pat6_000101", "pat6_110101", "pat6_000000", "pat6_111111"},
    "all": {r["id"] for r in RULES},
}


def rule_fire(rule, ctx, i):
    """Return (fired, pred) at target index i using only data < i."""
    SB = ctx["SB"]
    if i < 1:
        return False, 0
    k = rule["kind"]
    if k == "zig":
        if ctx["zig"][i - 1] >= rule["param"]:
            return True, int(SB[i - 1])
        return False, 0
    if k == "run":
        if ctx["run"][i - 1] >= rule["param"]:
            return True, int(SB[i - 1])
        return False, 0
    if k == "pat":
        bits = rule["bits"]; L = len(bits)
        if i < L:
            return False, 0
        if int(ctx["pat"][L][i]) == int(bits, 2):
            return True, int(rule["dir"])
        return False, 0
    if k == "hour":
        h = int(ctx["hour"][i])
        if rule["lo"] <= h <= rule["hi"]:
            return True, int(rule["dir"])
        return False, 0
    if k == "always":
        return True, int(SB[i - 1])
    return False, 0


# ---------------------------------------------------------------- calibrate
def calibrate(ctx, i0, i1):
    """Train-only direction-check + weight for each rule."""
    SB = ctx["SB"]
    out = {}
    for r in RULES:
        fires = 0; wins = 0
        for i in range(max(i0, 2), i1):
            f, p = rule_fire(r, ctx, i)
            if f:
                fires += 1
                wins += int(p == int(SB[i]))
        acc = wins / fires if fires else 0.5
        w = max(0.0, acc - 0.5) * 100.0     # scaled edge
        d = {"id": r["id"], "tier": r["tier"], "n": fires,
             "acc_train": acc, "weight": w}
        out[r["id"]] = d
    return out


def votes_at(ctx, i, weights, mode="structural"):
    allowed = _MODE[mode]
    big = 0.0; small = 0.0; detail = []
    for r in RULES:
        if r["id"] not in allowed:
            continue
        f, p = rule_fire(r, ctx, i)
        if not f:
            continue
        w = weights.get(r["id"], {}).get("weight", 0.0)
        if w <= 0:
            continue
        if p == 1:
            big += w
        else:
            small += w
        detail.append((r["id"], "BIG" if p else "SMALL", round(w, 2)))
    if big == small:
        return None, 0.0, detail
    pred = 1 if big > small else 0
    return pred, abs(big - small), detail


def consensus_votes_at(ctx, i, rule_ids, min_agree):
    """Unweighted rule-agreement vote. Returns (pred, n_agree, detail)."""
    big = []; small = []
    for r in RULES:
        if r["id"] not in rule_ids:
            continue
        f, p = rule_fire(r, ctx, i)
        if not f:
            continue
        (big if p == 1 else small).append(r["id"])
    if len(big) >= min_agree and len(big) > len(small):
        return 1, len(big), big
    if len(small) >= min_agree and len(small) > len(big):
        return 0, len(small), small
    return None, 0, []


def consensus_collect(ctx, a, b, rule_ids, min_agree):
    m = b - a
    preds = np.full(m, -1, np.int8)
    confs = np.zeros(m)
    for j, i in enumerate(range(a, b)):
        pred, n, _ = consensus_votes_at(ctx, i, rule_ids, min_agree)
        if pred is not None:
            preds[j] = pred
            confs[j] = n
    return preds, confs, ctx["SB"][a:b]


def collect_mode(ctx, weights, a, b, mode):
    """One pass: return per-position pred (>-1 decided) and confidence."""
    m = b - a
    preds = np.full(m, -1, np.int8)
    confs = np.zeros(m)
    for j, i in enumerate(range(a, b)):
        pred, conf, _ = votes_at(ctx, i, weights, mode)
        if pred is not None:
            preds[j] = pred
            confs[j] = conf
    return preds, confs, ctx["SB"][a:b]


def eval_from(preds, confs, tgt, margin):
    decided = (preds > -1) & (confs > margin)
    n_bets = int(decided.sum())
    if n_bets == 0:
        return {"n": 0, "acc": 0.5, "z": 0.0, "cov": 0.0, "ev": -1.0}
    acc = float((preds[decided] == tgt[decided]).mean())
    return {"n": n_bets, "acc": acc,
            "z": (acc - 0.5) / math.sqrt(0.25 / n_bets),
            "cov": n_bets / len(preds), "ev": PAYOUT * acc - 1.0}


def eval_single_rule(ctx, rid, a, b):
    rule = next(r for r in RULES if r["id"] == rid)
    SB = ctx["SB"]; n_bets = 0; wins = 0
    for i in range(a, b):
        f, p = rule_fire(rule, ctx, i)
        if not f:
            continue
        n_bets += 1; wins += int(p == int(SB[i]))
    if n_bets == 0:
        return {"n": 0, "acc": 0.5, "z": 0.0, "ev": -1.0}
    acc = wins / n_bets
    return {"n": n_bets, "acc": acc,
            "z": (acc - 0.5) / math.sqrt(0.25 / n_bets), "ev": PAYOUT * acc - 1.0}


# ---------------------------------------------------------------- live next
def _pad_for_next(ctx, next_min=None):
    """Extend pattern/hour arrays by one slot for the NEXT draw."""
    SB = ctx["SB"]; n = len(SB)
    pat_ext = {}
    for k, p in ctx["pat"].items():
        pe = np.zeros(n + 1, np.int64)
        pe[:n] = p
        bits = 0
        if n >= k:
            for t in range(k):
                bits += int(SB[n - k + t]) << (k - 1 - t)
        pe[n] = bits
        pat_ext[k] = pe
    hour_ext = np.zeros(n + 1, np.int16)
    hour_ext[:n] = ctx["hour"]
    hour_ext[n] = (next_min // 60) if next_min is not None else -1
    return {"SB": SB, "run": ctx["run"], "zig": ctx["zig"],
            "pat": pat_ext, "hour": hour_ext}


def predict_next(D, MIN, mode="structural", weights=None, next_min=None):
    """Predict the NEXT draw (after the last recorded one).

    next_min: minute index (0..1439) of the upcoming draw if known,
              enables the hour/time rules; else pass None.
    """
    ctx = build_ctx(D, MIN)
    if weights is None:
        weights = calibrate(ctx, 2, len(ctx["SB"]))   # fallback: all history
    ctx = _pad_for_next(ctx, next_min)
    pred, conf, detail = votes_at(ctx, len(ctx["SB"]), weights, mode)
    return {"pred": None if pred is None else ("BIG" if pred else "SMALL"),
            "confidence": conf, "votes": detail}


def predict_next_consensus(D, MIN, rule_ids, min_agree, next_min=None):
    """Predict NEXT draw only when >= min_agree rules agree."""
    ctx = build_ctx(D, MIN)
    ctx = _pad_for_next(ctx, next_min)
    pred, n, detail = consensus_votes_at(ctx, len(ctx["SB"]), set(rule_ids), min_agree)
    return {"pred": None if pred is None else ("BIG" if pred else "SMALL"),
            "confidence": float(n), "votes": detail}


# ---------------------------------------------------------------- main
def main():
    D, MIN = load_records()
    ctx = build_ctx(D, MIN)
    n = len(ctx["SB"])
    s1, s2 = int(n * 0.8), int(n * 0.9)
    print(f"records {n:,} | train {s1:,} val {s2-s1:,} test {n-s2:,}")

    weights = calibrate(ctx, 2, s1)     # TRAIN-only calibration
    print("\ncalibrated rules (TRAIN only):")
    print(f"  {'rule':<16s} {'tier':<8s} {'n_train':>8s} {'acc_tr':>7s} {'weight':>7s}")
    for r in RULES:
        w = weights[r["id"]]
        print(f"  {r['id']:<16s} {r['tier']:<8s} {w['n']:>8,} {w['acc_train']:>7.4f} "
              f"{w['weight']:>7.2f}")

    print("\nPER-RULE out-of-sample (VAL / TEST):")
    per = {}
    for r in RULES:
        sv = eval_single_rule(ctx, r["id"], s1, s2)
        st = eval_single_rule(ctx, r["id"], s2, n)
        per[r["id"]] = {"val": sv, "test": st}
        print(f"  {r['id']:<16s} VAL n={sv['n']:>6,} acc={sv['acc']:.4f} z={sv['z']:+.2f} "
              f"| TEST n={st['n']:>6,} acc={st['acc']:.4f} z={st['z']:+.2f} "
              f"EV={st['ev']*100:+.2f}%")

    print("\nCOMBINED modes (margin filter) — VAL / TEST:")
    comb = {}
    for mode in ("single_zig3", "structural", "all"):
        pv, cv, tv = collect_mode(ctx, weights, s1, s2, mode)
        pt, ct, tt = collect_mode(ctx, weights, s2, n, mode)
        for mg in (0.0, 1.0, 2.0, 3.0):
            sv = eval_from(pv, cv, tv, mg)
            st = eval_from(pt, ct, tt, mg)
            comb[f"{mode}|{mg}"] = {"val": sv, "test": st}
            print(f"  {mode:<12s} mg>{mg:<3.1f} VAL n={sv['n']:>6,} cov={sv['cov']*100:5.1f}% "
                  f"acc={sv['acc']:.4f} z={sv['z']:+.2f} | TEST n={st['n']:>6,} "
                  f"cov={st['cov']*100:5.1f}% acc={st['acc']:.4f} z={st['z']:+.2f} "
                  f"EV={st['ev']*100:+.2f}%")

    # ---------- CONSENSUS: require k rules to agree ----------
    universes = {
        "three_main": {"zig3_repeat", "run8_repeat", "pat6_111111"},
        "strong_struct": _MODE["structural"],
        "all": {r["id"] for r in RULES},
    }
    print("\nCONSENSUS modes (unweighted, need >=k rules agree) — VAL / TEST:")
    cons = {}
    for uname, ids in universes.items():
        for k in (2, 3):
            pv, cv, tv = consensus_collect(ctx, s1, s2, ids, k)
            pt, ct, tt = consensus_collect(ctx, s2, n, ids, k)
            sv = eval_from(pv, cv, tv, 0.0)
            st = eval_from(pt, ct, tt, 0.0)
            cons[f"{uname}|k{k}"] = {"val": sv, "test": st}
            print(f"  {uname:<14s} k>={k} VAL n={sv['n']:>6,} cov={sv['cov']*100:5.1f}% "
                  f"acc={sv['acc']:.4f} z={sv['z']:+.2f} | TEST n={st['n']:>6,} "
                  f"cov={st['cov']*100:5.1f}% acc={st['acc']:.4f} z={st['z']:+.2f} "
                  f"EV={st['ev']*100:+.2f}%")

    RULES_JSON.write_text(json.dumps(
        {"payout": PAYOUT, "breakeven": BREAKEVEN, "weights": weights,
         "rules": RULES, "consensus": cons,
         "universes": {k: sorted(v) for k, v in universes.items()}}, indent=2))
    print(f"\nsaved calibrated rules -> {RULES_JSON}")
    print("\nNEXT-DRAW prediction (structural mode, calibrated on all history):")
    w_all = calibrate(ctx, 2, n)
    p = predict_next(D, MIN, "structural", w_all)
    print(f"  -> {p['pred']}  (confidence margin {p['confidence']:.2f})  votes={p['votes']}")


if __name__ == "__main__":
    main()
