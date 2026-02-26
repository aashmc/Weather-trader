"""
Tomorrow forward-test quality + calibration state.

Maintains:
  - Per-market forecast snapshots (by lead bucket)
  - City+lead empirical forecast error history
  - Resolved forward-test metrics
  - Daily report send watermark
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from market import parse_bracket
from metar import fetch_metar

log = logging.getLogger("tomorrow_quality")

STATE_FILE = Path("tomorrow_quality_state.json")
HOURLY_LOG_FILE = Path("forward_hourly_hit_log.jsonl")
HOURLY_PRED_FILE = Path("forward_hourly_predictions.jsonl")


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError):
            log.warning("Tomorrow quality state corrupted, starting fresh")
    return {
        "snapshots": {},        # {city:date: {buckets:{near/mid/far}, latest:{...}}}
        "market_series": {},    # {city:date: {opened_at_utc, hours:{idx: {...}}}}
        "errors": {},           # {city: {near:[], mid:[], far:[]}}
        "resolved_metrics": {}, # {city: [records]}
        "resolved_hourly": [],  # [{market_key, city, date, hour_since_open, hit, ...}]
        "resolved_keys": {},    # {city:date: winner}
        "report_meta": {"last_sent_date_utc": ""},
    }


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except IOError as exc:
        log.warning("Failed to save tomorrow quality state: %s", exc)


def _normalize_dist(dist_market: dict[int, float]) -> dict[int, float]:
    clean: dict[int, float] = {}
    for k, v in (dist_market or {}).items():
        try:
            key = int(k)
            val = float(v)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        clean[key] = clean.get(key, 0.0) + val
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean.items()}


def _normalize_probs(bracket_probs: dict[str, float]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for label, val in (bracket_probs or {}).items():
        try:
            p = float(val)
        except (TypeError, ValueError):
            continue
        if p <= 0:
            continue
        clean[str(label)] = p
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean.items()}


def _append_jsonl(path: Path, row: dict):
    try:
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except IOError as exc:
        log.warning("Failed writing %s: %s", path, exc)


def _rolling_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(max(0.0, var))


def get_lead_bucket(city_tz: str, date_str: str) -> tuple[str, float]:
    tz = ZoneInfo(city_tz)
    now_local = datetime.now(timezone.utc).astimezone(tz)
    target_local = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
    lead_hours = (target_local - now_local).total_seconds() / 3600.0

    if lead_hours >= config.LEAD_BUCKET_FAR_HOURS:
        return "far", lead_hours
    if lead_hours >= config.LEAD_BUCKET_MID_HOURS:
        return "mid", lead_hours
    return "near", lead_hours


def record_forward_snapshot(
    *,
    city_name: str,
    date_str: str,
    lead_bucket: str,
    forecast_max_market: float | None,
    dist_market: dict[int, float],
    bracket_probs: dict[str, float],
    probability_source: str,
    probabilistic: bool,
):
    state = _load_state()
    key = f"{city_name}:{date_str}"
    ts = datetime.now(timezone.utc).isoformat()

    snap = {
        "timestamp": ts,
        "lead_bucket": lead_bucket,
        "forecast_max_market": (
            None if forecast_max_market is None else float(forecast_max_market)
        ),
        "dist_market": {str(k): float(v) for k, v in _normalize_dist(dist_market).items()},
        "bracket_probs": _normalize_probs(bracket_probs),
        "probability_source": str(probability_source or ""),
        "probabilistic": bool(probabilistic),
    }

    market_snap = state.setdefault("snapshots", {}).setdefault(key, {"buckets": {}})
    market_snap.setdefault("buckets", {})[lead_bucket] = snap
    market_snap["latest"] = snap

    # Hour-by-hour raw prediction timeline since this market was first seen.
    series = state.setdefault("market_series", {}).setdefault(
        key,
        {
            "city": city_name,
            "date": date_str,
            "opened_at_utc": ts,
            "hours": {},
        },
    )
    try:
        opened_dt = datetime.fromisoformat(str(series.get("opened_at_utc", ts)))
    except ValueError:
        opened_dt = datetime.fromisoformat(ts)
        series["opened_at_utc"] = ts
    now_dt = datetime.fromisoformat(ts)
    hour_idx = max(0, int((now_dt - opened_dt).total_seconds() // 3600))

    probs = _normalize_probs(bracket_probs)
    predicted = max(probs, key=probs.get) if probs else ""
    predicted_prob = float(probs.get(predicted, 0.0)) if predicted else 0.0
    hour_key = str(hour_idx)
    if hour_key not in series.setdefault("hours", {}):
        hour_row = {
            "timestamp": ts,
            "market_key": key,
            "city": city_name,
            "date": date_str,
            "opened_at_utc": str(series.get("opened_at_utc", ts)),
            "hour_since_open": hour_idx,
            "lead_bucket": lead_bucket,
            "forecast_max_market": (
                None if forecast_max_market is None else float(forecast_max_market)
            ),
            "predicted_bracket": predicted,
            "predicted_prob": predicted_prob,
            "probability_source": str(probability_source or ""),
            "probabilistic": bool(probabilistic),
        }
        series["hours"][hour_key] = hour_row
        _append_jsonl(HOURLY_PRED_FILE, hour_row)
    series["latest_ts"] = ts

    _save_state(state)


def _winner_temp_proxy(winner_label: str) -> float | None:
    p = parse_bracket(winner_label)
    if not p:
        return None
    kind = p.get("type")
    if kind == "single":
        return float(p["val"])
    if kind == "range":
        return (float(p["low"]) + float(p["high"])) / 2.0
    if kind == "below":
        return float(p["high"])
    if kind == "above":
        return float(p["low"])
    return None


def _score_probs(bracket_probs: dict[str, float], winner: str) -> tuple[float, float, str, float]:
    probs = _normalize_probs(bracket_probs)
    if not probs:
        return 0.0, 0.0, "", 0.0

    top_pick = max(probs, key=probs.get)
    top_pick_prob = float(probs[top_pick])
    winner_prob = float(probs.get(winner, 0.0))

    brier = 0.0
    for label, prob in probs.items():
        y = 1.0 if label == winner else 0.0
        brier += (prob - y) ** 2
    return winner_prob, brier, top_pick, top_pick_prob


async def record_forward_resolution(
    *,
    city_cfg: dict,
    date_str: str,
    winner: str,
) -> dict:
    city_name = city_cfg["name"]
    key = f"{city_name}:{date_str}"
    state = _load_state()

    if state.get("resolved_keys", {}).get(key) == winner:
        return {"recorded": False, "reason": "already_recorded", "record": None}

    market_snap = state.get("snapshots", {}).get(key, {})
    buckets = market_snap.get("buckets", {}) or {}
    snap = (
        buckets.get("near")
        or buckets.get("mid")
        or buckets.get("far")
        or market_snap.get("latest", {})
    )

    bracket_probs = snap.get("bracket_probs", {}) if isinstance(snap, dict) else {}
    forecast_max = snap.get("forecast_max_market") if isinstance(snap, dict) else None
    lead_bucket = (snap.get("lead_bucket") if isinstance(snap, dict) else "") or "near"
    probability_source = (snap.get("probability_source") if isinstance(snap, dict) else "") or ""

    # If we never captured any forward snapshot for this market/date,
    # don't create a fake zero-quality record.
    has_snapshot = bool(bracket_probs) or (forecast_max is not None)
    if not has_snapshot:
        state.setdefault("resolved_keys", {})[key] = winner
        _save_state(state)
        return {"recorded": False, "reason": "no_snapshot", "record": None}

    winner_prob, brier, top_pick, top_pick_prob = _score_probs(bracket_probs, winner)

    actual_temp = None
    try:
        metar = await fetch_metar(city_cfg, date_str)
        actual_temp = metar.get("day_high_market")
    except Exception:
        actual_temp = None
    if actual_temp is None:
        actual_temp = _winner_temp_proxy(winner)

    error = None
    if actual_temp is not None and forecast_max is not None:
        error = float(actual_temp) - float(forecast_max)

    if actual_temp is not None:
        city_errs = state.setdefault("errors", {}).setdefault(
            city_name, {"near": [], "mid": [], "far": []}
        )
        for bkt, snap_bkt in buckets.items():
            fm = snap_bkt.get("forecast_max_market")
            if fm is None:
                continue
            bkt_err = float(actual_temp) - float(fm)
            hist = city_errs.setdefault(bkt, [])
            hist.append(float(bkt_err))
            city_errs[bkt] = hist[-int(config.FORWARD_ERROR_WINDOW) :]

    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city_name,
        "date": date_str,
        "winner": winner,
        "winner_prob": float(winner_prob),
        "brier": float(brier),
        "top_pick": top_pick,
        "top_pick_prob": float(top_pick_prob),
        "top_pick_hit": bool(top_pick and top_pick == winner),
        "lead_bucket": lead_bucket,
        "forecast_max_market": forecast_max,
        "actual_temp_market": actual_temp,
        "error": error,
        "probability_source": probability_source,
    }

    series = state.get("market_series", {}).get(key, {}) or {}
    hourly_map = series.get("hours", {}) if isinstance(series, dict) else {}
    hourly_eval = []
    for hk, hv in sorted(
        (hourly_map or {}).items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 10**9,
    ):
        if not isinstance(hv, dict):
            continue
        pred = str(hv.get("predicted_bracket", "") or "")
        h = int(hv.get("hour_since_open", int(hk) if str(hk).isdigit() else 0))
        pp = float(hv.get("predicted_prob", 0.0) or 0.0)
        row = {
            "market_key": key,
            "city": city_name,
            "date": date_str,
            "winner": winner,
            "hour_since_open": h,
            "snapshot_ts": str(hv.get("timestamp", "")),
            "predicted_bracket": pred,
            "predicted_prob": pp,
            "hit": bool(pred and pred == winner),
            "probability_source": str(hv.get("probability_source", "") or ""),
            "probabilistic": bool(hv.get("probabilistic", False)),
        }
        hourly_eval.append(row)
        _append_jsonl(HOURLY_LOG_FILE, row)

    rec["hourly_points"] = len(hourly_eval)
    if hourly_eval:
        h0 = min(hourly_eval, key=lambda r: int(r.get("hour_since_open", 0)))
        hl = max(hourly_eval, key=lambda r: int(r.get("hour_since_open", 0)))
        rec["h0_hit"] = bool(h0.get("hit"))
        rec["h_last_hit"] = bool(hl.get("hit"))

    city_hist = state.setdefault("resolved_metrics", {}).setdefault(city_name, [])
    city_hist = [h for h in city_hist if h.get("date") != date_str]
    city_hist.append(rec)
    city_hist.sort(key=lambda x: x.get("date", ""))
    if len(city_hist) > 1500:
        city_hist = city_hist[-1500:]
    state["resolved_metrics"][city_name] = city_hist

    resolved_hourly = state.setdefault("resolved_hourly", [])
    resolved_hourly = [r for r in resolved_hourly if r.get("market_key") != key]
    resolved_hourly.extend(hourly_eval)
    if len(resolved_hourly) > 200_000:
        resolved_hourly = resolved_hourly[-200_000:]
    state["resolved_hourly"] = resolved_hourly

    state.setdefault("resolved_keys", {})[key] = winner
    _save_state(state)
    return {"recorded": True, "reason": "ok", "record": rec}


def apply_error_calibration(
    *,
    city_name: str,
    lead_bucket: str,
    dist_market: dict[int, float],
) -> tuple[dict[int, float], dict]:
    base = _normalize_dist(dist_market)
    if not base:
        return {}, {
            "used": False,
            "reason": "empty_distribution",
            "samples": 0,
            "mean_error": 0.0,
            "sd_error": 0.0,
        }

    state = _load_state()
    city_errs = state.get("errors", {}).get(city_name, {})
    errors = list(city_errs.get(lead_bucket, []))[-int(config.FORWARD_ERROR_WINDOW) :]
    mean_e, sd_e = _rolling_stats(errors)
    meta = {
        "used": False,
        "reason": "insufficient_samples",
        "samples": len(errors),
        "mean_error": float(mean_e),
        "sd_error": float(sd_e),
        "bucket": lead_bucket,
    }

    if len(errors) < int(config.FORWARD_ERROR_MIN_SAMPLES):
        return base, meta

    max_samples = max(1, int(config.FORWARD_ERROR_MAX_SAMPLES_FOR_CONVOLUTION))
    kernel = errors[-max_samples:]
    n = len(kernel)
    if n <= 0:
        return base, meta

    out: dict[int, float] = {}
    for temp, p in base.items():
        for e in kernel:
            shifted = int(round(float(temp) + float(e)))
            out[shifted] = out.get(shifted, 0.0) + (float(p) / n)

    calibrated = _normalize_dist(out)
    meta["used"] = True
    meta["reason"] = "empirical_error_convolution"
    return calibrated, meta


def build_daily_report_message(now_utc: datetime | None = None) -> dict | None:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    # Send once per UTC day after configured report time.
    send_after = now.replace(
        hour=int(config.FORWARD_DAILY_REPORT_UTC_HOUR),
        minute=int(config.FORWARD_DAILY_REPORT_UTC_MINUTE),
        second=0,
        microsecond=0,
    )
    if now < send_after:
        return None

    today_utc = now.date().isoformat()
    state = _load_state()
    report_meta = state.setdefault("report_meta", {})
    if report_meta.get("last_sent_date_utc") == today_utc:
        return None

    report_day = (now.date() - timedelta(days=1)).isoformat()
    rows = []
    for city_rows in (state.get("resolved_metrics") or {}).values():
        for rec in city_rows or []:
            if str(rec.get("date", "")) == report_day:
                rows.append(rec)
    rows.sort(key=lambda x: x.get("timestamp", ""))

    if not rows:
        message = (
            f"📘 <b>Forward Test Daily Update ({report_day} UTC)</b>\n"
            f"  No resolved markets were logged."
        )
        payload = {
            "message": message,
            "report_day": report_day,
            "count": 0,
        }
    else:
        n = len(rows)
        hit = sum(1 for r in rows if bool(r.get("top_pick_hit")))
        winner_prob_avg = sum(float(r.get("winner_prob", 0.0)) for r in rows) / n
        abs_errors = [abs(float(r.get("error"))) for r in rows if r.get("error") is not None]
        mae = (sum(abs_errors) / len(abs_errors)) if abs_errors else None

        by_city: dict[str, list] = {}
        for r in rows:
            city = str(r.get("city") or r.get("city_name") or "")
            city = city or "Unknown"
            by_city.setdefault(city, []).append(r)

        city_lines = []
        for city in sorted(by_city):
            cr = by_city[city]
            cn = len(cr)
            ch = sum(1 for x in cr if bool(x.get("top_pick_hit")))
            cwp = sum(float(x.get("winner_prob", 0.0)) for x in cr) / cn
            cerr = [abs(float(x.get("error"))) for x in cr if x.get("error") is not None]
            cmae = (sum(cerr) / len(cerr)) if cerr else None
            mae_str = f"{cmae:.2f}" if cmae is not None else "n/a"
            city_lines.append(
                f"  - {city}: {ch}/{cn} top-pick, avg winner p={cwp*100:.1f}%, MAE={mae_str}"
            )

        mae_str = f"{mae:.2f}" if mae is not None else "n/a"
        message = (
            f"📘 <b>Forward Test Daily Update ({report_day} UTC)</b>\n"
            f"  Resolved: {n}\n"
            f"  Top-pick hit rate: {hit}/{n} ({(hit/n)*100:.1f}%)\n"
            f"  Avg winner probability: {winner_prob_avg*100:.1f}%\n"
            f"  MAE (forecast max vs actual): {mae_str}\n"
            f"  <b>City inference:</b>\n"
            + "\n".join(city_lines)
        )
        payload = {
            "message": message,
            "report_day": report_day,
            "count": n,
            "hit_rate": hit / n if n else 0.0,
            "avg_winner_prob": winner_prob_avg,
            "mae": mae,
        }

    # Add cumulative hourly hit-rate summary (hour since market-open).
    hourly = state.get("resolved_hourly", []) or []
    agg: dict[int, dict] = {}
    for r in hourly:
        try:
            h = int(r.get("hour_since_open", 0))
        except (TypeError, ValueError):
            continue
        hit = 1 if bool(r.get("hit")) else 0
        pp = float(r.get("predicted_prob", 0.0) or 0.0)
        slot = agg.setdefault(h, {"n": 0, "hit": 0, "p_sum": 0.0})
        slot["n"] += 1
        slot["hit"] += hit
        slot["p_sum"] += pp

    hourly_lines = []
    hourly_payload = []
    for h in sorted(agg.keys()):
        slot = agg[h]
        n = int(slot["n"])
        if n <= 0:
            continue
        hr = float(slot["hit"]) / n
        ap = float(slot["p_sum"]) / n
        hourly_payload.append(
            {
                "hour_since_open": h,
                "samples": n,
                "hit_rate": hr,
                "avg_predicted_prob": ap,
            }
        )
        if h <= 24 and n >= 3:
            hourly_lines.append(
                f"  h{h}: {hr*100:.1f}% ({slot['hit']}/{n}), avg p={ap*100:.1f}%"
            )

    if hourly_lines:
        message = (
            payload["message"]
            + "\n  <b>Hourly hit-rate (cumulative):</b>\n"
            + "\n".join(hourly_lines[:12])
        )
        payload["message"] = message
    payload["hourly_hit_rate"] = hourly_payload

    return payload


def mark_daily_report_sent(sent_date_utc: str | None = None):
    state = _load_state()
    today = sent_date_utc or datetime.now(timezone.utc).date().isoformat()
    state.setdefault("report_meta", {})["last_sent_date_utc"] = str(today)
    _save_state(state)
