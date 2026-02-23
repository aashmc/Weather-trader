"""
Weather Trader Bot — Rolling Calibration + Model Quality Gates
Implements:
  - Item #3: rolling calibration by city and lead bucket
  - Item #5: model quality gates from resolved outcomes
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from market import parse_bracket
from metar import fetch_metar

log = logging.getLogger("quality")

STATE_FILE = Path("model_quality_state.json")


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            log.warning("Quality state file corrupted, starting fresh")
    return {
        "forecast_snapshots": {},  # {city:date: {ensemble_mean, lead_bucket, probs, ts}}
        "bias_errors": {},         # {city: {near:[], mid:[], far:[]}}
        "resolved_metrics": {},    # {city: [records]}
    }


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except IOError as e:
        log.warning(f"Failed to save quality state: {e}")


def get_lead_bucket(city_tz: str, date_str: str) -> tuple[str, float]:
    """
    Bucket lead time by hours until local market-date midnight.
    Returns (bucket, lead_hours).
    """
    tz = ZoneInfo(city_tz)
    now_local = datetime.now(timezone.utc).astimezone(tz)
    target_local = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
    lead_hours = (target_local - now_local).total_seconds() / 3600.0

    if lead_hours >= config.LEAD_BUCKET_FAR_HOURS:
        return "far", lead_hours
    if lead_hours >= config.LEAD_BUCKET_MID_HOURS:
        return "mid", lead_hours
    return "near", lead_hours


def _rolling_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def get_dynamic_bias(
    city_name: str,
    base_mean: float,
    base_sd: float,
    lead_bucket: str,
) -> tuple[float, float, dict]:
    """
    Return blended rolling + static bias parameters.
    """
    info = {
        "enabled": config.ROLLING_CALIBRATION_ENABLED,
        "bucket": lead_bucket,
        "samples": 0,
        "used_rolling": False,
    }
    if not config.ROLLING_CALIBRATION_ENABLED:
        return base_mean, base_sd, info

    state = _load_state()
    city_errs = state.get("bias_errors", {}).get(city_name, {})
    vals = city_errs.get(lead_bucket, [])[-config.ROLLING_CAL_WINDOW:]
    info["samples"] = len(vals)
    if len(vals) < config.ROLLING_CAL_MIN_SAMPLES:
        return base_mean, base_sd, info

    roll_mean, roll_sd = _rolling_stats(vals)
    alpha = config.ROLLING_CAL_BLEND
    blended_mean = (1 - alpha) * base_mean + alpha * roll_mean
    blended_sd = max(
        config.ROLLING_CAL_SD_FLOOR,
        (1 - alpha) * base_sd + alpha * roll_sd,
    )
    info["used_rolling"] = True
    info["rolling_mean"] = roll_mean
    info["rolling_sd"] = roll_sd
    return blended_mean, blended_sd, info


def record_forecast_snapshot(
    city_name: str,
    date_str: str,
    ensemble_mean: float,
    lead_bucket: str,
    corrected_probs: dict[str, float],
):
    """
    Keep most-recent forecast snapshot per market for later quality scoring.
    """
    state = _load_state()
    key = f"{city_name}:{date_str}"
    ts = datetime.now(timezone.utc).isoformat()
    market_snap = state.setdefault("forecast_snapshots", {}).setdefault(key, {"buckets": {}})
    market_snap["buckets"][lead_bucket] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ensemble_mean": ensemble_mean,
        "lead_bucket": lead_bucket,
        "corrected_probs": corrected_probs,
    }
    market_snap["latest"] = {
        "timestamp": ts,
        "ensemble_mean": ensemble_mean,
        "lead_bucket": lead_bucket,
        "corrected_probs": corrected_probs,
    }
    _save_state(state)


def _winner_temp_proxy(winner_label: str) -> float | None:
    """Fallback proxy temp from winner bracket label when METAR is unavailable."""
    p = parse_bracket(winner_label)
    if not p:
        return None
    if p.get("type") == "single":
        return float(p["val"])
    if p.get("type") == "range":
        return (p["low"] + p["high"]) / 2.0
    if p.get("type") == "below":
        return float(p["high"])
    if p.get("type") == "above":
        return float(p["low"])
    return None


def _score_probs(corrected_probs: dict[str, float], winner: str) -> tuple[float, float, float]:
    if not corrected_probs:
        return 0.0, 0.0, 0.0

    winner_prob = float(corrected_probs.get(winner, 0.0))
    p = max(winner_prob, 1e-9)
    logloss = -math.log(p)

    # Multiclass Brier score
    brier = 0.0
    for label, prob in corrected_probs.items():
        y = 1.0 if label == winner else 0.0
        brier += (prob - y) ** 2
    return winner_prob, brier, logloss


async def record_resolution_quality(city_cfg: dict, date_str: str, winner: str) -> dict:
    """
    Update quality and rolling calibration state when a market resolves.
    """
    city_name = city_cfg["name"]
    state = _load_state()
    key = f"{city_name}:{date_str}"
    snap_market = state.get("forecast_snapshots", {}).get(key, {})
    buckets = snap_market.get("buckets", {})
    snap = buckets.get("near") or buckets.get("mid") or buckets.get("far") or snap_market.get("latest", {})

    corrected_probs = snap.get("corrected_probs", {})
    ensemble_mean = snap.get("ensemble_mean")
    lead_bucket = snap.get("lead_bucket", "near")

    winner_prob, brier, logloss = _score_probs(corrected_probs, winner)

    actual_temp = None
    try:
        metar = await fetch_metar(city_cfg, date_str)
        actual_temp = metar.get("day_high_market")
    except Exception:
        actual_temp = None
    if actual_temp is None:
        actual_temp = _winner_temp_proxy(winner)

    error = None
    if actual_temp is not None:
        city_errs = state.setdefault("bias_errors", {}).setdefault(
            city_name, {"near": [], "mid": [], "far": []}
        )

        # Update calibration error history for each available lead bucket snapshot.
        for bkt, snap_bkt in buckets.items():
            ens_mean = snap_bkt.get("ensemble_mean")
            if ens_mean is None:
                continue
            bkt_err = float(actual_temp) - float(ens_mean)
            city_errs.setdefault(bkt, []).append(bkt_err)
            city_errs[bkt] = city_errs[bkt][-config.ROLLING_CAL_WINDOW:]

        if ensemble_mean is not None:
            error = float(actual_temp) - float(ensemble_mean)

    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": date_str,
        "winner": winner,
        "winner_prob": winner_prob,
        "brier": brier,
        "logloss": logloss,
        "lead_bucket": lead_bucket,
        "ensemble_mean": ensemble_mean,
        "actual_temp": actual_temp,
        "error": error,
    }
    hist = state.setdefault("resolved_metrics", {}).setdefault(city_name, [])
    hist.append(rec)
    if len(hist) > 500:
        del hist[:-500]

    _save_state(state)
    return rec


def check_quality_gate(city_name: str) -> tuple[bool, str, dict]:
    """
    Hard gate: block trading when recent quality drifts below configured thresholds.
    """
    if not config.QUALITY_GATE_ENABLED:
        return True, "disabled", {"enabled": False}

    state = _load_state()
    hist = state.get("resolved_metrics", {}).get(city_name, [])[-config.QUALITY_WINDOW:]
    if len(hist) < config.QUALITY_MIN_SAMPLES:
        return True, "insufficient_samples", {
            "enabled": True,
            "samples": len(hist),
            "required": config.QUALITY_MIN_SAMPLES,
        }

    avg_brier = sum(h.get("brier", 0.0) for h in hist) / len(hist)
    avg_winprob = sum(h.get("winner_prob", 0.0) for h in hist) / len(hist)

    reasons = []
    if avg_brier > config.QUALITY_MAX_BRIER:
        reasons.append(f"brier {avg_brier:.3f}>{config.QUALITY_MAX_BRIER:.3f}")
    if avg_winprob < config.QUALITY_MIN_WINNER_PROB:
        reasons.append(f"winner_prob {avg_winprob:.3f}<{config.QUALITY_MIN_WINNER_PROB:.3f}")

    passed = len(reasons) == 0
    return passed, ("; ".join(reasons) if reasons else "ok"), {
        "enabled": True,
        "samples": len(hist),
        "avg_brier": avg_brier,
        "avg_winner_prob": avg_winprob,
    }
