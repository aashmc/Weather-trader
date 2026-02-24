"""
Weather Trader Bot — Late Session Guard
Builds city-specific late-entry cutoffs from historical hourly temperatures.
"""

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

import config

log = logging.getLogger("late_guard")

STATE_FILE = Path("late_guard_state.json")
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            log.warning("Late guard state corrupted, rebuilding")
    return {"cutoffs": {}}


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except IOError as e:
        log.warning(f"Late guard state save failed: {e}")


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] + (vals[hi] - vals[lo]) * frac


def _fallback_cutoff(city_key: str) -> int:
    return int(
        config.LATE_GUARD_FALLBACK_CUTOFF_HOURS.get(
            city_key,
            config.LATE_GUARD_DEFAULT_CUTOFF_HOUR,
        )
    )


def _state_key(city_key: str, month: int) -> str:
    return f"{city_key}:{month:02d}"


def _is_fresh(updated_at: str) -> bool:
    if not updated_at:
        return False
    try:
        ts = datetime.fromisoformat(updated_at)
    except ValueError:
        return False
    age = datetime.now(timezone.utc) - ts
    return age.total_seconds() < config.LATE_GUARD_REFRESH_HOURS * 3600


async def _fetch_hourly_history(city: dict) -> tuple[list[str], list[float]]:
    tz = ZoneInfo(city["tz"])
    today_local = datetime.now(timezone.utc).astimezone(tz).date()
    start_date = (today_local - timedelta(days=config.LATE_GUARD_HISTORY_DAYS)).isoformat()
    end_date = (today_local - timedelta(days=1)).isoformat()

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": "temperature_2m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": city["tz"],
    }

    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(ARCHIVE_API, params=params)
        r.raise_for_status()
        data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", []) or []
    temps = hourly.get("temperature_2m", []) or []
    return times, temps


def _compute_daily_peak_hours(times: list[str], temps: list[float]) -> dict[str, int]:
    """
    Returns {YYYY-MM-DD: hour_of_first_daily_max}
    """
    by_day = {}
    for ts, temp in zip(times, temps):
        if not ts or temp is None:
            continue
        day = ts[:10]
        try:
            hour = int(ts[11:13])
            temp_val = float(temp)
        except (ValueError, TypeError):
            continue

        rec = by_day.get(day)
        if rec is None or temp_val > rec["mx"]:
            by_day[day] = {"mx": temp_val, "h": hour}
        elif abs(temp_val - rec["mx"]) <= 1e-9 and hour < rec["h"]:
            by_day[day]["h"] = hour

    return {d: v["h"] for d, v in by_day.items()}


def _derive_cutoff_from_hours(
    peak_hours: dict[str, int],
    target_month: int,
) -> tuple[int, dict]:
    all_hours = list(peak_hours.values())
    month_hours = [
        h
        for day, h in peak_hours.items()
        if int(day[5:7]) == target_month
    ]

    source = "month"
    chosen = month_hours
    if len(month_hours) < config.LATE_GUARD_MIN_MONTH_SAMPLES:
        source = "all"
        chosen = all_hours

    if len(chosen) < config.LATE_GUARD_MIN_GLOBAL_SAMPLES:
        return 0, {
            "ok": False,
            "source": "fallback",
            "samples": len(chosen),
            "samples_month": len(month_hours),
            "samples_all": len(all_hours),
        }

    qval = _quantile(chosen, config.LATE_GUARD_CUTOFF_QUANTILE)
    if qval is None:
        return 0, {
            "ok": False,
            "source": "fallback",
            "samples": len(chosen),
            "samples_month": len(month_hours),
            "samples_all": len(all_hours),
        }

    cutoff = int(round(qval))
    cutoff = max(config.LATE_GUARD_MIN_CUTOFF_HOUR, min(config.LATE_GUARD_MAX_CUTOFF_HOUR, cutoff))
    return cutoff, {
        "ok": True,
        "source": source,
        "samples": len(chosen),
        "samples_month": len(month_hours),
        "samples_all": len(all_hours),
        "quantile": round(float(qval), 3),
    }


async def get_city_cutoff(
    city_key: str,
    city: dict,
    market_month: int,
    refresh_from_api: bool = True,
) -> dict:
    """
    Returns cached or recomputed late-entry cutoff metadata.
    """
    fallback = _fallback_cutoff(city_key)
    if not config.LATE_GUARD_ENABLED:
        return {
            "enabled": False,
            "cutoff_hour": fallback,
            "freeze_hour": min(23, fallback + config.LATE_GUARD_FREEZE_HOURS_AFTER_CUTOFF),
            "source": "disabled",
            "samples": 0,
        }

    state = _load_state()
    key = _state_key(city_key, market_month)
    cached = state.get("cutoffs", {}).get(key, {})
    if cached and _is_fresh(cached.get("updated_at", "")):
        return cached

    if not refresh_from_api:
        if cached:
            return cached
        return {
            "enabled": True,
            "cutoff_hour": fallback,
            "freeze_hour": min(23, fallback + config.LATE_GUARD_FREEZE_HOURS_AFTER_CUTOFF),
            "source": "fallback_no_refresh",
            "samples": 0,
            "samples_month": 0,
            "samples_all": 0,
            "quantile_hour": float(fallback),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        times, temps = await _fetch_hourly_history(city)
        peak_hours = _compute_daily_peak_hours(times, temps)
        cutoff, meta = _derive_cutoff_from_hours(peak_hours, market_month)
        if not meta.get("ok"):
            cutoff = fallback

        payload = {
            "enabled": True,
            "cutoff_hour": cutoff,
            "freeze_hour": min(23, cutoff + config.LATE_GUARD_FREEZE_HOURS_AFTER_CUTOFF),
            "source": meta.get("source", "fallback"),
            "samples": int(meta.get("samples", 0)),
            "samples_month": int(meta.get("samples_month", 0)),
            "samples_all": int(meta.get("samples_all", 0)),
            "quantile_hour": float(meta.get("quantile", cutoff)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state.setdefault("cutoffs", {})[key] = payload
        _save_state(state)
        return payload
    except Exception as e:
        log.warning(f"{city['name']}: late cutoff history fetch failed: {e}")
        payload = {
            "enabled": True,
            "cutoff_hour": fallback,
            "freeze_hour": min(23, fallback + config.LATE_GUARD_FREEZE_HOURS_AFTER_CUTOFF),
            "source": "fallback_error",
            "samples": 0,
            "samples_month": 0,
            "samples_all": 0,
            "quantile_hour": float(fallback),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state.setdefault("cutoffs", {})[key] = payload
        _save_state(state)
        return payload


def build_timing_context(
    city: dict,
    date_str: str,
    cutoff_meta: dict,
    observed_day_high_market: float | None,
) -> dict:
    """
    Build per-cycle late-guard context passed into strategy.
    """
    tz = ZoneInfo(city["tz"])
    now_local = datetime.now(timezone.utc).astimezone(tz)
    market_date_local = datetime.strptime(date_str, "%Y-%m-%d").date()

    cutoff_hour = int(cutoff_meta.get("cutoff_hour", config.LATE_GUARD_DEFAULT_CUTOFF_HOUR))
    freeze_hour = int(cutoff_meta.get("freeze_hour", min(23, cutoff_hour + config.LATE_GUARD_FREEZE_HOURS_AFTER_CUTOFF)))

    applies_today = now_local.date() == market_date_local
    now_hour_float = now_local.hour + now_local.minute / 60.0

    phase = "inactive"
    if applies_today:
        if now_hour_float >= freeze_hour:
            phase = "freeze"
        elif now_hour_float >= cutoff_hour:
            phase = "after_cutoff"
        else:
            phase = "normal"

    market_end_local = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz) + timedelta(days=1)
    minutes_to_close = int((market_end_local - now_local).total_seconds() / 60.0)

    return {
        "enabled": bool(cutoff_meta.get("enabled", True)),
        "applies_today": applies_today,
        "phase": phase,
        "now_local_iso": now_local.isoformat(),
        "now_hour": round(now_hour_float, 2),
        "cutoff_hour": cutoff_hour,
        "freeze_hour": freeze_hour,
        "minutes_to_close": minutes_to_close,
        "observed_day_high_market": observed_day_high_market,
        "source": cutoff_meta.get("source", "unknown"),
        "samples": int(cutoff_meta.get("samples", 0)),
        "samples_month": int(cutoff_meta.get("samples_month", 0)),
        "samples_all": int(cutoff_meta.get("samples_all", 0)),
        "quantile_hour": cutoff_meta.get("quantile_hour"),
    }
