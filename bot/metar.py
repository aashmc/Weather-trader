"""
Weather Trader Bot — METAR Observations
Fetches METAR data from aviationweather.gov with timezone-aware filtering.
Midnight exclusion matches Wunderground resolution convention.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from copy import deepcopy
import config
from ensemble import c_to_f
from http_retry import get_json

log = logging.getLogger("metar")

METAR_API = "https://aviationweather.gov/api/data/metar"
_METAR_CACHE: dict[str, dict] = {}


async def fetch_metar(city: dict, date_str: str) -> dict:
    """
    Fetch METAR observations and compute day high for a specific local date.
    
    Returns:
        {
            "current_temp_c": float or None,
            "day_high_c": float or None,
            "day_high_market": float or None,   # in market units (°F or °C)
            "observation_count": int,
            "latest_raw": str,
        }
    """
    url = f"{METAR_API}?ids={city['icao']}&format=json&hours=72"
    cache_key = f"{city['name']}:{date_str}"
    now = datetime.now(timezone.utc)

    try:
        data = await get_json(
            url,
            timeout=15,
            attempts=config.API_RETRY_ATTEMPTS,
            base_backoff_seconds=config.API_RETRY_BACKOFF_SECONDS,
        )

        if not data:
            log.warning(f"{city['name']}: No METAR data returned")
            return _empty_result()

        tz = ZoneInfo(city["tz"])
        current_temp_c = data[0].get("temp")
        latest_raw = data[0].get("rawOb", "")[:50]
        latest_obs_time = next((obs.get("obsTime") for obs in data if obs.get("obsTime")), None)
        latest_obs_dt = (
            datetime.fromtimestamp(latest_obs_time, tz=timezone.utc)
            if latest_obs_time is not None
            else None
        )
        age_minutes = (
            (now - latest_obs_dt).total_seconds() / 60.0
            if latest_obs_dt is not None
            else None
        )

        # Filter to local date, excluding midnight (Wunderground convention)
        day_obs = []
        for obs in data:
            obs_time = obs.get("obsTime")
            temp = obs.get("temp")
            if obs_time is None or temp is None:
                continue

            dt = datetime.fromtimestamp(obs_time, tz=timezone.utc).astimezone(tz)
            local_date = dt.strftime("%Y-%m-%d")

            if local_date != date_str:
                continue

            # Exclude exact midnight — Wunderground assigns it to previous day
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                continue

            day_obs.append(temp)

        day_high_c = max(day_obs) if day_obs else None
        day_high_market = None
        if day_high_c is not None:
            day_high_market = (
                round(c_to_f(day_high_c)) if city["unit"] == "F" else day_high_c
            )

        freshness_ok = bool(
            age_minutes is not None
            and age_minutes <= float(config.FRESHNESS_MAX_METAR_AGE_MINUTES)
        )

        payload = {
            "current_temp_c": current_temp_c,
            "day_high_c": day_high_c,
            "day_high_market": day_high_market,
            "observation_count": len(day_obs),
            "latest_raw": latest_raw,
            "source": "metar_live",
            "latest_obs_time_utc": latest_obs_dt.isoformat() if latest_obs_dt else None,
            "age_minutes": age_minutes,
            "cache_age_minutes": 0.0,
            "freshness_ok": freshness_ok,
            "fetched_at_utc": now.isoformat(),
        }
        _METAR_CACHE[cache_key] = {"fetched_at": now, "payload": deepcopy(payload)}

        log.info(
            f"{city['name']}: METAR current={current_temp_c}°C, "
            f"day_high={day_high_market}{'°F' if city['unit'] == 'F' else '°C'} "
            f"({len(day_obs)} obs)"
        )
        return payload

    except Exception as e:
        log.error(f"{city['name']}: METAR fetch failed: {e}")

    # Cache fallback
    cached = _METAR_CACHE.get(cache_key)
    if cached:
        cache_age_minutes = (now - cached["fetched_at"]).total_seconds() / 60.0
        payload = deepcopy(cached["payload"])
        payload["source"] = "metar_cache"
        payload["cache_age_minutes"] = cache_age_minutes
        payload["freshness_ok"] = bool(
            payload.get("age_minutes") is not None
            and float(payload.get("age_minutes")) <= float(config.FRESHNESS_MAX_METAR_AGE_MINUTES)
        )
        log.warning(
            f"{city['name']}: Using cached METAR ({cache_age_minutes:.1f} min old cache)"
        )
        return payload

    # Fallback to Open-Meteo current temp
    try:
        return await _fallback_current_temp(city)
    except Exception as e2:
        log.error(f"{city['name']}: Fallback weather also failed: {e2}")
        return _empty_result()


async def _fallback_current_temp(city: dict) -> dict:
    """Fallback: get current temp from Open-Meteo if METAR is down."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={city['lat']}&longitude={city['lon']}"
        f"&current=temperature_2m&timezone={city['tz']}"
    )
    data = await get_json(
        url,
        timeout=10,
        attempts=config.API_RETRY_ATTEMPTS,
        base_backoff_seconds=config.API_RETRY_BACKOFF_SECONDS,
    )

    temp_c = data.get("current", {}).get("temperature_2m")
    log.warning(f"{city['name']}: Using Open-Meteo fallback: {temp_c}°C")

    return {
        "current_temp_c": temp_c,
        "day_high_c": None,
        "day_high_market": None,
        "observation_count": 0,
        "latest_raw": "Open-Meteo fallback",
        "source": "open_meteo_fallback",
        "latest_obs_time_utc": None,
        "age_minutes": None,
        "cache_age_minutes": None,
        "freshness_ok": False,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _empty_result() -> dict:
    return {
        "current_temp_c": None,
        "day_high_c": None,
        "day_high_market": None,
        "observation_count": 0,
        "latest_raw": "",
        "source": "none",
        "latest_obs_time_utc": None,
        "age_minutes": None,
        "cache_age_minutes": None,
        "freshness_ok": False,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
