"""
Weather Trader Bot — METAR Observations
Fetches METAR data from aviationweather.gov with timezone-aware filtering.
Midnight exclusion matches Wunderground resolution convention.
"""

import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import httpx
from ensemble import c_to_f

log = logging.getLogger("metar")

METAR_API = "https://aviationweather.gov/api/data/metar"


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

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()

        if not data:
            log.warning(f"{city['name']}: No METAR data returned")
            return _empty_result()

        tz = ZoneInfo(city["tz"])
        current_temp_c = data[0].get("temp")
        latest_raw = data[0].get("rawOb", "")[:50]

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

        log.info(
            f"{city['name']}: METAR current={current_temp_c}°C, "
            f"day_high={day_high_market}{'°F' if city['unit'] == 'F' else '°C'} "
            f"({len(day_obs)} obs)"
        )

        return {
            "current_temp_c": current_temp_c,
            "day_high_c": day_high_c,
            "day_high_market": day_high_market,
            "observation_count": len(day_obs),
            "latest_raw": latest_raw,
        }

    except Exception as e:
        log.error(f"{city['name']}: METAR fetch failed: {e}")
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
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    temp_c = data.get("current", {}).get("temperature_2m")
    log.warning(f"{city['name']}: Using Open-Meteo fallback: {temp_c}°C")

    return {
        "current_temp_c": temp_c,
        "day_high_c": None,
        "day_high_market": None,
        "observation_count": 0,
        "latest_raw": "Open-Meteo fallback",
    }


def _empty_result() -> dict:
    return {
        "current_temp_c": None,
        "day_high_c": None,
        "day_high_market": None,
        "observation_count": 0,
        "latest_raw": "",
    }
