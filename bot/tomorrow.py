"""
Tomorrow.io forecast interface for forward testing.

Fetches hourly temperature timelines and computes the daily max forecast
for the exact market station location.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import config
from http_retry import get_json

log = logging.getLogger("tomorrow")


def _c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def _to_market_units(temp_c: float, unit: str) -> float:
    return _c_to_f(temp_c) if unit == "F" else temp_c


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


async def fetch_tomorrow_daily_max(city: dict, date_str: str) -> dict:
    """
    Fetch Tomorrow.io hourly forecast and return predicted max temp for date_str.
    """
    api_key = (config.TOMORROW_API_KEY or "").strip()
    if not api_key:
        raise ValueError("TOMORROW_API_KEY not configured")

    tz_name = city["tz"]
    tz = ZoneInfo(tz_name)
    local_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
    local_end = local_start + timedelta(days=1)

    params = {
        "location": f"{city['lat']},{city['lon']}",
        "fields": "temperature",
        "timesteps": config.TOMORROW_TIMESTEP,
        "startTime": local_start.isoformat(),
        "endTime": local_end.isoformat(),
        "timezone": tz_name,
        "units": "metric",
        "apikey": api_key,
    }
    url = f"{config.TOMORROW_TIMELINES_API}?{urlencode(params)}"

    data = await get_json(
        url,
        timeout=20,
        attempts=config.API_RETRY_ATTEMPTS,
        base_backoff_seconds=config.API_RETRY_BACKOFF_SECONDS,
    )

    timelines = data.get("data", {}).get("timelines", [])
    if not timelines:
        raise ValueError("Tomorrow.io returned no timelines")

    intervals = timelines[0].get("intervals", []) or []
    if not intervals:
        raise ValueError("Tomorrow.io returned no intervals")

    points = []
    for item in intervals:
        start = item.get("startTime")
        values = item.get("values", {}) or {}
        temp_c = values.get("temperature")
        if start is None or temp_c is None:
            continue
        try:
            temp_c = float(temp_c)
        except (TypeError, ValueError):
            continue

        dt_utc = _parse_iso(start).astimezone(timezone.utc)
        dt_local = dt_utc.astimezone(tz)
        if dt_local.date().isoformat() != date_str:
            continue

        points.append(
            {
                "time_utc": dt_utc.isoformat(),
                "time_local": dt_local.isoformat(),
                "temp_c": temp_c,
                "temp_market": _to_market_units(temp_c, city["unit"]),
            }
        )

    if not points:
        raise ValueError(f"No Tomorrow.io forecast points for {city['name']} {date_str}")

    max_point = max(points, key=lambda p: p["temp_market"])
    max_market = float(max_point["temp_market"])
    max_c = float(max_point["temp_c"])

    log.info(
        "%s: Tomorrow.io %s max %.1f%s (%d points)",
        city["name"],
        date_str,
        max_market,
        "°F" if city["unit"] == "F" else "°C",
        len(points),
    )

    return {
        "city": city["name"],
        "date": date_str,
        "source": "tomorrow.io",
        "timestep": config.TOMORROW_TIMESTEP,
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "points": points,
        "point_count": len(points),
        "max_temp_c": max_c,
        "max_temp_market": max_market,
        "max_time_utc": max_point["time_utc"],
        "max_time_local": max_point["time_local"],
        "timezone": tz_name,
    }
