"""
Tomorrow.io forecast interface for forward testing.

Fetches hourly temperature timelines and computes the daily max forecast
for the exact market station location.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import logging
import random
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx

import config
from http_retry import get_json

log = logging.getLogger("tomorrow")
_TOMORROW_CACHE: dict[str, dict] = {}


def _c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def _to_market_units(temp_c: float, unit: str) -> float:
    return _c_to_f(temp_c) if unit == "F" else temp_c


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _cache_key(city: dict, date_str: str) -> str:
    return f"{city['name']}:{date_str}:{config.TOMORROW_TIMESTEP}"


def _extract_temp_c(values: dict) -> float | None:
    for key in ("temperature", "temperatureAvg", "temperatureP50"):
        v = values.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _extract_sigma_c(values: dict) -> float:
    def _f(key: str) -> float | None:
        v = values.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    p10, p90 = _f("temperatureP10"), _f("temperatureP90")
    if p10 is not None and p90 is not None and p90 >= p10:
        # For normal distributions, p90-p10 ~= 2*1.28155*sigma.
        return max((p90 - p10) / 2.5631, 0.05)

    p25, p75 = _f("temperatureP25"), _f("temperatureP75")
    if p25 is not None and p75 is not None and p75 >= p25:
        # IQR ~= 1.34898*sigma for normal distributions.
        return max((p75 - p25) / 1.34898, 0.05)

    tmin, tmax = _f("temperatureMin"), _f("temperatureMax")
    if tmin is not None and tmax is not None and tmax >= tmin:
        # Roughly treat [min,max] as ~95% interval.
        return max((tmax - tmin) / 3.92, 0.05)

    return max(0.05, float(config.TOMORROW_DEFAULT_SIGMA_C))


def _build_daily_max_distribution(points: list[dict], samples: int, seed: str) -> dict[int, float]:
    if not points:
        return {}
    rng = random.Random(seed)
    counts: dict[int, int] = {}

    for _ in range(max(1, samples)):
        max_temp = -10_000.0
        for p in points:
            mean = float(p["temp_market"])
            sigma = max(0.05, float(p.get("sigma_market", 0.0)))
            draw = rng.gauss(mean, sigma)
            if draw > max_temp:
                max_temp = draw
        rd = round(max_temp)
        counts[rd] = counts.get(rd, 0) + 1

    total = sum(counts.values())
    if total <= 0:
        return {}
    return {deg: cnt / total for deg, cnt in counts.items()}


async def _fetch_timeline(city: dict, date_str: str, fields: list[str]) -> tuple[list[dict], str]:
    tz_name = city["tz"]
    tz = ZoneInfo(tz_name)
    local_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
    local_end = local_start + timedelta(days=1)

    field_str = ",".join(dict.fromkeys(fields))
    params = {
        "location": f"{city['lat']},{city['lon']}",
        "fields": field_str,
        "timesteps": config.TOMORROW_TIMESTEP,
        "startTime": local_start.isoformat(),
        "endTime": local_end.isoformat(),
        "timezone": tz_name,
        "units": "metric",
        "apikey": (config.TOMORROW_API_KEY or "").strip(),
    }
    url = f"{config.TOMORROW_TIMELINES_API}?{urlencode(params)}"

    data = await get_json(
        url,
        timeout=20,
        attempts=config.API_RETRY_ATTEMPTS,
        base_backoff_seconds=config.API_RETRY_BACKOFF_SECONDS,
        no_retry_status_codes=[400, 401, 403, 404],
    )

    timelines = data.get("data", {}).get("timelines", [])
    if not timelines:
        raise ValueError("Tomorrow.io returned no timelines")

    intervals = timelines[0].get("intervals", []) or []
    if not intervals:
        raise ValueError("Tomorrow.io returned no intervals")

    return intervals, field_str


async def fetch_tomorrow_daily_max(city: dict, date_str: str) -> dict:
    """
    Fetch Tomorrow.io hourly forecast and return predicted max temp for date_str.
    """
    api_key = (config.TOMORROW_API_KEY or "").strip()
    if not api_key:
        raise ValueError("TOMORROW_API_KEY not configured")

    now = datetime.now(timezone.utc)
    ck = _cache_key(city, date_str)
    cached = _TOMORROW_CACHE.get(ck)
    if cached:
        age = (now - cached["fetched_at"]).total_seconds()
        if age <= config.TOMORROW_CACHE_TTL_SECONDS:
            payload = deepcopy(cached["payload"])
            payload["cache"] = "hit"
            payload["cache_age_seconds"] = int(age)
            return payload

    tz_name = city["tz"]
    tz = ZoneInfo(tz_name)
    unit = city["unit"]

    # Try probabilistic fields first, then fall back to plain temperature.
    requested_fields = list(config.TOMORROW_PROB_FIELDS or ["temperature"])
    if "temperature" not in requested_fields:
        requested_fields.insert(0, "temperature")

    try:
        probability_source = "probabilistic_fields"
        intervals, used_fields = await _fetch_timeline(city, date_str, requested_fields)
    except Exception as e:
        fallback = False
        if isinstance(e, httpx.HTTPStatusError):
            fallback = e.response.status_code in (400, 401, 403, 404)
        if fallback:
            log.info(
                "%s %s: probabilistic fields unavailable, fallback to temperature only",
                city["name"],
                date_str,
            )
            probability_source = "temperature_only"
            intervals, used_fields = await _fetch_timeline(city, date_str, ["temperature"])
        else:
            if cached:
                age = (now - cached["fetched_at"]).total_seconds()
                stale = deepcopy(cached["payload"])
                stale["cache"] = "stale_fallback"
                stale["cache_age_seconds"] = int(age)
                stale["note"] = "serving stale cached forecast after fetch error"
                log.warning(
                    "%s %s: Tomorrow fetch failed (%s); using stale cache age=%ss",
                    city["name"],
                    date_str,
                    e,
                    int(age),
                )
                return stale
            raise

    points = []
    has_prob_fields = False
    for item in intervals:
        start = item.get("startTime")
        values = item.get("values", {}) or {}
        temp_c = _extract_temp_c(values)
        if start is None or temp_c is None:
            continue

        dt_utc = _parse_iso(start).astimezone(timezone.utc)
        dt_local = dt_utc.astimezone(tz)
        if dt_local.date().isoformat() != date_str:
            continue

        sigma_c = _extract_sigma_c(values)
        sigma_market = sigma_c * (9.0 / 5.0 if unit == "F" else 1.0)
        if any(k in values for k in ("temperatureP10", "temperatureP25", "temperatureP50", "temperatureP75", "temperatureP90")):
            has_prob_fields = True

        points.append(
            {
                "time_utc": dt_utc.isoformat(),
                "time_local": dt_local.isoformat(),
                "temp_c": float(temp_c),
                "temp_market": _to_market_units(float(temp_c), unit),
                "sigma_c": float(sigma_c),
                "sigma_market": float(sigma_market),
            }
        )

    if not points:
        raise ValueError(f"No Tomorrow.io forecast points for {city['name']} {date_str}")

    max_point = max(points, key=lambda p: p["temp_market"])
    max_market = float(max_point["temp_market"])
    max_c = float(max_point["temp_c"])

    seed = f"{city['name']}:{date_str}:{config.TOMORROW_TIMESTEP}"
    max_dist_market = _build_daily_max_distribution(
        points,
        samples=config.TOMORROW_MONTE_CARLO_SAMPLES,
        seed=seed,
    )

    log.info(
        "%s: Tomorrow.io %s max %.1f%s (%d points, %s)",
        city["name"],
        date_str,
        max_market,
        "°F" if unit == "F" else "°C",
        len(points),
        "prob" if has_prob_fields else "point+calibrated",
    )

    payload = {
        "city": city["name"],
        "date": date_str,
        "source": "tomorrow.io",
        "probability_source": probability_source,
        "probabilistic": bool(has_prob_fields),
        "used_fields": used_fields,
        "timestep": config.TOMORROW_TIMESTEP,
        "as_of_utc": now.isoformat(),
        "points": points,
        "point_count": len(points),
        "max_temp_c": max_c,
        "max_temp_market": max_market,
        "max_time_utc": max_point["time_utc"],
        "max_time_local": max_point["time_local"],
        "max_dist_market": max_dist_market,
        "timezone": tz_name,
        "cache": "miss",
        "cache_age_seconds": 0,
    }
    _TOMORROW_CACHE[ck] = {"fetched_at": now, "payload": deepcopy(payload)}
    return payload
