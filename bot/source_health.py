"""
Source contract validation and health aggregation.

These checks are intentionally lightweight and non-invasive:
- They validate payload shape and basic data sanity.
- They can mark checks as blocking for strict cities (NYC-first rollout).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import config


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status(reasons_error: list[str], reasons_warn: list[str]) -> str:
    if reasons_error:
        return "error"
    if reasons_warn:
        return "warn"
    return "ok"


def check_market_contract(market: dict | None) -> dict:
    market = market or {}
    errors: list[str] = []
    warns: list[str] = []

    required = ("brackets", "prices", "token_ids", "slug", "resolved")
    missing = [k for k in required if k not in market]
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")

    brackets = market.get("brackets") or []
    prices = market.get("prices") or {}
    token_ids = market.get("token_ids") or {}
    winner = market.get("winner")

    if not isinstance(brackets, list) or len(brackets) == 0:
        errors.append("no brackets")
    if not isinstance(token_ids, dict) or len(token_ids) == 0:
        errors.append("no token_ids")
    if not isinstance(prices, dict) or len(prices) == 0:
        warns.append("no market prices")

    if isinstance(brackets, list) and isinstance(token_ids, dict) and len(brackets) > 0:
        coverage = len(token_ids) / max(1, len(brackets))
        if coverage < 0.60:
            errors.append(
                f"token coverage too low ({len(token_ids)}/{len(brackets)})"
            )
        elif coverage < 1.0:
            warns.append(f"token coverage partial ({len(token_ids)}/{len(brackets)})")

    if bool(market.get("resolved")) and not winner:
        warns.append("resolved market without winner label")

    return {
        "source": "market",
        "status": _status(errors, warns),
        "blocking": bool(errors),
        "errors": errors,
        "warnings": warns,
        "details": {
            "brackets": len(brackets) if isinstance(brackets, list) else 0,
            "prices": len(prices) if isinstance(prices, dict) else 0,
            "token_ids": len(token_ids) if isinstance(token_ids, dict) else 0,
            "resolved": bool(market.get("resolved", False)),
            "winner": winner or "",
            "slug": str(market.get("slug", "") or ""),
        },
    }


def check_books_contract(books: dict | None, token_ids: dict | None) -> dict:
    books = books or {}
    token_ids = token_ids or {}
    errors: list[str] = []
    warns: list[str] = []

    expected = len(token_ids) if isinstance(token_ids, dict) else 0
    got = len(books) if isinstance(books, dict) else 0

    if expected <= 0:
        errors.append("no expected token_ids for book fetch")
    if got <= 0:
        errors.append("no order books fetched")

    if expected > 0 and got > 0:
        coverage = got / expected
        if coverage < 0.70:
            errors.append(f"book coverage too low ({got}/{expected})")
        elif coverage < 1.0:
            warns.append(f"book coverage partial ({got}/{expected})")

    bad_shape = 0
    for label, book in (books or {}).items():
        if not isinstance(book, dict):
            bad_shape += 1
            continue
        bb = _as_float(book.get("bb"))
        ba = _as_float(book.get("ba"))
        if bb is None or ba is None:
            bad_shape += 1
            continue
        if ba <= 0 or bb < 0:
            bad_shape += 1
            continue
        if ba < bb:
            warns.append(f"crossed book for {label}")
    if bad_shape > 0:
        errors.append(f"invalid book shape in {bad_shape} entries")

    return {
        "source": "books",
        "status": _status(errors, warns),
        "blocking": bool(errors),
        "errors": errors,
        "warnings": warns,
        "details": {
            "expected_books": expected,
            "fetched_books": got,
        },
    }


def check_tomorrow_contract(forecast: dict | None) -> dict:
    forecast = forecast or {}
    errors: list[str] = []
    warns: list[str] = []

    required = (
        "source",
        "as_of_utc",
        "max_temp_market",
        "point_count",
        "points",
        "probability_source",
    )
    missing = [k for k in required if k not in forecast]
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")

    max_temp_market = _as_float(forecast.get("max_temp_market"))
    if max_temp_market is None:
        errors.append("invalid max_temp_market")

    points = forecast.get("points") or []
    point_count = int(forecast.get("point_count") or 0)
    if not isinstance(points, list) or len(points) == 0 or point_count <= 0:
        errors.append("no forecast points")

    as_of_utc = _parse_ts(forecast.get("as_of_utc"))
    age_minutes = None
    if as_of_utc is None:
        warns.append("invalid as_of_utc timestamp")
    else:
        age_minutes = (_utc_now() - as_of_utc).total_seconds() / 60.0
        max_age = float(config.FRESHNESS_MAX_FORECAST_AGE_MINUTES)
        if age_minutes > max_age:
            errors.append(f"forecast stale ({age_minutes:.0f}m > {max_age:.0f}m)")

    probability_source = str(forecast.get("probability_source", "") or "")
    probabilistic = bool(forecast.get("probabilistic", False))
    if probability_source == "temperature_only" or not probabilistic:
        warns.append("temperature-only mode (no provider bracket probabilities)")

    max_dist_market = forecast.get("max_dist_market") or {}
    if not isinstance(max_dist_market, dict) or len(max_dist_market) == 0:
        warns.append("empty max_dist_market")

    return {
        "source": "tomorrow",
        "status": _status(errors, warns),
        "blocking": bool(errors),
        "errors": errors,
        "warnings": warns,
        "details": {
            "probability_source": probability_source,
            "probabilistic": probabilistic,
            "point_count": point_count,
            "age_minutes": None if age_minutes is None else round(age_minutes, 2),
            "cache": str(forecast.get("cache", "") or ""),
        },
    }


def check_ensemble_contract(ensemble: dict | None) -> dict:
    ensemble = ensemble or {}
    errors: list[str] = []
    warns: list[str] = []

    required = ("mean", "count", "raw_dist", "max_temps")
    missing = [k for k in required if k not in ensemble]
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")

    count = int(ensemble.get("count") or 0)
    max_temps = ensemble.get("max_temps") or []
    raw_dist = ensemble.get("raw_dist") or {}
    if count <= 0:
        errors.append("ensemble count <= 0")
    if not isinstance(max_temps, list) or len(max_temps) == 0:
        errors.append("empty max_temps")
    if not isinstance(raw_dist, dict) or len(raw_dist) == 0:
        errors.append("empty raw_dist")

    source = str(ensemble.get("source", "") or "")
    if source == "cache":
        cache_age = _as_float(ensemble.get("cache_age_minutes"))
        if cache_age is not None and cache_age > float(config.FRESHNESS_MAX_FORECAST_AGE_MINUTES):
            errors.append(
                f"ensemble cache stale ({cache_age:.0f}m > {config.FRESHNESS_MAX_FORECAST_AGE_MINUTES:.0f}m)"
            )
        else:
            warns.append("ensemble served from cache")

    if not bool(ensemble.get("freshness_ok", True)):
        errors.append("ensemble freshness_not_ok")

    return {
        "source": "ensemble",
        "status": _status(errors, warns),
        "blocking": bool(errors),
        "errors": errors,
        "warnings": warns,
        "details": {
            "count": count,
            "source": source,
            "cache_age_minutes": ensemble.get("cache_age_minutes"),
        },
    }


def check_metar_contract(metar: dict | None, *, city_tz: str, date_str: str) -> dict:
    metar = metar or {}
    errors: list[str] = []
    warns: list[str] = []

    required = ("source", "observation_count", "freshness_ok")
    missing = [k for k in required if k not in metar]
    if missing:
        warns.append(f"missing fields: {', '.join(missing)}")

    source = str(metar.get("source", "") or "")
    obs_count = int(metar.get("observation_count") or 0)
    age_minutes = _as_float(metar.get("age_minutes"))
    freshness_ok = bool(metar.get("freshness_ok", False))
    day_high_market = metar.get("day_high_market")

    # METAR strictness should only apply to same/past local date, not future dates.
    try:
        market_day = datetime.strptime(date_str, "%Y-%m-%d").date()
        local_today = _utc_now().astimezone(ZoneInfo(city_tz)).date()
        is_future_market = market_day > local_today
    except Exception:
        is_future_market = False

    if not is_future_market:
        if source in ("none", ""):
            errors.append("no METAR source")
        if obs_count <= 0 and day_high_market is None:
            warns.append("no METAR observations yet")
        if age_minutes is not None and age_minutes > float(config.FRESHNESS_MAX_METAR_AGE_MINUTES):
            errors.append(
                f"METAR stale ({age_minutes:.0f}m > {config.FRESHNESS_MAX_METAR_AGE_MINUTES:.0f}m)"
            )
        if not freshness_ok and source not in ("open_meteo_fallback", "metar_cache"):
            warns.append("METAR freshness_not_ok")
    else:
        if source in ("none", ""):
            warns.append("future-date METAR unavailable")

    return {
        "source": "metar",
        "status": _status(errors, warns),
        "blocking": bool(errors),
        "errors": errors,
        "warnings": warns,
        "details": {
            "source": source,
            "observation_count": obs_count,
            "age_minutes": age_minutes,
            "freshness_ok": freshness_ok,
            "day_high_market": day_high_market,
            "future_market": is_future_market,
        },
    }


def aggregate_source_health(*, city_key: str, checks: dict[str, dict]) -> dict:
    checks = checks or {}
    strict_keys = set(config.SOURCE_CONTRACT_STRICT_CITY_KEYS or [])
    strict_city = city_key in strict_keys

    reasons: list[str] = []
    blocking_reasons: list[str] = []
    has_error = False
    has_warn = False

    for name, chk in checks.items():
        errs = list(chk.get("errors") or [])
        warns = list(chk.get("warnings") or [])
        if errs:
            has_error = True
            for e in errs:
                msg = f"{name}: {e}"
                reasons.append(msg)
                if bool(chk.get("blocking", False)):
                    blocking_reasons.append(msg)
        if warns:
            has_warn = True
            for w in warns:
                reasons.append(f"{name}: {w}")

    status = "error" if has_error else ("warn" if has_warn else "ok")
    block = bool(strict_city and blocking_reasons)
    return {
        "enabled": bool(config.SOURCE_CONTRACT_CHECKS_ENABLED),
        "city_key": city_key,
        "strict_city": strict_city,
        "status": status,
        "block": block,
        "reasons": reasons,
        "blocking_reasons": blocking_reasons,
        "checks": checks,
    }
