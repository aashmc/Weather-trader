"""
NYC ingestion pipeline (capture-only, no trading dependency).

Captures hourly forecast snapshots from configured source models plus KLGA METAR
and persists them for the NYC quant model build.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from http_retry import get_json
from metar import fetch_metar

log = logging.getLogger("nyc_ingestion")

STATE_FILE = Path("nyc_ingestion_state.json")
LOG_FILE = Path("nyc_ingestion_log.jsonl")


def _to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return {"latest": {}, "latest_by_date": {}}


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except IOError as exc:
        log.warning("Failed to save NYC ingestion state: %s", exc)


def _append_log(row: dict):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except IOError as exc:
        log.warning("Failed to append NYC ingestion log: %s", exc)


def _parse_dt_local(value: str, tz: ZoneInfo) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _extract_run_time_utc(payload: dict) -> str | None:
    candidates = (
        "model_run",
        "modelrun_utc",
        "run_time_utc",
        "runTimeUtc",
        "run_time",
        "initial_time",
        "reference_time",
    )
    for key in candidates:
        val = payload.get(key)
        if val:
            return str(val)
    return None


def _endpoint_for_source(source_name: str) -> str:
    if source_name in ("gefs", "ecmwf_ens"):
        return config.NYC_INGESTION_ENSEMBLE_API
    return config.NYC_INGESTION_FORECAST_API


def _extract_hourly_temperature_series(hourly: dict) -> tuple[list, list]:
    """
    Return (times, temps_c) where temps_c is either direct temperature_2m values
    or an average across ensemble member series when direct values are missing.
    """
    times = list((hourly or {}).get("time", []) or [])
    if not times:
        return [], []

    direct = list((hourly or {}).get("temperature_2m", []) or [])
    if len(direct) == len(times):
        valid_direct = sum(1 for v in direct if _safe_float(v) is not None)
        if valid_direct > 0:
            return times, direct

    member_keys = sorted(
        k for k in (hourly or {}).keys() if str(k).startswith("temperature_2m_member")
    )
    if not member_keys:
        return times, []

    member_series = [list((hourly or {}).get(k, []) or []) for k in member_keys]
    member_series = [s for s in member_series if len(s) == len(times)]
    if not member_series:
        return times, []

    temps = []
    for idx in range(len(times)):
        vals = []
        for series in member_series:
            fv = _safe_float(series[idx])
            if fv is not None:
                vals.append(fv)
        if vals:
            temps.append(sum(vals) / len(vals))
        else:
            temps.append(None)
    return times, temps


def _summarize_hourly(
    *,
    payload: dict,
    date_str: str,
    tz_name: str,
) -> dict:
    hourly = payload.get("hourly", {}) or {}
    times, temps = _extract_hourly_temperature_series(hourly)
    if not isinstance(times, list) or not isinstance(temps, list):
        raise ValueError("invalid hourly payload shape")
    if len(times) != len(temps):
        raise ValueError("hourly time/value length mismatch")

    tz = ZoneInfo(tz_name)
    points = []
    for ts, temp in zip(times, temps):
        t_c = _safe_float(temp)
        if t_c is None:
            continue
        dt_local = _parse_dt_local(ts, tz)
        if dt_local is None:
            continue
        if dt_local.date().isoformat() != date_str:
            continue
        points.append(
            {
                "time_local": dt_local.isoformat(),
                "time_utc": dt_local.astimezone(timezone.utc).isoformat(),
                "temp_c": t_c,
                "temp_f": _to_f(t_c),
            }
        )

    if not points:
        raise ValueError(f"no hourly points for {date_str}")

    max_point = max(points, key=lambda p: p["temp_f"])
    return {
        "point_count": len(points),
        "max_temp_c": round(float(max_point["temp_c"]), 2),
        "max_temp_f": round(float(max_point["temp_f"]), 2),
        "max_time_local": max_point["time_local"],
        "max_time_utc": max_point["time_utc"],
        "hourly_points": points,
    }


async def _fetch_source(city: dict, date_str: str, source_name: str, model_id: str) -> dict:
    started_at = datetime.now(timezone.utc)
    if not model_id:
        return {
            "status": "disabled",
            "source": source_name,
            "model": "",
            "error": "empty model id",
            "as_of_utc": started_at.isoformat(),
        }

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": "temperature_2m",
        "models": model_id,
        "forecast_days": int(config.NYC_INGESTION_FORECAST_DAYS),
        "timezone": city["tz"],
    }
    endpoint = _endpoint_for_source(source_name)
    try:
        payload = await get_json(
            endpoint,
            params=params,
            timeout=float(config.NYC_INGESTION_TIMEOUT_SECONDS),
            attempts=config.API_RETRY_ATTEMPTS,
            base_backoff_seconds=config.API_RETRY_BACKOFF_SECONDS,
            no_retry_status_codes={400, 401, 403, 404},
        )
        summary = _summarize_hourly(
            payload=payload,
            date_str=date_str,
            tz_name=city["tz"],
        )
        return {
            "status": "ok",
            "source": source_name,
            "model": model_id,
            "endpoint": endpoint,
            "run_time_utc": _extract_run_time_utc(payload),
            "generationtime_ms": _safe_float(payload.get("generationtime_ms")),
            "as_of_utc": started_at.isoformat(),
            **summary,
        }
    except Exception as exc:
        return {
            "status": "error",
            "source": source_name,
            "model": model_id,
            "endpoint": endpoint,
            "error": str(exc)[:240],
            "as_of_utc": started_at.isoformat(),
        }


async def capture_nyc_ingestion(date_str: str) -> dict:
    """
    Capture one NYC ingestion snapshot for a market date.
    """
    city = config.CITIES["nyc"]
    now = datetime.now(timezone.utc)

    sources = {}
    source_order = ("nbm", "hrrr", "gefs", "ecmwf_ens")
    tasks = []
    for src in source_order:
        model_id = (config.NYC_INGESTION_MODELS.get(src) or "").strip()
        tasks.append(_fetch_source(city, date_str, src, model_id))

    source_rows = await asyncio.gather(*tasks)
    for row in source_rows:
        key = str(row.get("source") or "unknown")
        sources[key] = row

    metar_row = {
        "status": "error",
        "source": "metar",
        "error": "not fetched",
    }
    try:
        metar = await fetch_metar(city, date_str)
        metar_status = "ok" if metar.get("source") not in ("none", "") else "error"
        if metar_status == "ok" and not bool(metar.get("freshness_ok", False)):
            metar_status = "warn"
        metar_row = {
            "status": metar_status,
            "source": str(metar.get("source", "metar")),
            "current_temp_c": metar.get("current_temp_c"),
            "day_high_market": metar.get("day_high_market"),
            "observation_count": int(metar.get("observation_count", 0) or 0),
            "age_minutes": metar.get("age_minutes"),
            "latest_obs_time_utc": metar.get("latest_obs_time_utc"),
        }
    except Exception as exc:
        metar_row = {
            "status": "error",
            "source": "metar",
            "error": str(exc)[:240],
        }

    total_sources = len(sources)
    ok_sources = sum(1 for row in sources.values() if row.get("status") == "ok")
    err_sources = sum(1 for row in sources.values() if row.get("status") == "error")

    record = {
        "timestamp_utc": now.isoformat(),
        "city_key": "nyc",
        "city_name": city["name"],
        "station": city["icao"],
        "date": date_str,
        "location": {"lat": city["lat"], "lon": city["lon"], "tz": city["tz"]},
        "coverage": {
            "ok_sources": ok_sources,
            "error_sources": err_sources,
            "total_sources": total_sources,
        },
        "sources": sources,
        "metar": metar_row,
    }

    state = _load_state()
    by_date = state.setdefault("latest_by_date", {})
    by_date[date_str] = record
    if len(by_date) > 45:
        keys = sorted(by_date.keys())
        for k in keys[:-45]:
            by_date.pop(k, None)
    state["latest"] = record
    _save_state(state)
    _append_log(record)
    return record


def get_latest_nyc_ingestion(date_str: str | None = None) -> dict | None:
    state = _load_state()
    if date_str:
        return (state.get("latest_by_date") or {}).get(date_str)
    return state.get("latest")
