"""
Weekly auto-tuning for city thresholds using resolved quality metrics.

This adjusts runtime overrides conservatively once per ISO week.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import config

QUALITY_STATE_FILE = Path("model_quality_state.json")
AUTO_TUNE_STATE_FILE = Path("auto_tune_state.json")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return default


def _save_json(path: Path, payload: dict):
    try:
        path.write_text(json.dumps(payload, indent=2))
    except IOError:
        return


def _iso_week_id(now_utc: datetime) -> str:
    iso_year, iso_week, _ = now_utc.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _avg(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
    return sum(vals) / len(vals)


def run_weekly_parameter_tune(now_utc: datetime | None = None) -> dict:
    """
    Returns:
      {
        "enabled": bool,
        "applied": bool,
        "week": "YYYY-Www",
        "changes": {city_key: {...}}
      }
    """
    if not config.AUTO_TUNE_ENABLED:
        return {"enabled": False, "applied": False, "changes": {}}

    now = now_utc or datetime.now(timezone.utc)
    week_id = _iso_week_id(now)
    state = _load_json(AUTO_TUNE_STATE_FILE, {})
    last_week = state.get("last_week")

    # Run once per configured weekday and once per week.
    if now.weekday() != int(config.AUTO_TUNE_WEEKDAY_UTC):
        return {"enabled": True, "applied": False, "week": week_id, "changes": {}}
    if last_week == week_id:
        return {"enabled": True, "applied": False, "week": week_id, "changes": {}}

    quality_state = _load_json(QUALITY_STATE_FILE, {})
    resolved = quality_state.get("resolved_metrics", {})
    payload = config.get_runtime_overrides() or {}
    city_overrides = payload.get("city_overrides", {})
    if not isinstance(city_overrides, dict):
        city_overrides = {}

    changes: dict[str, dict] = {}

    for city_key, city_cfg in config.CITIES.items():
        city_name = city_cfg.get("name", city_key)
        rows = resolved.get(city_name, [])
        if len(rows) < int(config.AUTO_TUNE_MIN_SAMPLES):
            continue

        recent = rows[-int(config.AUTO_TUNE_WINDOW) :]
        avg_brier = _avg(recent, "brier")
        avg_wp = _avg(recent, "winner_prob")

        old_edge = float(city_cfg.get("min_edge", config.AUTO_TUNE_MIN_EDGE))
        old_conc = float(city_cfg.get("min_concentration", config.CONCENTRATION_MIN))

        edge_delta = 0.0
        conc_delta = 0.0
        if avg_brier <= config.AUTO_TUNE_GOOD_BRIER and avg_wp >= config.AUTO_TUNE_GOOD_WINPROB:
            edge_delta = -float(config.AUTO_TUNE_EDGE_STEP)
            conc_delta = -float(config.AUTO_TUNE_CONC_STEP)
        elif avg_brier >= config.AUTO_TUNE_BAD_BRIER or avg_wp <= config.AUTO_TUNE_BAD_WINPROB:
            edge_delta = float(config.AUTO_TUNE_EDGE_STEP)
            conc_delta = float(config.AUTO_TUNE_CONC_STEP)

        new_edge = _clip(
            old_edge + edge_delta,
            float(config.AUTO_TUNE_MIN_EDGE),
            float(config.AUTO_TUNE_MAX_EDGE),
        )
        new_conc = _clip(
            old_conc + conc_delta,
            float(config.AUTO_TUNE_MIN_CONC),
            float(config.AUTO_TUNE_MAX_CONC),
        )

        if abs(new_edge - old_edge) < 1e-9 and abs(new_conc - old_conc) < 1e-9:
            continue

        ov = city_overrides.get(city_key, {})
        if not isinstance(ov, dict):
            ov = {}
        ov["min_edge"] = round(new_edge, 3)
        ov["min_concentration"] = round(new_conc, 3)
        city_overrides[city_key] = ov

        changes[city_key] = {
            "samples": len(recent),
            "avg_brier": round(avg_brier, 3),
            "avg_winner_prob": round(avg_wp, 3),
            "old_edge": old_edge,
            "new_edge": new_edge,
            "old_concentration": old_conc,
            "new_concentration": new_conc,
        }

    if changes:
        payload["city_overrides"] = city_overrides
        config.write_runtime_overrides(payload)

    state["last_week"] = week_id
    state["last_run_utc"] = now.isoformat()
    state["last_changes"] = changes
    _save_json(AUTO_TUNE_STATE_FILE, state)

    return {
        "enabled": True,
        "applied": True,
        "week": week_id,
        "changes": changes,
    }

