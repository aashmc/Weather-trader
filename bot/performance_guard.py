"""
Live performance guardrails.

When recent realized performance degrades, this module tightens
entry requirements and position sizing automatically.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import config
from risk import get_state, position_key

GUARD_STATE_FILE = Path("performance_guard_state.json")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _load_state() -> dict:
    if not GUARD_STATE_FILE.exists():
        return {}
    try:
        return json.loads(GUARD_STATE_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def _save_state(payload: dict):
    try:
        GUARD_STATE_FILE.write_text(json.dumps(payload, indent=2))
    except IOError:
        return


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _is_filled_status(status: str) -> bool:
    s = (status or "").strip().lower()
    if not s:
        return False
    if s in {"pending", "cancelled", "canceled", "rejected", "expired"}:
        return False
    return True


def evaluate_live_guardrails() -> dict:
    """
    Returns guardrail controls and diagnostics.
    """
    meta = {
        "enabled": bool(config.LIVE_GUARDRAIL_ENABLED),
        "level": 0,
        "size_mult": 1.0,
        "edge_bump": 0.0,
        "reason": "ok",
        "resolved_count": 0,
        "fill_rate": 0.0,
        "avg_roi": 0.0,
        "avg_expected_edge": 0.0,
        "edge_capture": 1.0,
    }
    if not config.LIVE_GUARDRAIL_ENABLED:
        return meta

    state = get_state()
    positions = state.get("positions", {})

    # Recent resolved, filled positions for realized ROI checks.
    resolved_rows = []
    for pos in positions.values():
        if not pos.get("resolved"):
            continue
        wager = float(pos.get("wager", 0.0) or 0.0)
        pnl = pos.get("pnl")
        if wager <= 0 or pnl is None:
            continue
        if not _is_filled_status(str(pos.get("fill_status", ""))):
            continue
        resolved_rows.append(
            {
                "ts": _parse_ts(pos.get("timestamp", "")),
                "wager": wager,
                "pnl": float(pnl),
                "edge": float(pos.get("edge", 0.0) or 0.0),
            }
        )
    resolved_rows.sort(key=lambda r: r["ts"], reverse=True)
    resolved_rows = resolved_rows[: int(config.LIVE_GUARDRAIL_WINDOW_TRADES)]

    # Recent trade fill-rate checks from trade history.
    trade_rows = []
    for tr in state.get("trade_history", [])[-int(config.LIVE_GUARDRAIL_WINDOW_TRADES) :]:
        key = position_key(tr.get("city", ""), tr.get("market_date", ""), tr.get("bracket", ""))
        pos = positions.get(key, {})
        status = str(pos.get("fill_status", tr.get("fill_status", "")))
        trade_rows.append(status)

    known_status = [s for s in trade_rows if (s or "").strip()]
    filled = [s for s in known_status if _is_filled_status(s)]
    fill_rate = len(filled) / max(1, len(known_status)) if known_status else 1.0

    meta["resolved_count"] = len(resolved_rows)
    meta["fill_rate"] = fill_rate

    if len(resolved_rows) < int(config.LIVE_GUARDRAIL_MIN_RESOLVED):
        meta["reason"] = f"insufficient resolved trades ({len(resolved_rows)})"
        return meta

    avg_roi = sum(r["pnl"] / r["wager"] for r in resolved_rows) / len(resolved_rows)
    avg_expected_edge = sum(r["edge"] for r in resolved_rows) / len(resolved_rows)
    if avg_expected_edge > 0:
        edge_capture = avg_roi / avg_expected_edge
    else:
        edge_capture = 1.0

    meta["avg_roi"] = avg_roi
    meta["avg_expected_edge"] = avg_expected_edge
    meta["edge_capture"] = edge_capture

    severe = (
        avg_roi <= float(config.LIVE_GUARDRAIL_SEVERE_ROI)
        or fill_rate <= float(config.LIVE_GUARDRAIL_SEVERE_FILL_RATE)
        or edge_capture <= float(config.LIVE_GUARDRAIL_SEVERE_EDGE_CAPTURE)
    )
    warn = (
        avg_roi <= float(config.LIVE_GUARDRAIL_WARN_ROI)
        or fill_rate <= float(config.LIVE_GUARDRAIL_WARN_FILL_RATE)
        or edge_capture <= float(config.LIVE_GUARDRAIL_WARN_EDGE_CAPTURE)
    )

    if severe:
        meta["level"] = 2
        meta["size_mult"] = float(config.LIVE_GUARDRAIL_SEVERE_SIZE_MULT)
        meta["edge_bump"] = float(config.LIVE_GUARDRAIL_SEVERE_EDGE_BUMP)
        meta["reason"] = "severe degradation"
    elif warn:
        meta["level"] = 1
        meta["size_mult"] = float(config.LIVE_GUARDRAIL_WARN_SIZE_MULT)
        meta["edge_bump"] = float(config.LIVE_GUARDRAIL_WARN_EDGE_BUMP)
        meta["reason"] = "warning degradation"

    meta["size_mult"] = _clip(meta["size_mult"], 0.0, 1.0)
    meta["edge_bump"] = max(0.0, float(meta["edge_bump"]))
    return meta


def should_alert_guardrails(meta: dict) -> bool:
    """
    Alert on level increases or after cooldown while guardrails remain active.
    """
    if not meta.get("enabled"):
        return False
    if int(meta.get("level", 0)) <= 0:
        return False

    now = datetime.now(timezone.utc)
    state = _load_state()
    last_level = int(state.get("last_level", 0) or 0)
    last_ts = _parse_ts(state.get("last_alert_ts", ""))
    cooldown = timedelta(seconds=max(60, int(config.LIVE_GUARDRAIL_ALERT_COOLDOWN_SECONDS)))

    should = meta["level"] > last_level or (now - last_ts) >= cooldown
    if should:
        state["last_level"] = int(meta.get("level", 0))
        state["last_alert_ts"] = now.isoformat()
        _save_state(state)
    return should


def format_guardrail_alert(meta: dict) -> str:
    return (
        "⚠️ <b>LIVE GUARDRAIL ACTIVE</b>\n"
        f"  Level: {meta.get('level')} ({meta.get('reason')})\n"
        f"  Size multiplier: {meta.get('size_mult', 1.0):.2f}x\n"
        f"  Extra edge requirement: +{meta.get('edge_bump', 0.0)*100:.1f}pt\n"
        f"  Recent ROI: {meta.get('avg_roi', 0.0)*100:.1f}%\n"
        f"  Fill rate: {meta.get('fill_rate', 0.0)*100:.0f}%\n"
        f"  Edge capture: {meta.get('edge_capture', 1.0):.2f}x"
    )

