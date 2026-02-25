"""
Weather Trader Bot — Risk Management
Tracks positions, daily P&L, enforces kill switch.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

log = logging.getLogger("risk")

STATE_FILE = Path("state.json")


def _load_state() -> dict:
    """Load persisted state from disk."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            log.warning("State file corrupted, starting fresh")
    return {
        "positions": {},       # {city:date:bracket: position_data}
        "daily_pnl": {},       # {date_str: float}
        "kill_switch": {},     # {date_str: bool}
        "trade_history": [],   # all trades ever
        "resolutions": {},     # {city:date: {winner, pnl}}
    }


def _save_state(state: dict):
    """Persist state to disk."""
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except IOError as e:
        log.error(f"Failed to save state: {e}")


def get_state() -> dict:
    return _load_state()


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def position_key(city_name: str, date_str: str, bracket: str) -> str:
    return f"{city_name}:{date_str}:{bracket}"


def is_kill_switch_active() -> bool:
    """Check if today's kill switch is triggered."""
    state = _load_state()
    return state.get("kill_switch", {}).get(today_str(), False)


def get_daily_pnl() -> float:
    """Get today's realized P&L."""
    state = _load_state()
    return state.get("daily_pnl", {}).get(today_str(), 0.0)


def get_daily_exposure() -> float:
    """Get total $ wagered across all active (unresolved) positions."""
    state = _load_state()
    total = 0.0
    for pos in state.get("positions", {}).values():
        if not pos.get("resolved"):
            total += pos.get("wager", 0.0)
    return total


def get_existing_positions(city_name: str, date_str: str) -> set[str]:
    """Get bracket labels where we already have a position for this city/date."""
    state = _load_state()
    positions = set()
    prefix = f"{city_name}:{date_str}:"
    for key in state.get("positions", {}):
        if key.startswith(prefix):
            positions.add(key.split(":", 2)[2])
    return positions


def can_trade() -> tuple[bool, str]:
    """Check if trading is allowed right now."""
    if is_kill_switch_active():
        return False, f"Kill switch active (daily loss ≥ ${config.DAILY_LOSS_LIMIT})"

    exposure = get_daily_exposure()
    if exposure >= config.MAX_TOTAL_EXPOSURE:
        return False, f"Max exposure reached (${exposure:.2f} ≥ ${config.MAX_TOTAL_EXPOSURE})"

    return True, "OK"


def record_trade(
    city_name: str,
    date_str: str,
    bracket: str,
    signal: str,
    corrected_prob: float,
    ask: float,
    kelly_bet: float,
    contracts: int,
    edge: float,
    model_votes: int,
    order_id: str | None = None,
):
    """Record a new trade in state."""
    state = _load_state()

    key = position_key(city_name, date_str, bracket)
    trade_data = {
        "city": city_name,
        "market_date": date_str,
        "bracket": bracket,
        "signal": signal,
        "corrected_prob": corrected_prob,
        "ask": ask,
        "wager": kelly_bet,
        "contracts": contracts,
        "edge": edge,
        "model_votes": model_votes,
        "order_id": order_id,
        "fill_status": "pending",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolved": False,
        "pnl": None,
    }

    state.setdefault("positions", {})[key] = trade_data
    state.setdefault("trade_history", []).append(trade_data)

    _save_state(state)
    log.info(f"Trade recorded: {key} — ${kelly_bet:.2f} @ {ask:.3f}")


def update_fill_status(city_name: str, date_str: str, bracket: str, status: str):
    """Update an order's fill status."""
    state = _load_state()
    key = position_key(city_name, date_str, bracket)
    if key in state.get("positions", {}):
        state["positions"][key]["fill_status"] = status
        _save_state(state)
        log.info(f"Fill status updated: {key} → {status}")


def record_resolution(city_name: str, date_str: str, winner: str):
    """
    Record market resolution and compute P&L for all positions in this market.
    Returns total P&L for this resolution.
    """
    state = _load_state()
    prefix = f"{city_name}:{date_str}:"
    total_pnl = 0.0
    positions_resolved = 0

    for key, pos in state.get("positions", {}).items():
        if not key.startswith(prefix):
            continue
        if pos.get("resolved"):
            continue

        bracket = pos["bracket"]
        wager = pos.get("wager", 0.0)
        contracts = pos.get("contracts", 0)
        fill = pos.get("fill_status", "pending")

        # Only count confidently filled trades.
        # "pending"/"partial"/"cancelled" are treated as no-fill until
        # we have matched-size accounting in state.
        if fill in ("pending", "partial", "cancelled"):
            if fill == "pending":
                log.warning(f"Resolution skipped PnL for pending order: {key}")
            elif fill == "partial":
                log.warning(f"Resolution skipped PnL for partial order (no filled-size tracking): {key}")
            pos["resolved"] = True
            pos["pnl"] = 0.0
            continue

        if bracket == winner:
            # WIN: contracts pay $1 each
            pnl = contracts * 1.0 - wager
        else:
            # LOSS: shares worthless
            pnl = -wager

        pos["resolved"] = True
        pos["pnl"] = pnl
        total_pnl += pnl
        positions_resolved += 1

        log.info(
            f"Resolution: {key} → {'WIN' if bracket == winner else 'LOSS'} "
            f"P&L: {'+'if pnl >= 0 else ''}{pnl:.2f}"
        )

    # Update daily P&L
    today = today_str()
    state.setdefault("daily_pnl", {})[today] = (
        state.get("daily_pnl", {}).get(today, 0.0) + total_pnl
    )

    # Store resolution
    res_key = f"{city_name}:{date_str}"
    state.setdefault("resolutions", {})[res_key] = {
        "winner": winner,
        "pnl": total_pnl,
        "positions_resolved": positions_resolved,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Check kill switch
    daily = state["daily_pnl"].get(today, 0.0)
    if daily <= -config.DAILY_LOSS_LIMIT:
        state.setdefault("kill_switch", {})[today] = True
        log.warning(
            f"KILL SWITCH TRIGGERED: daily P&L ${daily:.2f} "
            f"exceeds -${config.DAILY_LOSS_LIMIT} limit"
        )

    _save_state(state)
    return total_pnl


def get_pending_resolutions() -> list[dict]:
    """Get all positions awaiting resolution (past dates, not yet resolved)."""
    state = _load_state()
    pending = []
    now = datetime.now(timezone.utc)

    seen_markets = set()
    for key, pos in state.get("positions", {}).items():
        if pos.get("resolved"):
            continue
        market_key = f"{pos['city']}:{pos['market_date']}"
        if market_key in seen_markets:
            continue
        # Only check markets where the date has passed
        try:
            market_date = datetime.strptime(pos["market_date"], "%Y-%m-%d")
            # Give 6 hours after market date for resolution
            if now.date() > market_date.date():
                pending.append({
                    "city": pos["city"],
                    "date": pos["market_date"],
                })
                seen_markets.add(market_key)
        except ValueError:
            continue

    return pending


def get_portfolio_summary() -> dict:
    """Get overall portfolio statistics."""
    state = _load_state()
    positions = state.get("positions", {})

    active = [p for p in positions.values() if not p.get("resolved")]
    resolved = [p for p in positions.values() if p.get("resolved")]

    total_wagered = sum(p.get("wager", 0) for p in resolved if p.get("pnl") is not None)
    total_pnl = sum(p.get("pnl", 0) for p in resolved if p.get("pnl") is not None)
    wins = sum(1 for p in resolved if (p.get("pnl") or 0) > 0)
    losses = sum(1 for p in resolved if (p.get("pnl") or 0) < 0)

    return {
        "active_positions": len(active),
        "active_exposure": sum(p.get("wager", 0) for p in active),
        "total_trades": len(resolved),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(1, wins + losses),
        "total_wagered": total_wagered,
        "total_pnl": total_pnl,
        "roi": (total_pnl / total_wagered * 100) if total_wagered > 0 else 0,
    }
