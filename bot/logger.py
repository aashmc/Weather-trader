"""
Weather Trader Bot — Logging
Comprehensive data capture for recalibration + Google Sheets + local backup.
Every piece of data needed to reconstruct decisions and improve the model.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import httpx
from config import GOOGLE_SHEET_WEBHOOK

log = logging.getLogger("logger")

LOCAL_LOG_FILE = Path("trade_log.jsonl")
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)


async def log_cycle(
    city: str,
    date_str: str,
    ensemble_mean: float,
    ensemble_count: int,
    metar_high: float | None,
    metar_current: float | None,
    metar_obs_count: int,
    brackets_data: list[dict],
    signals: list[dict],
    trades_placed: list[dict],
    raw_dist: dict = None,
    corrected_dist: dict = None,
    model_votes: dict = None,
    market_prices: dict = None,
    books_snapshot: dict = None,
    max_temps: list = None,
    gas_cost: float = 0,
    pol_price: float = 0,
    bias_mean: float = 0,
    bias_sd: float = 0,
    current_exposure: float = 0,
    daily_pnl: float = 0,
    # v5 filter metadata
    v5_favorite: str = None,
    v5_concentration: float = 0,
    v5_kelly_multiplier: float = 0,
    v5_maturity: dict = None,
    v5_divergence: dict = None,
    v5_candidates: list = None,
):
    """Log a complete cycle — both summary (Sheets) and full snapshot (local)."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── GOOGLE SHEETS: Summary row ──
    row = {
        "timestamp": timestamp,
        "city": city,
        "market_date": date_str,
        "ensemble_mean": round(ensemble_mean, 2),
        "ensemble_members": ensemble_count,
        "metar_current": metar_current,
        "metar_day_high": metar_high,
        "metar_obs_count": metar_obs_count,
        "num_brackets": len(brackets_data),
        "num_signals": len(signals),
        "num_trades": len(trades_placed),
        "current_exposure": round(current_exposure, 2),
        "daily_pnl": round(daily_pnl, 2),
        "gas_cost_usd": round(gas_cost, 5),
        "pol_price_usd": round(pol_price, 4),
        "v5_favorite": v5_favorite or "",
        "v5_concentration": round(v5_concentration * 100, 1),
        "v5_kelly_mult": round(v5_kelly_multiplier, 2),
        "v5_mature": v5_maturity.get("mature", True) if v5_maturity else True,
        "v5_mature_reason": v5_maturity.get("reason", "") if v5_maturity else "",
        "v5_diverged": v5_divergence.get("diverged", False) if v5_divergence else False,
        "v5_diverge_dist": v5_divergence.get("distance", 0) if v5_divergence else 0,
        "v5_candidates": ",".join(v5_candidates or []),
    }

    # Per-bracket data (all 9)
    for i, b in enumerate(brackets_data[:9]):
        prefix = f"b{i}"
        row[f"{prefix}_label"] = b.get("bracket", "")
        row[f"{prefix}_corr"] = round(b.get("corrected_prob", 0) * 100, 1)
        row[f"{prefix}_mkt"] = round(b.get("market_prob", 0) * 100, 1)
        row[f"{prefix}_raw"] = round(b.get("raw_prob", 0) * 100, 1)
        row[f"{prefix}_edge"] = round(b.get("true_edge", 0) * 100, 1)
        row[f"{prefix}_ask"] = round(b.get("ask", 0), 4)
        row[f"{prefix}_bid"] = round(b.get("bid", 0), 4)
        row[f"{prefix}_askD"] = round(b.get("ask_depth", 0), 0)
        row[f"{prefix}_bidD"] = round(b.get("bid_depth", 0), 0)
        row[f"{prefix}_spread"] = round(b.get("spread", 0), 4)
        row[f"{prefix}_models"] = b.get("model_votes", 0)
        row[f"{prefix}_signal"] = b.get("signal", "")
        row[f"{prefix}_eAsk"] = round(b.get("effective_ask", b.get("ask", 0)), 4)
        row[f"{prefix}_fill"] = round(b.get("fill_fraction", 1.0), 3)
        row[f"{prefix}_slip"] = round(b.get("slippage_penalty", 0), 4)
        row[f"{prefix}_fee"] = round(b.get("fee_per_contract", 0), 4)
        row[f"{prefix}_gas"] = round(b.get("gas_per_contract", 0), 4)
        row[f"{prefix}_kellyFull"] = round(b.get("kelly_full", 0), 4)
        row[f"{prefix}_bet"] = round(b.get("kelly_bet", 0), 2)
        row[f"{prefix}_ep"] = round(b.get("expected_profit", 0), 2)
        row[f"{prefix}_filters"] = "; ".join(b.get("filter_reasons", []))

    # Trade details
    for i, t in enumerate(trades_placed[:3]):
        prefix = f"t{i}"
        row[f"{prefix}_bracket"] = t.get("bracket", "")
        row[f"{prefix}_ask"] = t.get("ask", 0)
        row[f"{prefix}_bet"] = t.get("kelly_bet", 0)
        row[f"{prefix}_edge"] = round(t.get("true_edge", 0) * 100, 1)
        row[f"{prefix}_contracts"] = t.get("contracts", 0)
        row[f"{prefix}_corrProb"] = round(t.get("corrected_prob", 0) * 100, 1)
        row[f"{prefix}_order_id"] = t.get("order_id", "")

    await _log_to_sheets(row)

    # ── LOCAL SNAPSHOT: Full data for recalibration ──
    snapshot = {
        "timestamp": timestamp,
        "city": city,
        "market_date": date_str,
        "config_at_time": {
            "bias_mean": bias_mean,
            "bias_sd": bias_sd,
        },
        "v5_filters": {
            "favorite": v5_favorite,
            "concentration": round(v5_concentration, 4),
            "kelly_multiplier": round(v5_kelly_multiplier, 2),
            "maturity": v5_maturity or {},
            "divergence": v5_divergence or {},
            "candidates": v5_candidates or [],
        },
        "ensemble": {
            "mean": round(ensemble_mean, 3),
            "member_count": ensemble_count,
            "max_temps": [round(t, 2) for t in (max_temps or [])],
            "raw_distribution": {str(k): round(v, 6) for k, v in (raw_dist or {}).items()},
            "corrected_distribution": {str(k): round(v, 6) for k, v in (corrected_dist or {}).items()},
            "model_votes_per_degree": {str(k): v for k, v in (model_votes or {}).items()},
        },
        "metar": {
            "current_temp_c": metar_current,
            "day_high_market_units": metar_high,
            "observation_count": metar_obs_count,
        },
        "market": {
            "prices": market_prices or {},
            "books": {
                label: {
                    "bb": round(b.get("bb", 0), 4),
                    "ba": round(b.get("ba", 0), 4),
                    "spread": round(b.get("spread", 0), 4),
                    "bid_depth": round(b.get("bid_depth", 0), 1),
                    "ask_depth": round(b.get("ask_depth", 0), 1),
                }
                for label, b in (books_snapshot or {}).items()
            },
            "gas_cost_usd": round(gas_cost, 5),
            "pol_price_usd": round(pol_price, 4),
        },
        "portfolio": {
            "current_exposure": round(current_exposure, 2),
            "daily_pnl": round(daily_pnl, 2),
        },
        "analysis": [
            {
                "bracket": b.get("bracket"),
                "raw_prob": round(b.get("raw_prob", 0), 6),
                "corrected_prob": round(b.get("corrected_prob", 0), 6),
                "market_prob": round(b.get("market_prob", 0), 6),
                "bid": round(b.get("bid", 0), 4),
                "ask": round(b.get("ask", 0), 4),
                "spread": round(b.get("spread", 0), 4),
                "bid_depth": round(b.get("bid_depth", 0), 1),
                "ask_depth": round(b.get("ask_depth", 0), 1),
                "raw_edge": round(b.get("raw_edge", 0), 6),
                "true_edge": round(b.get("true_edge", 0), 6),
                "edge_after_costs": round(b.get("edge_after_costs", 0), 6),
                "effective_ask": round(b.get("effective_ask", b.get("ask", 0)), 6),
                "fill_fraction": round(b.get("fill_fraction", 1.0), 4),
                "immediate_fill_fraction": round(b.get("immediate_fill_fraction", 1.0), 4),
                "continuation_fill_prob": round(b.get("continuation_fill_prob", 1.0), 4),
                "fee_per_contract": round(b.get("fee_per_contract", 0), 6),
                "gas_per_contract": round(b.get("gas_per_contract", 0), 6),
                "slippage_penalty": round(b.get("slippage_penalty", 0), 6),
                "model_votes": b.get("model_votes", 0),
                "total_models": b.get("total_models", 0),
                "kelly_full": round(b.get("kelly_full", 0), 6),
                "kelly_bet": round(b.get("kelly_bet", 0), 2),
                "expected_profit": round(b.get("expected_profit", 0), 2),
                "contracts": b.get("contracts", 0),
                "signal": b.get("signal"),
                "filter_reasons": b.get("filter_reasons", []),
            }
            for b in brackets_data
        ],
        "trades": [
            {
                "bracket": t.get("bracket"),
                "ask": t.get("ask"),
                "corrected_prob": round(t.get("corrected_prob", 0), 6),
                "true_edge": round(t.get("true_edge", 0), 6),
                "kelly_bet": round(t.get("kelly_bet", 0), 2),
                "contracts": t.get("contracts", 0),
                "model_votes": t.get("model_votes", 0),
                "order_id": t.get("order_id"),
            }
            for t in trades_placed
        ],
    }

    _log_to_local(snapshot)
    _save_snapshot(city, date_str, snapshot)


async def log_resolution(
    city: str,
    date_str: str,
    winner: str,
    pnl: float,
    positions: int,
    actual_temp: float | None = None,
):
    """Log market resolution — the truth data for backtesting."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "resolution",
        "city": city,
        "market_date": date_str,
        "winner": winner,
        "actual_temp": actual_temp,
        "pnl": round(pnl, 2),
        "positions_resolved": positions,
    }
    await _log_to_sheets(row)
    _log_to_local(row)


async def log_fill_update(
    city: str,
    date_str: str,
    bracket: str,
    order_id: str,
    status: str,
    fill_price: float | None = None,
    size_filled: float | None = None,
):
    """Log order fill status changes — crucial for slippage analysis."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "fill_update",
        "city": city,
        "market_date": date_str,
        "bracket": bracket,
        "order_id": order_id,
        "status": status,
        "fill_price": fill_price,
        "size_filled": size_filled,
    }
    _log_to_local(row)


async def log_daily_summary(summary: dict):
    """Log end-of-day summary."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "daily_summary",
        **summary,
    }
    await _log_to_sheets(row)
    _log_to_local(row)


def _save_snapshot(city: str, date_str: str, data: dict):
    """
    Save comprehensive snapshot to a per-city-per-date file.
    Multiple cycles appended as JSONL lines.
    THIS IS THE GOLD DATA FOR RECALIBRATION.
    """
    fname = SNAPSHOT_DIR / f"{city}_{date_str}.jsonl"
    try:
        with open(fname, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except IOError as e:
        log.warning(f"Snapshot save failed: {e}")


async def _log_to_sheets(row: dict):
    """Send data to Google Sheets via Apps Script webhook."""
    if not GOOGLE_SHEET_WEBHOOK:
        log.debug("Google Sheets webhook not configured, skipping")
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                GOOGLE_SHEET_WEBHOOK,
                json=row,
                follow_redirects=True,
            )
            if r.status_code == 200:
                log.debug("Logged to Google Sheets")
            else:
                log.warning(f"Sheets webhook error: {r.status_code}")
    except Exception as e:
        log.warning(f"Sheets logging failed: {e}")


def _log_to_local(row: dict):
    """Append to local JSONL file as backup."""
    try:
        with open(LOCAL_LOG_FILE, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except IOError as e:
        log.warning(f"Local log write failed: {e}")
