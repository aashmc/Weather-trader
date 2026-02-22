"""
Weather Trader Bot — Telegram Alerts
Sends trade notifications, resolution alerts, and heartbeats.
"""

import json
import logging
import httpx
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger("alerts")


async def send_telegram(message: str, buttons: list[list[dict]] = None):
    """Send a message via Telegram bot, optionally with inline buttons."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.debug("Telegram not configured, skipping alert")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    if buttons:
        payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                log.warning(f"Telegram API error: {r.status_code} — {r.text[:100]}")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


async def alert_trade(
    city: str,
    date_str: str,
    bracket: str,
    ask: float,
    edge: float,
    kelly_bet: float,
    contracts: int,
    model_votes: int,
    total_models: int,
    expected_profit: float,
):
    """Alert on a new trade."""
    msg = (
        f"🟢 <b>TRADE: {city} {date_str}</b>\n"
        f"  Buy YES <b>{bracket}</b> @ {ask * 100:.1f}¢ (limit)\n"
        f"  Edge: +{edge * 100:.1f}pt | Kelly: ${kelly_bet:.2f}\n"
        f"  Contracts: {contracts} | Models: {model_votes}/{total_models}\n"
        f"  E[Profit]: +${expected_profit:.2f}"
    )
    await send_telegram(msg)


async def alert_resolution(
    city: str,
    date_str: str,
    winner: str,
    pnl: float,
    positions_count: int,
):
    """Alert on market resolution."""
    emoji = "🏆" if pnl >= 0 else "🏁"
    sign = "+" if pnl >= 0 else ""
    msg = (
        f"{emoji} <b>RESOLVED: {city} {date_str} → {winner}</b>\n"
        f"  P&L: {sign}${pnl:.2f} ({positions_count} positions)\n"
        f"  ⚠️ Redeem manually on Polymarket"
    )
    await send_telegram(msg)


async def alert_kill_switch(daily_pnl: float, limit: float):
    """Alert when kill switch triggers."""
    msg = (
        f"💀 <b>KILL SWITCH ACTIVATED</b>\n"
        f"  Daily loss: ${abs(daily_pnl):.2f} ≥ ${limit:.2f} cap\n"
        f"  Bot paused until tomorrow"
    )
    await send_telegram(msg)


async def alert_no_signals(cities_checked: int):
    """Periodic update when no trades found."""
    # Only send occasionally to avoid spam — caller controls frequency
    msg = (
        f"💤 No actionable signals\n"
        f"  Checked {cities_checked} cities, all below thresholds"
    )
    await send_telegram(msg)


async def alert_heartbeat(
    cycle_num: int,
    cities_summary: str,
    active_positions: int,
    daily_pnl: float,
):
    """Periodic heartbeat to confirm bot is alive."""
    msg = (
        f"💚 <b>HEARTBEAT</b> — Cycle #{cycle_num}\n"
        f"  {cities_summary}\n"
        f"  Active positions: {active_positions}\n"
        f"  Daily P&L: {'+'if daily_pnl >= 0 else ''}${daily_pnl:.2f}"
    )
    await send_telegram(msg)


async def alert_error(error_msg: str):
    """Alert on critical errors."""
    msg = f"🔴 <b>ERROR</b>\n  {error_msg}"
    await send_telegram(msg)


async def alert_order_update(bracket: str, status: str, order_id: str):
    """Alert on order fill/cancel."""
    emoji = "✅" if status == "filled" else ("⏳" if status == "partial" else "❌")
    msg = f"{emoji} Order {status}: <b>{bracket}</b>\n  ID: {order_id[:20]}..."
    await send_telegram(msg)


async def alert_startup():
    """Send startup notification."""
    msg = (
        "🚀 <b>Weather Trader Bot Started</b>\n"
        "  Monitoring: London, Seoul, NYC, Seattle\n"
        "  Mode: Live auto-trading\n"
        "  Cycle: every 30 min"
    )
    await send_telegram(msg)
