"""
Weather Trader Bot — Main Loop
Orchestrates 30-minute cycles across all cities.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from config import CITIES, CYCLE_INTERVAL_SECONDS, ORDER_TIMEOUT_SECONDS
from ensemble import fetch_ensemble, bias_correct, map_to_brackets
from metar import fetch_metar
from market import (
    fetch_market, fetch_all_books, place_limit_order,
    cancel_order, check_order_status, bracket_degree,
)
from strategy import analyze_brackets, get_actionable_signals, compute_totals
from risk import (
    can_trade, record_trade, update_fill_status, get_existing_positions,
    get_daily_exposure, get_daily_pnl, is_kill_switch_active,
    get_pending_resolutions, record_resolution, get_portfolio_summary,
)
from alerts import (
    alert_trade, alert_resolution, alert_kill_switch, alert_heartbeat,
    alert_error, alert_order_update, alert_startup, alert_no_signals,
)
from logger import log_cycle, log_resolution, log_fill_update

# ══════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-10s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", mode="a"),
    ],
)
log = logging.getLogger("bot")

cycle_count = 0


def get_market_dates() -> list[str]:
    """
    Get dates to trade: today and tomorrow.
    Uses UTC date as reference.
    """
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    return [today, tomorrow]


async def process_city(city_key: str, city: dict, date_str: str) -> dict:
    """
    Run full analysis + trading cycle for one city on one date.
    Returns summary dict.
    """
    city_name = city["name"]
    log.info(f"━━━ {city_name} · {date_str} ━━━")

    result = {
        "city": city_name,
        "date": date_str,
        "signals": 0,
        "trades": 0,
        "error": None,
    }

    try:
        # 1. Fetch market data (brackets + prices)
        market = await fetch_market(city, date_str)

        if market["resolved"]:
            log.info(f"{city_name} {date_str}: Already resolved → {market['winner']}")
            result["status"] = f"resolved:{market['winner']}"
            return result

        brackets = market["brackets"]
        prices = market["prices"]
        token_ids = market["token_ids"]

        if not brackets:
            log.warning(f"{city_name} {date_str}: No brackets found")
            result["error"] = "no_brackets"
            return result

        # 2. Fetch ensemble forecasts + bias correction
        ensemble = await fetch_ensemble(city, date_str)
        corrected_dist = bias_correct(
            ensemble["raw_dist"], city["bias_mean"], city["bias_sd"]
        )
        raw_probs = map_to_brackets(ensemble["raw_dist"], brackets)
        corrected_probs = map_to_brackets(corrected_dist, brackets)

        # 3. Fetch METAR
        metar = await fetch_metar(city, date_str)

        # 4. Fetch order books
        books = await fetch_all_books(token_ids)

        # 5. Run strategy analysis
        existing = get_existing_positions(city_name, date_str)
        exposure = get_daily_exposure()

        analysis = analyze_brackets(
            brackets=brackets,
            corrected_probs=corrected_probs,
            raw_probs=raw_probs,
            market_probs=prices,
            books=books,
            model_votes=ensemble["model_votes"],
            total_models=city["total_models"],
            existing_positions=existing,
            current_exposure=exposure,
        )

        signals = get_actionable_signals(analysis)
        totals = compute_totals(analysis)
        result["signals"] = totals["buy_count"]

        # 6. Check if trading allowed
        tradeable, reason = can_trade()

        # 7. Place orders for BUY signals
        trades_placed = []
        if tradeable and signals:
            for sig in signals:
                bracket = sig["bracket"]
                token_id = token_ids.get(bracket)

                if not token_id:
                    log.warning(f"No token ID for {bracket}, skipping")
                    continue

                # Place limit order at best ask
                order_result = await place_limit_order(
                    token_id=token_id,
                    price=sig["ask"],
                    size=sig["contracts"],
                    bracket_label=bracket,
                )

                if order_result["success"]:
                    # Record in risk manager
                    record_trade(
                        city_name=city_name,
                        date_str=date_str,
                        bracket=bracket,
                        signal=sig["signal"],
                        corrected_prob=sig["corrected_prob"],
                        ask=sig["ask"],
                        kelly_bet=sig["kelly_bet"],
                        contracts=sig["contracts"],
                        edge=sig["true_edge"],
                        model_votes=sig["model_votes"],
                        order_id=order_result["order_id"],
                    )

                    # Send Telegram alert
                    await alert_trade(
                        city=city_name,
                        date_str=date_str,
                        bracket=bracket,
                        ask=sig["ask"],
                        edge=sig["true_edge"],
                        kelly_bet=sig["kelly_bet"],
                        contracts=sig["contracts"],
                        model_votes=sig["model_votes"],
                        total_models=city["total_models"],
                        expected_profit=sig["expected_profit"],
                    )

                    trades_placed.append({
                        **sig,
                        "order_id": order_result["order_id"],
                    })
                    result["trades"] += 1

                    log.info(
                        f"✅ ORDER: {bracket} @ {sig['ask']:.3f} "
                        f"x{sig['contracts']} = ${sig['kelly_bet']:.2f}"
                    )
                else:
                    log.error(
                        f"❌ ORDER FAILED: {bracket} — {order_result['error']}"
                    )
                    await alert_error(
                        f"Order failed: {city_name} {bracket} — {order_result['error']}"
                    )

        elif not tradeable:
            log.info(f"{city_name}: Trading blocked — {reason}")
        elif not signals:
            # Log best edge even if no signal
            best = analysis[0] if analysis else None
            if best:
                log.info(
                    f"{city_name}: No signals. Best edge: "
                    f"{best['bracket']} {best['true_edge'] * 100:.1f}pt "
                    f"(need ≥5pt)"
                )

        # 8. Log cycle — comprehensive data for recalibration
        await log_cycle(
            city=city_name,
            date_str=date_str,
            ensemble_mean=ensemble["mean"],
            ensemble_count=ensemble["count"],
            metar_high=metar["day_high_market"],
            metar_current=metar["current_temp_c"],
            metar_obs_count=metar["observation_count"],
            brackets_data=analysis,
            signals=signals,
            trades_placed=trades_placed,
            raw_dist=ensemble["raw_dist"],
            corrected_dist=corrected_dist,
            model_votes=ensemble["model_votes"],
            market_prices=prices,
            books_snapshot=books,
            max_temps=ensemble["max_temps"],
            bias_mean=city["bias_mean"],
            bias_sd=city["bias_sd"],
            current_exposure=exposure,
            daily_pnl=get_daily_pnl(),
        )

        result["status"] = "ok"
        return result

    except Exception as e:
        log.error(f"{city_name} {date_str}: Error — {e}", exc_info=True)
        result["error"] = str(e)
        return result


async def check_pending_orders():
    """
    Check status of pending orders and cancel if timed out.
    """
    from risk import get_state

    state = get_state()
    now = datetime.now(timezone.utc)

    for key, pos in state.get("positions", {}).items():
        if pos.get("resolved") or pos.get("fill_status") != "pending":
            continue

        order_id = pos.get("order_id")
        if not order_id:
            continue

        # Check if order has been pending too long
        try:
            placed_at = datetime.fromisoformat(pos["timestamp"])
            elapsed = (now - placed_at).total_seconds()
        except (ValueError, KeyError):
            elapsed = 0

        if elapsed > ORDER_TIMEOUT_SECONDS:
            # Cancel the order
            log.info(f"Order timeout ({elapsed:.0f}s): cancelling {order_id}")
            cancelled = await cancel_order(order_id)
            if cancelled:
                update_fill_status(
                    pos["city"], pos["market_date"], pos["bracket"], "cancelled"
                )
                await log_fill_update(
                    pos["city"], pos["market_date"], pos["bracket"],
                    order_id, "cancelled_timeout",
                )
                await alert_order_update(pos["bracket"], "cancelled (timeout)", order_id)
        else:
            # Check if filled
            status = await check_order_status(order_id)
            if status != "open" and status != "unknown":
                update_fill_status(
                    pos["city"], pos["market_date"], pos["bracket"], status
                )
                await log_fill_update(
                    pos["city"], pos["market_date"], pos["bracket"],
                    order_id, status,
                )
                await alert_order_update(pos["bracket"], status, order_id)


async def check_resolutions():
    """Check if any past markets have resolved."""
    pending = get_pending_resolutions()

    for p in pending:
        city_name = p["city"]
        date_str = p["date"]

        # Find city config by name
        city = None
        for cfg in CITIES.values():
            if cfg["name"] == city_name:
                city = cfg
                break

        if not city:
            continue

        try:
            from market import check_resolution as market_check_resolution

            res = await market_check_resolution(city, date_str)
            if res["resolved"] and res["winner"]:
                pnl = record_resolution(city_name, date_str, res["winner"])

                await alert_resolution(
                    city=city_name,
                    date_str=date_str,
                    winner=res["winner"],
                    pnl=pnl,
                    positions_count=len(
                        [
                            k
                            for k in get_state().get("positions", {})
                            if k.startswith(f"{city_name}:{date_str}:")
                        ]
                    ),
                )

                await log_resolution(
                    city=city_name,
                    date_str=date_str,
                    winner=res["winner"],
                    pnl=pnl,
                    positions=0,
                )

                # Check if kill switch should trigger
                if is_kill_switch_active():
                    await alert_kill_switch(get_daily_pnl(), DAILY_LOSS_LIMIT)

        except Exception as e:
            log.warning(f"Resolution check failed for {city_name} {date_str}: {e}")


async def run_cycle():
    """Run one complete cycle across all cities and dates."""
    global cycle_count
    cycle_count += 1

    log.info(f"╔══════════════════════════════════════╗")
    log.info(f"║  CYCLE #{cycle_count}  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ║")
    log.info(f"╚══════════════════════════════════════╝")

    # Check kill switch
    if is_kill_switch_active():
        log.warning("Kill switch active — skipping trading cycle")
        return

    dates = get_market_dates()
    city_summaries = []

    for city_key, city in CITIES.items():
        for date_str in dates:
            try:
                result = await process_city(city_key, city, date_str)
                city_summaries.append(result)
            except Exception as e:
                log.error(f"Unhandled error in {city['name']} {date_str}: {e}")
                await alert_error(f"Cycle error: {city['name']} {date_str} — {e}")

    # Check pending orders from previous cycles
    await check_pending_orders()

    # Check resolutions
    await check_resolutions()

    # Heartbeat every 6 cycles (~3 hours)
    if cycle_count % 6 == 0:
        portfolio = get_portfolio_summary()
        summaries_str = " | ".join(
            f"{r['city']}: {'signal' if r.get('signals', 0) > 0 else 'no edge'}"
            for r in city_summaries
            if r.get("date") == dates[0]  # today only
        )
        await alert_heartbeat(
            cycle_num=cycle_count,
            cities_summary=summaries_str,
            active_positions=portfolio["active_positions"],
            daily_pnl=get_daily_pnl(),
        )

    total_trades = sum(r.get("trades", 0) for r in city_summaries)
    total_signals = sum(r.get("signals", 0) for r in city_summaries)

    log.info(
        f"Cycle #{cycle_count} complete: "
        f"{total_signals} signals, {total_trades} trades placed"
    )


async def main():
    """Main entry point — run cycles forever."""
    from config import POLYGON_PRIVATE_KEY, TELEGRAM_BOT_TOKEN

    log.info("=" * 60)
    log.info("  WEATHER TRADER BOT — STARTING")
    log.info("=" * 60)
    log.info(f"  Bankroll: $50 | Kelly: ⅓ | Max exposure: $20")
    log.info(f"  Min edge: 5pt | Min models: 2 | Max ask: 50¢")
    log.info(f"  Kill switch: $30 daily loss")
    log.info(f"  Cycle interval: {CYCLE_INTERVAL_SECONDS}s")
    log.info(f"  Wallet configured: {'YES' if POLYGON_PRIVATE_KEY else 'NO'}")
    log.info(f"  Telegram configured: {'YES' if TELEGRAM_BOT_TOKEN else 'NO'}")
    log.info("=" * 60)

    await alert_startup()

    while True:
        try:
            await run_cycle()
        except Exception as e:
            log.error(f"Critical cycle error: {e}", exc_info=True)
            await alert_error(f"Critical: {e}")

        log.info(f"Next cycle in {CYCLE_INTERVAL_SECONDS}s...")
        await asyncio.sleep(CYCLE_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
    except Exception as e:
        log.critical(f"Bot crashed: {e}", exc_info=True)
