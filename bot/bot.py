"""
Weather Trader Bot — Main Loop
Orchestrates 30-minute cycles across all cities.
Supports DRY RUN mode and Telegram commands for remote control.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from config import CITIES, CYCLE_INTERVAL_SECONDS, ORDER_TIMEOUT_SECONDS, TELEGRAM_POLL_SECONDS, update_bankroll
from ensemble import fetch_ensemble, bias_correct, map_to_brackets
from metar import fetch_metar
from model_quality import (
    get_lead_bucket,
    get_dynamic_bias,
    record_forecast_snapshot,
    check_quality_gate,
    record_resolution_quality,
)
from market import (
    fetch_market, fetch_all_books, place_limit_order,
    cancel_order, check_order_status, bracket_degree,
    fetch_wallet_balance,
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
    send_telegram,
)
from logger import log_cycle, log_resolution, log_fill_update

# ══════════════════════════════════════════════════════
# CONTROL FILE — persists bot state across restarts
# ══════════════════════════════════════════════════════
CONTROL_FILE = Path("control.json")


def load_control() -> dict:
    if CONTROL_FILE.exists():
        try:
            return json.loads(CONTROL_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {"mode": "dryrun", "paused": False}
    # modes: "dryrun" = log only, "manual" = approve each, "auto" = trade all


def save_control(ctrl: dict):
    CONTROL_FILE.write_text(json.dumps(ctrl, indent=2))


def get_mode() -> str:
    return load_control().get("mode", "dryrun")


def is_dry_run() -> bool:
    return get_mode() == "dryrun"


def is_paused() -> bool:
    return load_control().get("paused", False)


def set_mode(mode: str | None = None, paused: bool | None = None):
    ctrl = load_control()
    if mode is not None:
        ctrl["mode"] = mode
    if paused is not None:
        ctrl["paused"] = paused
    save_control(ctrl)


# ══════════════════════════════════════════════════════
# PENDING SIGNALS — queue for manual approval
# ══════════════════════════════════════════════════════
PENDING_FILE = Path("pending_signals.json")
PENDING_META_FILE = Path("pending_meta.json")


def load_pending() -> list[dict]:
    if PENDING_FILE.exists():
        try:
            return json.loads(PENDING_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_pending(signals: list[dict]):
    PENDING_FILE.write_text(json.dumps(signals, indent=2, default=str))


def _load_pending_meta() -> dict:
    if PENDING_META_FILE.exists():
        try:
            data = json.loads(PENDING_META_FILE.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return {"next_pending_id": 1}


def _save_pending_meta(meta: dict):
    PENDING_META_FILE.write_text(json.dumps(meta, indent=2, default=str))


def _allocate_pending_id(pending: list[dict]) -> int:
    """
    Allocate a monotonic pending ID so old Telegram buttons can never map
    to newly queued signals.
    """
    meta = _load_pending_meta()
    next_from_meta = int(meta.get("next_pending_id", 1))
    max_existing = max((int(s.get("pending_id", 0) or 0) for s in pending), default=0)
    pid = max(next_from_meta, max_existing + 1)
    meta["next_pending_id"] = pid + 1
    _save_pending_meta(meta)
    return pid


def clear_pending():
    save_pending([])


def add_pending(sig: dict):
    pending = load_pending()
    sig["pending_id"] = _allocate_pending_id(pending)
    pending.append(sig)
    save_pending(pending)


# ══════════════════════════════════════════════════════
# TELEGRAM COMMANDS — poll for /commands from user
# ══════════════════════════════════════════════════════
last_update_id = 0


async def poll_telegram_commands():
    """Check for Telegram commands and button presses."""
    global last_update_id
    import httpx

    token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"offset": last_update_id + 1, "timeout": 0, "allowed_updates": '["message","callback_query"]'}

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()

        for update in data.get("result", []):
            last_update_id = update["update_id"]

            # Handle button presses (callback queries)
            cb = update.get("callback_query")
            if cb:
                cb_data = cb.get("data", "")
                cb_id = cb.get("id")
                sender = str(cb.get("from", {}).get("id", ""))

                if sender != chat_id:
                    continue

                # Answer the callback to remove loading spinner
                await answer_callback(token, cb_id)

                if cb_data.startswith("buy_"):
                    try:
                        sig_id = int(cb_data.replace("buy_", ""))
                        await execute_pending_signal(sig_id)
                    except ValueError:
                        await send_telegram("❌ Invalid signal ID.")
                elif cb_data.startswith("skip_"):
                    try:
                        sig_id = int(cb_data.replace("skip_", ""))
                        pending = load_pending()
                        pending = [s for s in pending if s.get("pending_id") != sig_id]
                        save_pending(pending)
                        await send_telegram(f"⏭ Signal #{sig_id} skipped.")
                    except ValueError:
                        await send_telegram("❌ Invalid signal ID.")
                elif cb_data == "buyall":
                    pending = load_pending()
                    if pending:
                        for sig in list(pending):
                            await execute_pending_signal(sig["pending_id"])
                    else:
                        await send_telegram("No pending signals.")
                continue

            # Handle text commands
            msg = update.get("message", {})
            text = msg.get("text", "").strip().lower()
            sender = str(msg.get("chat", {}).get("id", ""))

            if sender != chat_id:
                continue

            if text == "/pause":
                set_mode(paused=True)
                await send_telegram("⏸ <b>BOT PAUSED</b>\nNo trades until you send /resume")

            elif text == "/resume":
                set_mode(paused=False)
                await send_telegram("▶️ <b>BOT RESUMED</b>\nBack to scanning for signals")

            elif text == "/dryrun":
                set_mode(mode="dryrun")
                await send_telegram("🧪 <b>DRY RUN MODE</b>\nLogging signals but NOT placing real orders")

            elif text == "/manual":
                set_mode(mode="manual")
                await send_telegram(
                    "👆 <b>MANUAL MODE</b>\n"
                    "Signals will have ✅ Approve buttons.\n"
                    "Tap to trade, or skip."
                )

            elif text == "/auto":
                set_mode(mode="auto")
                await send_telegram("🔴 <b>AUTO TRADING MODE</b>\n⚠️ All signals will be traded automatically!")

            elif text.startswith("/buy_"):
                try:
                    sig_id = int(text.replace("/buy_", ""))
                    await execute_pending_signal(sig_id)
                except ValueError:
                    await send_telegram("❌ Invalid signal ID.")

            elif text == "/buyall":
                pending = load_pending()
                if not pending:
                    await send_telegram("No pending signals.")
                else:
                    for sig in list(pending):
                        await execute_pending_signal(sig["pending_id"])

            elif text == "/signals":
                pending = load_pending()
                if not pending:
                    await send_telegram("No pending signals right now.")
                else:
                    lines = ["📋 <b>PENDING SIGNALS</b>\n"]
                    for s in pending:
                        lines.append(
                            f"  #{s['pending_id']} → {s['city']} {s['date']} "
                            f"<b>{s['bracket']}</b> @ {s['ask']*100:.1f}¢ "
                            f"+{s['true_edge']*100:.1f}pt ${s['kelly_bet']:.2f}"
                        )
                    await send_telegram("\n".join(lines),
                        buttons=[[{"text": "✅ Buy All", "callback_data": "buyall"}]]
                    )

            elif text == "/clear":
                clear_pending()
                await send_telegram("🗑 Pending signals cleared.")

            elif text == "/status":
                ctrl = load_control()
                portfolio = get_portfolio_summary()
                pending = load_pending()
                mode_map = {"dryrun": "🧪 DRY RUN", "manual": "👆 MANUAL", "auto": "🔴 AUTO"}
                mode_str = mode_map.get(ctrl.get("mode", "dryrun"), "🧪 DRY RUN")
                state = "⏸ PAUSED" if ctrl.get("paused") else "▶️ RUNNING"
                await send_telegram(
                    f"📊 <b>STATUS</b>\n"
                    f"  Mode: {mode_str}\n"
                    f"  State: {state}\n"
                    f"  Pending signals: {len(pending)}\n"
                    f"  Active positions: {portfolio['active_positions']}\n"
                    f"  Active exposure: ${portfolio['active_exposure']:.2f}\n"
                    f"  Total trades: {portfolio['total_trades']}\n"
                    f"  Daily P&L: {'+'if get_daily_pnl()>=0 else ''}${get_daily_pnl():.2f}"
                )

            elif text == "/help":
                await send_telegram(
                    "🤖 <b>COMMANDS</b>\n\n"
                    "<b>Modes:</b>\n"
                    "  /dryrun — log only, no orders\n"
                    "  /manual — approve each trade ✅\n"
                    "  /auto — trade all automatically\n\n"
                    "<b>Trading:</b>\n"
                    "  /signals — view pending\n"
                    "  /buy_1 — approve signal #1\n"
                    "  /buyall — approve all\n"
                    "  /clear — discard pending\n\n"
                    "<b>Control:</b>\n"
                    "  /pause — stop bot\n"
                    "  /resume — resume bot\n"
                    "  /status — current state"
                )

    except Exception as e:
        log.debug(f"Telegram poll error: {e}")


async def answer_callback(token: str, callback_id: str):
    """Answer a callback query to dismiss the loading spinner on the button."""
    import httpx
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json={"callback_query_id": callback_id})
    except Exception:
        pass


async def execute_pending_signal(sig_id: int):
    """Execute a pending signal by its ID."""
    pending = load_pending()
    sig = None
    for s in pending:
        if s.get("pending_id") == sig_id:
            sig = s
            break

    if not sig:
        await send_telegram(f"❌ Signal #{sig_id} not found or already executed.")
        return

    token_id = sig.get("token_id")
    if not token_id:
        await send_telegram(f"❌ No token ID for signal #{sig_id}")
        return

    # Place the order
    order_result = await place_limit_order(
        token_id=token_id,
        price=sig["ask"],
        size=sig["contracts"],
        bracket_label=sig["bracket"],
    )

    if order_result["success"]:
        record_trade(
            city_name=sig["city"],
            date_str=sig["date"],
            bracket=sig["bracket"],
            signal="BUY",
            corrected_prob=sig["corrected_prob"],
            ask=sig["ask"],
            kelly_bet=sig["kelly_bet"],
            contracts=sig["contracts"],
            edge=sig["true_edge"],
            model_votes=sig["model_votes"],
            order_id=order_result["order_id"],
        )

        await send_telegram(
            f"✅ <b>ORDER PLACED: #{sig_id}</b>\n"
            f"  {sig['city']} {sig['date']} — <b>{sig['bracket']}</b>\n"
            f"  @ {sig['ask']*100:.1f}¢ x{sig['contracts']} = ${sig['kelly_bet']:.2f}\n"
            f"  Order ID: {order_result['order_id'][:20]}..."
        )

        log.info(f"✅ Manual order: {sig['bracket']} @ {sig['ask']:.3f} x{sig['contracts']}")
        # Remove from pending only after successful placement.
        pending = [s for s in pending if s.get("pending_id") != sig_id]
        save_pending(pending)
    else:
        await send_telegram(
            f"❌ <b>ORDER FAILED: #{sig_id}</b>\n"
            f"  {sig['city']} {sig['date']} — {sig['bracket']}\n"
            f"  {order_result['error']}\n"
            f"  (Signal kept pending. You can retry /buy_{sig_id} after fixing config, or /skip_{sig_id})"
        )

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
    Get dates to trade: today, tomorrow, and day after tomorrow.
    Polymarket weather markets open 2-3 days ahead.
    Uses UTC date as reference.
    """
    now = datetime.now(timezone.utc)
    dates = []
    for i in range(3):  # today, tomorrow, day after
        d = (now + timedelta(days=i)).strftime("%Y-%m-%d")
        dates.append(d)
    return dates


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
        lead_bucket, _lead_hours = get_lead_bucket(city["tz"], date_str)
        bias_mean_used, bias_sd_used, bias_meta = get_dynamic_bias(
            city_name=city_name,
            base_mean=city["bias_mean"],
            base_sd=city["bias_sd"],
            lead_bucket=lead_bucket,
        )
        corrected_dist = bias_correct(
            ensemble["raw_dist"], bias_mean_used, bias_sd_used
        )
        raw_probs = map_to_brackets(ensemble["raw_dist"], brackets)
        corrected_probs = map_to_brackets(corrected_dist, brackets)
        record_forecast_snapshot(
            city_name=city_name,
            date_str=date_str,
            ensemble_mean=ensemble["mean"],
            lead_bucket=lead_bucket,
            corrected_probs=corrected_probs,
        )
        if bias_meta.get("used_rolling"):
            log.info(
                f"  Rolling bias ({lead_bucket}, {bias_meta.get('samples', 0)} samples): "
                f"{city['bias_mean']:+.2f}/{city['bias_sd']:.2f} -> "
                f"{bias_mean_used:+.2f}/{bias_sd_used:.2f}"
            )

        # 3. Fetch METAR
        metar = await fetch_metar(city, date_str)

        # 4. Fetch order books
        books = await fetch_all_books(token_ids)

        # 5. Run v5 strategy (maturity → favorite → ±1 → filters → concentration)
        existing = get_existing_positions(city_name, date_str)
        exposure = get_daily_exposure()

        analysis = analyze_brackets(
            brackets=brackets,
            corrected_probs=corrected_probs,
            raw_probs=raw_probs,
            market_probs=prices,
            books=books,
            model_votes=ensemble["model_votes"],
            family_votes=ensemble.get("family_votes", {}),
            total_models=city["total_models"],
            total_families=city.get("total_families", city["total_models"]),
            existing_positions=existing,
            current_exposure=exposure,
            city_config=city,
        )

        # Handle maturity failure
        mature, maturity_reason = analysis["maturity"]
        if not mature:
            log.info(f"  ⏳ Market immature: {maturity_reason}")
            # Still log the cycle data for recalibration
            await log_cycle(
                city=city_name, date_str=date_str,
                ensemble_mean=ensemble["mean"], ensemble_count=ensemble["count"],
                metar_high=metar["day_high_market"], metar_current=metar["current_temp_c"],
                metar_obs_count=metar["observation_count"],
                brackets_data=[], signals=[], trades_placed=[],
                raw_dist=ensemble["raw_dist"], corrected_dist=corrected_dist,
                model_votes=ensemble["model_votes"], market_prices=prices,
                books_snapshot=books, max_temps=ensemble["max_temps"],
                bias_mean=bias_mean_used, bias_sd=bias_sd_used,
                current_exposure=exposure, daily_pnl=get_daily_pnl(),
                v5_favorite=None, v5_concentration=0, v5_kelly_multiplier=0,
                v5_maturity={"mature": False, "reason": maturity_reason},
                v5_divergence={"diverged": False, "distance": 0, "model_top": "", "market_fav": ""},
                v5_candidates=[],
            )
            result["status"] = f"immature:{maturity_reason}"
            return result

        # Handle divergence warning
        diverged, div_dist, model_top, market_fav = analysis["divergence"]
        if diverged:
            await send_telegram(
                f"⚠️ <b>DIVERGENCE: {city_name} {date_str}</b>\n"
                f"  Model top: {model_top}\n"
                f"  Market fav: {market_fav}\n"
                f"  Distance: {div_dist} brackets\n"
                f"  → Skipping (model & market disagree)"
            )
            # Still log for recalibration
            await log_cycle(
                city=city_name, date_str=date_str,
                ensemble_mean=ensemble["mean"], ensemble_count=ensemble["count"],
                metar_high=metar["day_high_market"], metar_current=metar["current_temp_c"],
                metar_obs_count=metar["observation_count"],
                brackets_data=analysis.get("signals", []), signals=[], trades_placed=[],
                raw_dist=ensemble["raw_dist"], corrected_dist=corrected_dist,
                model_votes=ensemble["model_votes"], market_prices=prices,
                books_snapshot=books, max_temps=ensemble["max_temps"],
                bias_mean=bias_mean_used, bias_sd=bias_sd_used,
                current_exposure=exposure, daily_pnl=get_daily_pnl(),
                v5_favorite=analysis.get("favorite"), v5_concentration=analysis.get("concentration", 0),
                v5_kelly_multiplier=analysis.get("kelly_multiplier", 0),
                v5_maturity={"mature": True, "reason": ""},
                v5_divergence={"diverged": True, "distance": div_dist, "model_top": model_top, "market_fav": market_fav},
                v5_candidates=[c["label"] for c in analysis.get("candidates", [])],
            )
            result["status"] = f"divergence:{div_dist}"
            return result

        signals = get_actionable_signals(analysis)
        totals = compute_totals(analysis)
        result["signals"] = totals["buy_count"]

        # Log favorite and concentration
        fav = analysis.get("favorite", "?")
        conc = analysis.get("concentration", 0)
        km = analysis.get("kelly_multiplier", 0)
        log.info(f"  Fav: {fav} | Conc: {conc*100:.0f}% | Kelly×{km:.2f} | Signals: {totals['buy_count']}")

        # 6. Check if trading allowed
        tradeable, reason = can_trade()
        quality_ok, quality_reason, _quality_meta = check_quality_gate(city_name)
        if tradeable and not quality_ok:
            tradeable = False
            reason = f"quality gate: {quality_reason}"
            log.warning(f"{city_name}: Trading blocked — quality gate ({quality_reason})")
        mode = get_mode()

        # 7. Handle signals based on mode
        trades_placed = []
        if tradeable and signals:
            # Clear old pending for this city/date before adding new ones
            if mode == "manual":
                pending = load_pending()
                pending = [p for p in pending if not (p.get("city") == city_name and p.get("date") == date_str)]
                save_pending(pending)

            for sig in signals:
                bracket = sig["bracket"]
                token_id = token_ids.get(bracket)

                if not token_id:
                    log.warning(f"No token ID for {bracket}, skipping")
                    continue

                fav_tag = " ⭐FAV" if sig.get("is_favorite") else ""
                conc_tag = f"Conc {sig.get('concentration', 0)*100:.0f}%"
                km_val = sig.get("kelly_multiplier", 0)

                if mode == "dryrun":
                    # DRY RUN — log everything but don't place order
                    log.info(
                        f"🧪 DRY RUN: WOULD BUY {bracket}{fav_tag} @ {sig['ask']:.3f} "
                        f"x{sig['contracts']} = ${sig['kelly_bet']:.2f} "
                        f"(edge +{sig['true_edge']*100:.1f}pt)"
                    )
                    await send_telegram(
                        f"🧪 <b>DRY RUN: {city_name} {date_str}</b>\n"
                        f"  <b>{bracket}</b>{fav_tag} @ {sig['ask']*100:.1f}¢\n"
                        f"  Edge: +{sig['true_edge']*100:.1f}pt | ${sig['kelly_bet']:.2f}\n"
                        f"  Models: {sig['model_votes']}/{city['total_models']} | {conc_tag}\n"
                        f"  Kelly×{km_val:.2f} | E[P]: +${sig['expected_profit']:.2f}"
                    )
                    trades_placed.append({**sig, "order_id": "DRY_RUN"})
                    result["trades"] += 1

                elif mode == "manual":
                    # MANUAL — queue signal with approve/skip buttons
                    pending_sig = {
                        "city": city_name,
                        "date": date_str,
                        "bracket": bracket,
                        "token_id": token_id,
                        "ask": sig["ask"],
                        "contracts": sig["contracts"],
                        "kelly_bet": sig["kelly_bet"],
                        "true_edge": sig["true_edge"],
                        "corrected_prob": sig["corrected_prob"],
                        "model_votes": sig["model_votes"],
                        "total_models": city["total_models"],
                        "expected_profit": sig["expected_profit"],
                        "concentration": sig.get("concentration", 0),
                        "kelly_multiplier": sig.get("kelly_multiplier", 0),
                        "is_favorite": sig.get("is_favorite", False),
                    }
                    add_pending(pending_sig)
                    pid = load_pending()[-1]["pending_id"]

                    log.info(
                        f"👆 QUEUED #{pid}: {bracket}{fav_tag} @ {sig['ask']:.3f} "
                        f"x{sig['contracts']} = ${sig['kelly_bet']:.2f}"
                    )
                    await send_telegram(
                        f"👆 <b>SIGNAL #{pid}: {city_name} {date_str}</b>\n"
                        f"  <b>{bracket}</b>{fav_tag} @ {sig['ask']*100:.1f}¢\n"
                        f"  Edge: +{sig['true_edge']*100:.1f}pt | ${sig['kelly_bet']:.2f}\n"
                        f"  Models: {sig['model_votes']}/{city['total_models']} | {conc_tag}\n"
                        f"  Kelly×{km_val:.2f} | E[P]: +${sig['expected_profit']:.2f}",
                        buttons=[[
                            {"text": "✅ Approve", "callback_data": f"buy_{pid}"},
                            {"text": "⏭ Skip", "callback_data": f"skip_{pid}"},
                        ]]
                    )
                    trades_placed.append({**sig, "order_id": f"PENDING_{pid}"})
                    result["trades"] += 1

                elif mode == "auto":
                    # AUTO — place real order immediately
                    order_result = await place_limit_order(
                        token_id=token_id,
                        price=sig["ask"],
                        size=sig["contracts"],
                        bracket_label=bracket,
                    )

                    if order_result["success"]:
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
            # Log best candidate edge even if no signal passed
            all_sigs = analysis.get("signals", [])
            if all_sigs:
                best = all_sigs[0]
                reasons = ", ".join(best.get("filter_reasons", []))
                log.info(
                    f"{city_name}: No signals passed. Best: "
                    f"{best['bracket']} edge={best['true_edge']*100:.1f}pt [{reasons}]"
                )
            else:
                log.info(f"{city_name}: No candidates after favorite ±1 filter")

        # 8. Log cycle — comprehensive data for recalibration
        await log_cycle(
            city=city_name,
            date_str=date_str,
            ensemble_mean=ensemble["mean"],
            ensemble_count=ensemble["count"],
            metar_high=metar["day_high_market"],
            metar_current=metar["current_temp_c"],
            metar_obs_count=metar["observation_count"],
            brackets_data=analysis.get("signals", []),
            signals=signals,
            trades_placed=trades_placed,
            raw_dist=ensemble["raw_dist"],
            corrected_dist=corrected_dist,
            model_votes=ensemble["model_votes"],
            market_prices=prices,
            books_snapshot=books,
            max_temps=ensemble["max_temps"],
            bias_mean=bias_mean_used,
            bias_sd=bias_sd_used,
            current_exposure=exposure,
            daily_pnl=get_daily_pnl(),
            v5_favorite=analysis.get("favorite"),
            v5_concentration=analysis.get("concentration", 0),
            v5_kelly_multiplier=analysis.get("kelly_multiplier", 0),
            v5_maturity={"mature": analysis.get("maturity", (True, ""))[0], "reason": analysis.get("maturity", (True, ""))[1]},
            v5_divergence={"diverged": analysis.get("divergence", (False, 0, "", ""))[0], "distance": analysis.get("divergence", (False, 0, "", ""))[1], "model_top": analysis.get("divergence", (False, 0, "", ""))[2], "market_fav": analysis.get("favorite", "")},
            v5_candidates=[c["label"] for c in analysis.get("candidates", [])],
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
                try:
                    q = await record_resolution_quality(city, date_str, res["winner"])
                    log.info(
                        f"Quality update {city_name} {date_str}: "
                        f"winner_p={q.get('winner_prob', 0):.3f}, brier={q.get('brier', 0):.3f}"
                    )
                except Exception as qe:
                    log.warning(f"Quality update failed for {city_name} {date_str}: {qe}")

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
                    await alert_kill_switch(get_daily_pnl(), config.DAILY_LOSS_LIMIT)

        except Exception as e:
            log.warning(f"Resolution check failed for {city_name} {date_str}: {e}")


async def run_cycle():
    """Run one complete cycle across all cities and dates."""
    global cycle_count
    cycle_count += 1

    # Poll Telegram commands first
    await poll_telegram_commands()

    dry_run = is_dry_run()
    paused = is_paused()
    mode = get_mode()
    mode_map = {"dryrun": "🧪 DRY RUN", "manual": "👆 MANUAL", "auto": "🔴 AUTO"}
    mode_tag = mode_map.get(mode, "🧪 DRY RUN")

    log.info(f"╔══════════════════════════════════════╗")
    log.info(f"║  CYCLE #{cycle_count}  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ║")
    log.info(f"║  Mode: {mode_tag}  {'⏸ PAUSED' if paused else ''}  ║")
    log.info(f"╚══════════════════════════════════════╝")

    # Check pause
    if paused:
        log.info("Bot is PAUSED — skipping cycle. Send /resume on Telegram.")
        return

    # Check kill switch
    if is_kill_switch_active():
        log.warning("Kill switch active — skipping trading cycle")
        return

    # Fetch live wallet balance and update limits
    balance = await fetch_wallet_balance()
    if balance > 0:
        update_bankroll(balance)
        log.info(
            f"💰 Bankroll: ${config.BANKROLL:.2f} | "
            f"Max bet: ${config.MAX_BET_PER_BRACKET:.2f} | "
            f"Max exposure: ${config.MAX_TOTAL_EXPOSURE:.2f} | "
            f"Kill: ${config.DAILY_LOSS_LIMIT:.2f}"
        )
    elif balance == 0:
        update_bankroll(0)
        log.warning("Wallet balance is $0 — no trades this cycle")
    else:
        log.warning("Balance fetch failed — skipping trades this cycle (bankroll $0)")
        if config.BANKROLL <= 0:
            # Don't trade on stale/missing data
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

    # Check pending orders from previous cycles (skip in dry run)
    if mode == "auto":
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
        f"{total_signals} signals, {total_trades} {'queued' if mode == 'manual' else 'dry run signals' if mode == 'dryrun' else 'trades placed'}"
    )


async def main():
    """Main entry point — run cycles forever."""
    from config import POLYGON_PRIVATE_KEY, TELEGRAM_BOT_TOKEN

    # Initialize control file with dryrun mode
    if not CONTROL_FILE.exists():
        save_control({"mode": "dryrun", "paused": False})

    ctrl = load_control()
    mode_map = {"dryrun": "🧪 DRY RUN", "manual": "👆 MANUAL", "auto": "🔴 AUTO"}
    mode = mode_map.get(ctrl.get("mode", "dryrun"), "🧪 DRY RUN")

    log.info("=" * 60)
    log.info("  WEATHER TRADER BOT — STARTING")
    log.info("=" * 60)

    # Fetch initial wallet balance
    balance = await fetch_wallet_balance()
    if balance > 0:
        update_bankroll(balance)
        bankroll_str = f"${config.BANKROLL:.2f}"
    elif balance == 0:
        bankroll_str = "⚠️ $0.00 (empty wallet)"
    else:
        bankroll_str = "⚠️ FETCH FAILED (no trades until resolved)"

    log.info(f"  Mode: {mode}")
    log.info(f"  Bankroll: {bankroll_str} | Kelly: full × conc | Max exposure: ${config.MAX_TOTAL_EXPOSURE:.2f}")
    log.info(f"  Strategy: v5 (favorite ±1, per-city filters, concentration)")
    log.info(f"  Max ask: 50¢ | Telegram poll: {TELEGRAM_POLL_SECONDS}s")
    log.info(f"  Kill switch: ${config.DAILY_LOSS_LIMIT:.2f} daily loss")
    log.info(f"  Cycle interval: {CYCLE_INTERVAL_SECONDS}s")
    log.info(f"  Wallet configured: {'YES' if POLYGON_PRIVATE_KEY else 'NO'}")
    log.info(f"  Telegram configured: {'YES' if TELEGRAM_BOT_TOKEN else 'NO'}")
    log.info("=" * 60)

    await send_telegram(
        f"🚀 <b>Weather Trader Bot v5 Started</b>\n"
        f"  Mode: {mode}\n"
        f"  Strategy: favorite ±1, concentration Kelly\n"
        f"  Bankroll: {bankroll_str} | Kelly: full × conc\n"
        f"  Monitoring: London, Seoul, NYC, Seattle\n"
        f"  Cycle: every 30 min | Poll: every 60s\n\n"
        f"  <b>Commands:</b>\n"
        f"  /manual — approve each trade ✅\n"
        f"  /dryrun — log only (current)\n"
        f"  /auto — auto-trade all signals\n"
        f"  /signals — view pending signals\n"
        f"  /buy_1 — approve signal #1\n"
        f"  /status — check bot state\n"
        f"  /help — all commands"
    )

    while True:
        try:
            await run_cycle()
        except Exception as e:
            log.error(f"Critical cycle error: {e}", exc_info=True)
            await alert_error(f"Critical: {e}")

        # Fast polling between cycles — check Telegram every 60s
        log.info(f"Next cycle in {CYCLE_INTERVAL_SECONDS}s (polling Telegram every {TELEGRAM_POLL_SECONDS}s)...")
        elapsed = 0
        while elapsed < CYCLE_INTERVAL_SECONDS:
            await asyncio.sleep(TELEGRAM_POLL_SECONDS)
            elapsed += TELEGRAM_POLL_SECONDS
            await poll_telegram_commands()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
    except Exception as e:
        log.critical(f"Bot crashed: {e}", exc_info=True)
