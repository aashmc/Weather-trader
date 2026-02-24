"""
Weather Trader Bot — Dashboard Control API
Provides read/write endpoints for live dashboard state and bot controls.
"""

import asyncio
from collections import deque
import json
import logging
import mimetypes
import os
import subprocess
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import parse_qs

import httpx
import config
from ensemble import fetch_ensemble, bias_correct, map_to_brackets
from market import fetch_wallet_balance
from market import fetch_market, fetch_all_books, build_slug, GAMMA_API
from metar import fetch_metar
from risk import get_portfolio_summary, get_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-12s] %(levelname)s: %(message)s",
)
log = logging.getLogger("dashboard_api")

HOST = os.getenv("DASHBOARD_API_HOST", "0.0.0.0")
PORT = int(os.getenv("DASHBOARD_API_PORT", "8090"))
API_TOKEN = os.getenv("DASHBOARD_API_TOKEN", "").strip()

CONTROL_FILE = Path("control.json")
PENDING_FILE = Path("pending_signals.json")
WEB_ROOT = Path(__file__).resolve().parent.parent
BOT_LOG_FILE = WEB_ROOT / "bot.log"
BOT_SERVICE_NAME = os.getenv("WEATHER_BOT_SERVICE", "weather-trader.service").strip()


def _load_json(path: Path, fallback):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, IOError):
            return fallback
    return fallback


def _save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2))


def _load_control() -> dict:
    return _load_json(CONTROL_FILE, {"mode": "dryrun", "paused": False})


def _save_control(ctrl: dict):
    _save_json(CONTROL_FILE, ctrl)


def _pending_count() -> int:
    pending = _load_json(PENDING_FILE, [])
    return len(pending) if isinstance(pending, list) else 0


def _active_positions_payload() -> tuple[list[dict], dict[str, list[str]]]:
    state = get_state()
    active = []
    by_market = {}
    for pos in state.get("positions", {}).values():
        if pos.get("resolved"):
            continue
        entry = {
            "city": pos.get("city", ""),
            "market_date": pos.get("market_date", ""),
            "bracket": pos.get("bracket", ""),
            "wager": pos.get("wager", 0.0),
            "ask": pos.get("ask", 0.0),
            "fill_status": pos.get("fill_status", ""),
            "order_id": pos.get("order_id", ""),
        }
        active.append(entry)
        key = f"{entry['city']}:{entry['market_date']}"
        by_market.setdefault(key, []).append(entry["bracket"])
    return active, by_market


def _fetch_cash_balance_sync() -> float:
    try:
        return float(asyncio.run(fetch_wallet_balance()))
    except Exception:
        return -1.0


def build_summary() -> dict:
    config.refresh_runtime_overrides()
    ctrl = _load_control()
    portfolio = get_portfolio_summary()
    active, active_by_market = _active_positions_payload()
    cash = _fetch_cash_balance_sync()
    if cash >= 0:
        config.update_bankroll(cash)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "control": {
            "mode": ctrl.get("mode", "dryrun"),
            "paused": bool(ctrl.get("paused", False)),
        },
        "bankroll": {
            "cash": cash,
            "max_bet_per_bracket": config.MAX_BET_PER_BRACKET,
            "max_total_exposure": config.MAX_TOTAL_EXPOSURE,
            "daily_loss_limit": config.DAILY_LOSS_LIMIT,
        },
        "portfolio": portfolio,
        "pending_signals": _pending_count(),
        "active_positions": active,
        "active_by_market": active_by_market,
        "runtime_effective": config.get_runtime_effective(),
    }


def _safe_city_key(raw: str) -> str:
    key = (raw or "").strip().lower()
    if key not in config.CITIES:
        raise ValueError(f"invalid city '{raw}'")
    return key


def _safe_date(raw: str) -> str:
    try:
        d = datetime.strptime((raw or "").strip(), "%Y-%m-%d")
        return d.strftime("%Y-%m-%d")
    except Exception:
        raise ValueError(f"invalid date '{raw}' (expected YYYY-MM-DD)")


def _book_to_dashboard_shape(book: dict) -> dict:
    bb = float(book.get("bb", 0.0))
    ba = float(book.get("ba", 1.0))
    spread = float(book.get("spread", max(0.0, ba - bb)))
    bid_depth = float(book.get("bid_depth", 0.0))
    ask_depth = float(book.get("ask_depth", 0.0))
    bids = book.get("bids", []) or []
    asks = book.get("asks", []) or []
    return {
        "bb": bb,
        "ba": ba,
        "mid": (bb + ba) / 2.0,
        "spd": spread,
        "spdPct": ((spread / ((bb + ba) / 2.0)) * 100.0) if (bb + ba) > 0 else 0.0,
        "bidD": bid_depth,
        "askD": ask_depth,
        "last": float(book.get("last_trade_price", 0.0) or 0.0),
        "bids": bids,
        "asks": asks,
    }


async def _fetch_gas_estimate() -> dict:
    # Same rough estimate used by dashboard-side logic.
    gas_url = "https://gasstation.polygon.technology/v2"
    pol_url = (
        "https://api.coingecko.com/api/v3/simple/price"
        "?ids=polygon-ecosystem-token&vs_currencies=usd"
    )
    gas_usd = 0.03
    pol_usd = 0.35
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            gas_r, pol_r = await asyncio.gather(
                client.get(gas_url),
                client.get(pol_url),
            )
        gas_r.raise_for_status()
        pol_r.raise_for_status()
        gas = gas_r.json()
        pol = pol_r.json()
        base = float(gas.get("estimatedBaseFee", 0.0))
        prio = float((gas.get("standard") or {}).get("maxPriorityFee", 0.0))
        pol_usd = float((pol.get("polygon-ecosystem-token") or {}).get("usd", pol_usd))
        total_gwei = base + prio
        gas_usd = 200000 * total_gwei / 1e9 * pol_usd
    except Exception:
        pass
    return {"gas_usd": float(gas_usd), "pol_usd": float(pol_usd)}


async def _discover_city_markets_async(city_key: str) -> dict:
    city = config.CITIES[city_key]
    today = datetime.now(timezone.utc).date()
    dates = []
    async with httpx.AsyncClient(timeout=10) as client:
        for offset in range(-3, 2):
            d = today + timedelta(days=offset)
            date_str = d.strftime("%Y-%m-%d")
            slug = build_slug(city, date_str)
            url = f"{GAMMA_API}?slug={slug}"
            try:
                r = await client.get(url)
                r.raise_for_status()
                payload = r.json()
                if not payload:
                    continue
                event = payload[0] if isinstance(payload, list) else payload
                markets = event.get("markets") or []
                if not markets:
                    continue
                dates.append(
                    {
                        "date_str": date_str,
                        "active": bool(event.get("active", False)),
                        "closed": bool(event.get("closed", False)),
                        "slug": slug,
                    }
                )
            except Exception:
                continue
    return {"city": city_key, "dates": dates}


def discover_city_markets(city_key: str) -> dict:
    return asyncio.run(_discover_city_markets_async(city_key))


async def _build_dashboard_snapshot_async(city_key: str, date_str: str) -> dict:
    city = config.CITIES[city_key]
    market = await fetch_market(city, date_str)
    brackets = market.get("brackets", [])
    prices = market.get("prices", {})
    token_ids = market.get("token_ids", {})

    ensemble_task = fetch_ensemble(city, date_str)
    metar_task = fetch_metar(city, date_str)
    books_task = fetch_all_books(token_ids)
    gas_task = _fetch_gas_estimate()
    ensemble_res, metar_res, books_res, gas_res = await asyncio.gather(
        ensemble_task,
        metar_task,
        books_task,
        gas_task,
        return_exceptions=True,
    )

    if isinstance(ensemble_res, Exception):
        raise ensemble_res
    ensemble_data = ensemble_res

    raw_probs = map_to_brackets(ensemble_data["raw_dist"], brackets)
    corrected_dist = bias_correct(
        ensemble_data["raw_dist"], city["bias_mean"], city["bias_sd"]
    )
    corrected_probs = map_to_brackets(corrected_dist, brackets)

    if isinstance(metar_res, Exception):
        metar_data = {
            "current_temp_c": None,
            "day_high_c": None,
            "day_high_market": None,
            "observation_count": 0,
            "latest_raw": "",
        }
    else:
        metar_data = metar_res

    books_raw = {} if isinstance(books_res, Exception) else books_res
    books = {
        label: _book_to_dashboard_shape(book)
        for label, book in (books_raw or {}).items()
    }

    gas_data = {"gas_usd": 0.03, "pol_usd": 0.35} if isinstance(gas_res, Exception) else gas_res

    return {
        "city": city_key,
        "date": date_str,
        "market": {
            "slug": market.get("slug"),
            "active": bool(market.get("active", True)),
            "resolved": bool(market.get("resolved", False)),
            "winner": market.get("winner"),
            "brackets": [b["label"] for b in brackets],
            "prices": prices,
            "token_ids": token_ids,
        },
        "books": books,
        "ensemble": {
            "mean": float(ensemble_data.get("mean", 0.0)),
            "count": int(ensemble_data.get("count", 0)),
            "raw_probs": raw_probs,
            "corrected_probs": corrected_probs,
            "model_agreement": ensemble_data.get("model_votes", {}),
            "family_agreement": ensemble_data.get("family_votes", {}),
        },
        "metar": metar_data,
        "gas": gas_data,
    }


def build_dashboard_snapshot(city_key: str, date_str: str) -> dict:
    return asyncio.run(_build_dashboard_snapshot_async(city_key, date_str))


def _tail_file_lines(path: Path, n: int) -> list[str]:
    if n <= 0:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return list(deque((line.rstrip("\n") for line in f), maxlen=n))
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _file_is_recent(path: Path, max_age_seconds: int = 180) -> bool:
    try:
        mtime = path.stat().st_mtime
        age = datetime.now(timezone.utc).timestamp() - float(mtime)
        return age <= max_age_seconds
    except Exception:
        return False


def _tail_journal_lines(unit_name: str, n: int) -> list[str]:
    if not unit_name:
        return []
    try:
        result = subprocess.run(
            [
                "journalctl",
                "-u",
                unit_name,
                "-n",
                str(max(1, n)),
                "--no-pager",
                "--output=short-iso",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=4,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _get_live_bot_logs(n: int) -> tuple[str, list[str]]:
    # Prefer file logs when they are fresh; otherwise fall back to systemd journal.
    if _file_is_recent(BOT_LOG_FILE, max_age_seconds=180):
        return str(BOT_LOG_FILE), _tail_file_lines(BOT_LOG_FILE, n)

    journal_lines = _tail_journal_lines(BOT_SERVICE_NAME, n)
    if journal_lines:
        return f"journalctl:{BOT_SERVICE_NAME}", journal_lines

    # Last fallback: stale file tail (if present) so UI still shows context.
    return str(BOT_LOG_FILE), _tail_file_lines(BOT_LOG_FILE, n)


class Handler(BaseHTTPRequestHandler):
    server_version = "WeatherDashboardAPI/1.0"

    def _set_headers(self, code=200, content_type="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _json(self, code: int, payload: dict):
        self._set_headers(code)
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _authorized(self) -> bool:
        if not API_TOKEN:
            return True
        return self.headers.get("X-API-Key", "").strip() == API_TOKEN

    def _serve_file(self, rel_path: str):
        path = (WEB_ROOT / rel_path).resolve()
        if not str(path).startswith(str(WEB_ROOT)):
            self._set_headers(403, "text/plain; charset=utf-8")
            self.wfile.write(b"forbidden")
            return
        if not path.exists() or not path.is_file():
            self._set_headers(404, "text/plain; charset=utf-8")
            self.wfile.write(b"not found")
            return
        ctype, _ = mimetypes.guess_type(str(path))
        self._set_headers(200, ctype or "application/octet-stream")
        self.wfile.write(path.read_bytes())

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Serve dashboard static page from the same service/origin.
        # This avoids cross-origin/API-base issues for users.
        if path in ("/", "/index.html"):
            return self._serve_file("index.html")

        if not self._authorized():
            return self._json(401, {"ok": False, "error": "unauthorized"})

        if path == "/api/health":
            return self._json(200, {"ok": True, "ts": datetime.now(timezone.utc).isoformat()})
        if path == "/api/summary":
            return self._json(200, {"ok": True, "data": build_summary()})
        if path == "/api/dashboard/markets":
            try:
                q = parse_qs(parsed.query)
                city_key = _safe_city_key((q.get("city") or [""])[0])
                data = discover_city_markets(city_key)
                return self._json(200, {"ok": True, "data": data})
            except Exception as e:
                return self._json(400, {"ok": False, "error": str(e)})
        if path == "/api/dashboard/snapshot":
            try:
                q = parse_qs(parsed.query)
                city_key = _safe_city_key((q.get("city") or [""])[0])
                date_str = _safe_date((q.get("date") or [""])[0])
                data = build_dashboard_snapshot(city_key, date_str)
                return self._json(200, {"ok": True, "data": data})
            except Exception as e:
                return self._json(400, {"ok": False, "error": str(e)})
        if path == "/api/dashboard/logs":
            q = parse_qs(parsed.query)
            try:
                n = int((q.get("lines") or ["120"])[0])
            except Exception:
                n = 120
            n = max(20, min(n, 500))
            source, lines = _get_live_bot_logs(n)
            return self._json(
                200,
                {
                    "ok": True,
                    "data": {
                        "source": source,
                        "lines": lines,
                    },
                },
            )
        if path == "/api/config":
            config.refresh_runtime_overrides()
            return self._json(
                200,
                {
                    "ok": True,
                    "data": {
                        "overrides": config.get_runtime_overrides(),
                        "effective": config.get_runtime_effective(),
                    },
                },
            )
        return self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        if not self._authorized():
            return self._json(401, {"ok": False, "error": "unauthorized"})

        path = urlparse(self.path).path
        body = self._read_json()

        if path == "/api/bot/pause":
            ctrl = _load_control()
            paused = bool(body.get("paused", True))
            ctrl["paused"] = paused
            _save_control(ctrl)
            return self._json(200, {"ok": True, "data": {"paused": paused}})

        if path == "/api/bot/mode":
            mode = str(body.get("mode", "manual")).lower()
            if mode not in ("dryrun", "manual", "auto"):
                return self._json(400, {"ok": False, "error": "invalid_mode"})
            ctrl = _load_control()
            ctrl["mode"] = mode
            _save_control(ctrl)
            return self._json(200, {"ok": True, "data": {"mode": mode}})

        if path == "/api/config":
            try:
                config.write_runtime_overrides(body)
                return self._json(
                    200,
                    {
                        "ok": True,
                        "data": {
                            "overrides": config.get_runtime_overrides(),
                            "effective": config.get_runtime_effective(),
                        },
                    },
                )
            except Exception as e:
                return self._json(400, {"ok": False, "error": str(e)})

        return self._json(404, {"ok": False, "error": "not_found"})

    def log_message(self, format, *args):
        log.info("%s - %s", self.address_string(), format % args)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("Dashboard API listening on %s:%s", HOST, PORT)
    if API_TOKEN:
        log.info("Dashboard API auth: token required")
    else:
        log.warning("Dashboard API auth: token disabled (set DASHBOARD_API_TOKEN)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
