"""
Weather Trader Bot — Dashboard Control API
Provides read/write endpoints for live dashboard state and bot controls.
"""

import asyncio
import json
import logging
import mimetypes
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import config
from market import fetch_wallet_balance
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
        path = urlparse(self.path).path

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
