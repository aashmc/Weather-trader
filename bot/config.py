"""
Weather Trader Bot — Configuration
All city configs, trading thresholds, and bias correction parameters.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════
# SECRETS (from .env)
# ══════════════════════════════════════════════════════
POLYGON_PRIVATE_KEY = os.getenv("POLYGON_PRIVATE_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GOOGLE_SHEET_WEBHOOK = os.getenv("GOOGLE_SHEET_WEBHOOK", "")

# ══════════════════════════════════════════════════════
# TRADING PARAMETERS
# ══════════════════════════════════════════════════════
BANKROLL_FALLBACK = 0.0                # If wallet query fails, don't trade
KELLY_FRACTION = 1.0           # Full Kelly (before concentration scaling)
KELLY_CAP = 0.75               # Max Kelly multiplier after concentration scaling

# These are ratios — actual $ amounts computed dynamically from wallet balance
MAX_BET_RATIO = 0.10           # 10% of bankroll per bracket
MAX_EXPOSURE_RATIO = 0.40      # 40% of bankroll total exposure
DAILY_LOSS_RATIO = 0.60        # 60% of bankroll — kill switch

# Live values — updated each cycle by update_bankroll()
BANKROLL = BANKROLL_FALLBACK
MAX_BET_PER_BRACKET = round(BANKROLL * MAX_BET_RATIO, 2)
MAX_TOTAL_EXPOSURE = round(BANKROLL * MAX_EXPOSURE_RATIO, 2)
DAILY_LOSS_LIMIT = round(BANKROLL * DAILY_LOSS_RATIO, 2)


def update_bankroll(balance: float):
    """Update all bankroll-derived limits from live wallet balance."""
    global BANKROLL, MAX_BET_PER_BRACKET, MAX_TOTAL_EXPOSURE, DAILY_LOSS_LIMIT
    if balance < 0:
        return  # Keep previous values on fetch failure
    if balance == 0:
        BANKROLL = 0.0
        MAX_BET_PER_BRACKET = 0.0
        MAX_TOTAL_EXPOSURE = 0.0
        DAILY_LOSS_LIMIT = 0.0
        return
    BANKROLL = round(balance, 2)
    MAX_BET_PER_BRACKET = round(BANKROLL * MAX_BET_RATIO, 2)
    MAX_TOTAL_EXPOSURE = round(BANKROLL * MAX_EXPOSURE_RATIO, 2)
    DAILY_LOSS_LIMIT = round(BANKROLL * DAILY_LOSS_RATIO, 2)

# Global filters (applied to all cities)
MIN_ASK_DEPTH = 20             # 20 contracts minimum on ask side
MAX_ASK_PRICE = 0.50           # Don't buy >50¢
ORDER_TIMEOUT_SECONDS = 300    # Cancel unfilled limit orders after 5 min

CYCLE_INTERVAL_SECONDS = 1800  # 30 minutes between main cycles
TELEGRAM_POLL_SECONDS = 60     # Fast polling for button presses

# ══════════════════════════════════════════════════════
# MARKET MATURITY THRESHOLDS
# ══════════════════════════════════════════════════════
MATURITY_MIN_FAV_PRICE = 0.25  # At least one bracket must be ≥25¢
MATURITY_MAX_FAV_SPREAD = 0.03 # Favorite spread must be ≤3¢
MATURITY_HARD_MAX_FAV_SPREAD = 0.12  # Absolute no-trade if favorite spread exceeds 12¢
MATURITY_MIN_LIQUID_BRACKETS = 3  # At least 3 brackets with depth ≥10

# ══════════════════════════════════════════════════════
# CONCENTRATION-BASED KELLY SCALING
# ══════════════════════════════════════════════════════
CONCENTRATION_MIN = 0.50       # <50% = no trade
CONCENTRATION_TIERS = [
    (0.70, 0.75),  # ≥70% → 0.75x Kelly
    (0.60, 0.60),  # 60-69% → 0.60x Kelly
    (0.50, 0.50),  # 50-59% → 0.50x Kelly
]

# ══════════════════════════════════════════════════════
# MODEL VS MARKET DIVERGENCE
# ══════════════════════════════════════════════════════
MAX_DIVERGENCE_BRACKETS = 2    # If model top and market fav 3+ apart → warn & skip

# ══════════════════════════════════════════════════════
# EXECUTION-AWARE EDGE
# ══════════════════════════════════════════════════════
EXECUTION_ADJUSTED_EDGE_ENABLED = True
EXECUTION_FEE_RATE = 0.0            # Per-contract taker fee rate (default weather fee-free)
EXECUTION_GAS_USD_PER_ORDER = 0.0   # Gas cost per order in USD (default 0 for relayed flow)
EXECUTION_SPREAD_REF = 0.04         # 4c spread reference for fill/queue penalties
EXECUTION_QUEUE_PENALTY = 0.50      # Penalize continuation fill when spread is wide
EXECUTION_MIN_FILL = 0.05           # Floor on expected fill fraction
EXECUTION_MAX_FILL = 0.98           # Ceiling on expected fill fraction

# ══════════════════════════════════════════════════════
# ROLLING CALIBRATION (ITEM #3)
# ══════════════════════════════════════════════════════
ROLLING_CALIBRATION_ENABLED = True
ROLLING_CAL_WINDOW = 45            # Keep last N resolved errors per lead bucket
ROLLING_CAL_MIN_SAMPLES = 20       # Minimum resolved samples before override
ROLLING_CAL_BLEND = 0.35           # Blend weight for rolling bias vs static bias
ROLLING_CAL_SD_FLOOR = 0.10        # Prevent collapse to near-zero variance

# Lead-time buckets by hours until market-date local midnight
LEAD_BUCKET_FAR_HOURS = 36
LEAD_BUCKET_MID_HOURS = 12

# ══════════════════════════════════════════════════════
# MODEL QUALITY GATES (ITEM #5)
# ══════════════════════════════════════════════════════
# Default off to avoid unapproved PnL-impacting threshold changes.
QUALITY_GATE_ENABLED = False
QUALITY_WINDOW = 30
QUALITY_MIN_SAMPLES = 20
QUALITY_MAX_BRIER = 0.95
QUALITY_MIN_WINNER_PROB = 0.12

# ══════════════════════════════════════════════════════
# LATE-SESSION ENTRY GUARD
# ══════════════════════════════════════════════════════
LATE_GUARD_ENABLED = True
LATE_GUARD_HISTORY_DAYS = 365
LATE_GUARD_REFRESH_HOURS = 24
LATE_GUARD_CUTOFF_QUANTILE = 0.80
LATE_GUARD_MIN_MONTH_SAMPLES = 20
LATE_GUARD_MIN_GLOBAL_SAMPLES = 90
LATE_GUARD_MIN_CUTOFF_HOUR = 10
LATE_GUARD_MAX_CUTOFF_HOUR = 20
LATE_GUARD_DEFAULT_CUTOFF_HOUR = 15
LATE_GUARD_FALLBACK_CUTOFF_HOURS = {
    "london": 15,
    "seoul": 15,
    "nyc": 16,
    "seattle": 16,
}
LATE_GUARD_FREEZE_HOURS_AFTER_CUTOFF = 6
LATE_GUARD_AFTER_CUTOFF_FAVORITE_ONLY = True
LATE_GUARD_AFTER_CUTOFF_MIN_EDGE = 0.08
LATE_GUARD_AFTER_CUTOFF_FAMILY_BUMP = 1
LATE_GUARD_FREEZE_ALL_NEW_ENTRIES = True

# ══════════════════════════════════════════════════════
# BOT MODE
# ══════════════════════════════════════════════════════
DRY_RUN = True
BOT_PAUSED = False

# ══════════════════════════════════════════════════════
# MONTE CARLO BIAS CORRECTION
# ══════════════════════════════════════════════════════
MC_SAMPLES = 50000
MC_SEED = 42

# ══════════════════════════════════════════════════════
# CITY CONFIGURATIONS
# ══════════════════════════════════════════════════════
CITIES = {
    "london": {
        "name": "London",
        "icao": "EGLC",
        "lat": 51.5053,
        "lon": 0.0553,
        "unit": "C",
        "tz": "Europe/London",
        "poly_city": "london",
        "ensemble_models": [
            "icon_seamless_eps",
            "ecmwf_ifs025_ensemble",
            "ecmwf_aifs025_ensemble",
            "gem_global_ensemble",
            "ukmo_global_ensemble_20km",
        ],
        "synthetic_model": None,
        "bias_mean": 0.17,
        "bias_sd": 0.66,
        "bias_note": "45-day METAR backtest Jan-Feb 2026",
        "min_models": 3,    # 3/5 models must agree
        "min_edge": 0.05,   # 5pt minimum edge
        # Spread policy: keep strict for London by default.
        "fav_soft_spread": 0.03,
        "fav_relaxed_spread": 0.03,
        "fav_relaxed_min_edge": 0.05,
        "fav_relaxed_min_depth": 20,
    },
    "seoul": {
        "name": "Seoul",
        "icao": "RKSI",
        "lat": 37.4602,
        "lon": 126.4407,
        "unit": "C",
        "tz": "Asia/Seoul",
        "poly_city": "seoul",
        "ensemble_models": [
            "ecmwf_ifs025_ensemble",
            "ecmwf_aifs025_ensemble",
            "gem_global_ensemble",
        ],
        "synthetic_model": "kma_seamless",
        "bias_mean": 0.69,
        "bias_sd": 0.86,
        "bias_note": "45-day METAR backtest Jan-Feb 2026",
        "min_models": 3,    # 3/4 models must agree
        "min_edge": 0.05,
        # Seoul: allow favorite entries in moderate spreads if edge/depth are strong.
        "fav_soft_spread": 0.03,
        "fav_relaxed_spread": 0.08,
        "fav_relaxed_min_edge": 0.08,
        "fav_relaxed_min_depth": 30,
    },
    "nyc": {
        "name": "NYC",
        "icao": "KLGA",
        "lat": 40.7772,
        "lon": -73.8726,
        "unit": "F",
        "tz": "America/New_York",
        "poly_city": "nyc",
        "ensemble_models": [
            "icon_seamless_eps",
            "ecmwf_ifs025_ensemble",
        ],
        "synthetic_model": None,
        "bias_mean": 1.79,
        "bias_sd": 2.49,
        "bias_note": "45-day METAR backtest Jan-Feb 2026",
        "min_models": 2,    # 2/2 — both must agree
        "min_edge": 0.08,   # 8pt higher threshold
        "fav_soft_spread": 0.03,
        "fav_relaxed_spread": 0.03,
        "fav_relaxed_min_edge": 0.08,
        "fav_relaxed_min_depth": 20,
    },
    "seattle": {
        "name": "Seattle",
        "icao": "KSEA",
        "lat": 47.4502,
        "lon": -122.3088,
        "unit": "F",
        "tz": "America/Los_Angeles",
        "poly_city": "seattle",
        "ensemble_models": [
            "icon_seamless_eps",
            "ecmwf_ifs025_ensemble",
            "gem_global_ensemble",
        ],
        "synthetic_model": None,
        "bias_mean": 1.66,
        "bias_sd": 1.93,
        "bias_note": "45-day METAR backtest Jan-Feb 2026",
        "min_models": 2,    # 2/3 models must agree
        "min_edge": 0.05,
        "fav_soft_spread": 0.03,
        "fav_relaxed_spread": 0.03,
        "fav_relaxed_min_edge": 0.05,
        "fav_relaxed_min_depth": 20,
    },
}

# Total model count per city
for city_cfg in CITIES.values():
    city_cfg["total_models"] = len(city_cfg["ensemble_models"]) + (
        1 if city_cfg["synthetic_model"] else 0
    )


def _family_from_model(model_name: str) -> str:
    name = model_name.lower()
    if "ecmwf_ifs" in name or "ecmwf_aifs" in name:
        return "ecmwf"
    if "icon" in name:
        return "icon"
    if "gem" in name:
        return "gem"
    if "ukmo" in name:
        return "ukmo"
    if "gfs" in name or "ncep" in name:
        return "gfs"
    if "kma" in name:
        return "kma"
    return name


# Family-aware agreement defaults (ITEM #2)
for city_cfg in CITIES.values():
    fams = {_family_from_model(m) for m in city_cfg["ensemble_models"]}
    if city_cfg.get("synthetic_model"):
        fams.add(_family_from_model(city_cfg["synthetic_model"]))
    city_cfg["total_families"] = len(fams)
    city_cfg["min_families"] = min(
        city_cfg.get("min_families", city_cfg["min_models"]),
        city_cfg["total_families"],
    )

# ══════════════════════════════════════════════════════
# RUNTIME OVERRIDES (editable from dashboard API)
# ══════════════════════════════════════════════════════
RUNTIME_OVERRIDES_FILE = Path(
    os.getenv("RUNTIME_OVERRIDES_FILE", "runtime_overrides.json")
)

_BASE_GLOBALS = {
    "MAX_BET_RATIO": MAX_BET_RATIO,
    "MAX_EXPOSURE_RATIO": MAX_EXPOSURE_RATIO,
    "DAILY_LOSS_RATIO": DAILY_LOSS_RATIO,
    "MIN_ASK_DEPTH": MIN_ASK_DEPTH,
    "MAX_ASK_PRICE": MAX_ASK_PRICE,
    "ORDER_TIMEOUT_SECONDS": ORDER_TIMEOUT_SECONDS,
}
_BASE_CITY_THRESHOLDS = {
    city_key: {
        "min_models": cfg.get("min_models"),
        "min_edge": cfg.get("min_edge"),
        "min_families": cfg.get("min_families"),
    }
    for city_key, cfg in CITIES.items()
}
_LAST_OVERRIDES_MTIME = None


def _safe_float(v, fallback):
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback


def _safe_int(v, fallback):
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _read_overrides_file() -> dict:
    if not RUNTIME_OVERRIDES_FILE.exists():
        return {}
    try:
        return json.loads(RUNTIME_OVERRIDES_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def _apply_defaults():
    global MAX_BET_RATIO, MAX_EXPOSURE_RATIO, DAILY_LOSS_RATIO
    global MIN_ASK_DEPTH, MAX_ASK_PRICE, ORDER_TIMEOUT_SECONDS
    MAX_BET_RATIO = _BASE_GLOBALS["MAX_BET_RATIO"]
    MAX_EXPOSURE_RATIO = _BASE_GLOBALS["MAX_EXPOSURE_RATIO"]
    DAILY_LOSS_RATIO = _BASE_GLOBALS["DAILY_LOSS_RATIO"]
    MIN_ASK_DEPTH = _BASE_GLOBALS["MIN_ASK_DEPTH"]
    MAX_ASK_PRICE = _BASE_GLOBALS["MAX_ASK_PRICE"]
    ORDER_TIMEOUT_SECONDS = _BASE_GLOBALS["ORDER_TIMEOUT_SECONDS"]

    for city_key, base in _BASE_CITY_THRESHOLDS.items():
        city_cfg = CITIES.get(city_key)
        if not city_cfg:
            continue
        city_cfg["min_models"] = base["min_models"]
        city_cfg["min_edge"] = base["min_edge"]
        city_cfg["min_families"] = min(base["min_families"], city_cfg["total_families"])


def _apply_overrides(payload: dict):
    """
    Apply validated runtime overrides on top of defaults.
    """
    global MAX_BET_RATIO, MAX_EXPOSURE_RATIO, DAILY_LOSS_RATIO
    global MIN_ASK_DEPTH, MAX_ASK_PRICE, ORDER_TIMEOUT_SECONDS

    _apply_defaults()

    if not isinstance(payload, dict):
        return

    MAX_BET_RATIO = max(0.0, min(1.0, _safe_float(payload.get("max_bet_ratio"), MAX_BET_RATIO)))
    MAX_EXPOSURE_RATIO = max(0.0, min(1.0, _safe_float(payload.get("max_exposure_ratio"), MAX_EXPOSURE_RATIO)))
    DAILY_LOSS_RATIO = max(0.0, min(1.0, _safe_float(payload.get("daily_loss_ratio"), DAILY_LOSS_RATIO)))
    MIN_ASK_DEPTH = max(0, _safe_int(payload.get("min_ask_depth"), MIN_ASK_DEPTH))
    MAX_ASK_PRICE = max(0.01, min(0.99, _safe_float(payload.get("max_ask_price"), MAX_ASK_PRICE)))
    ORDER_TIMEOUT_SECONDS = max(30, _safe_int(payload.get("order_timeout_seconds"), ORDER_TIMEOUT_SECONDS))

    city_overrides = payload.get("city_overrides", {})
    if isinstance(city_overrides, dict):
        for city_key, ov in city_overrides.items():
            city_cfg = CITIES.get(city_key)
            if not city_cfg or not isinstance(ov, dict):
                continue

            if "min_models" in ov:
                city_cfg["min_models"] = max(1, _safe_int(ov.get("min_models"), city_cfg["min_models"]))
            if "min_edge" in ov:
                city_cfg["min_edge"] = max(0.0, min(0.5, _safe_float(ov.get("min_edge"), city_cfg["min_edge"])))
            if "min_families" in ov:
                city_cfg["min_families"] = max(1, _safe_int(ov.get("min_families"), city_cfg["min_families"]))

            city_cfg["min_families"] = min(city_cfg["min_families"], city_cfg["total_families"])
            city_cfg["min_models"] = min(city_cfg["min_models"], city_cfg["total_models"])

    # Recompute dollar limits from current bankroll after ratio changes.
    update_bankroll(BANKROLL)


def refresh_runtime_overrides(force: bool = False) -> dict:
    """
    Reload overrides file when changed. Returns active override payload.
    """
    global _LAST_OVERRIDES_MTIME
    mtime = None
    if RUNTIME_OVERRIDES_FILE.exists():
        try:
            mtime = RUNTIME_OVERRIDES_FILE.stat().st_mtime
        except OSError:
            mtime = None

    if not force and mtime == _LAST_OVERRIDES_MTIME:
        return _read_overrides_file()

    payload = _read_overrides_file()
    _apply_overrides(payload)
    _LAST_OVERRIDES_MTIME = mtime
    return payload


def write_runtime_overrides(payload: dict):
    """
    Persist overrides and apply immediately.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    RUNTIME_OVERRIDES_FILE.write_text(json.dumps(payload, indent=2))
    refresh_runtime_overrides(force=True)


def get_runtime_overrides() -> dict:
    return deepcopy(_read_overrides_file())


def get_runtime_effective() -> dict:
    return {
        "max_bet_ratio": MAX_BET_RATIO,
        "max_exposure_ratio": MAX_EXPOSURE_RATIO,
        "daily_loss_ratio": DAILY_LOSS_RATIO,
        "min_ask_depth": MIN_ASK_DEPTH,
        "max_ask_price": MAX_ASK_PRICE,
        "order_timeout_seconds": ORDER_TIMEOUT_SECONDS,
        "city_thresholds": {
            city_key: {
                "min_models": cfg.get("min_models"),
                "min_edge": cfg.get("min_edge"),
                "min_families": cfg.get("min_families"),
            }
            for city_key, cfg in CITIES.items()
        },
    }


# Apply persisted runtime overrides on import.
refresh_runtime_overrides(force=True)

# ══════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════
ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
METAR_API = "https://aviationweather.gov/api/data/metar"
GAMMA_API = "https://gamma-api.polymarket.com/events"
CLOB_API = "https://clob.polymarket.com"
POLYGON_GAS_API = "https://gasstation.polygon.technology/v2"
COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"

POLYMARKET_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet
