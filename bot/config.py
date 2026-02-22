"""
Weather Trader Bot — Configuration
All city configs, trading thresholds, and bias correction parameters.
"""

import os
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
BANKROLL = 9.0
KELLY_FRACTION = 0.33          # ⅓ Kelly
MAX_BET_PER_BRACKET = 0.90     # 10% of bankroll
MAX_TOTAL_EXPOSURE = 3.60      # 40% of bankroll
DAILY_LOSS_LIMIT = 5.40        # 60% of bankroll — kill switch

MIN_TRUE_EDGE = 0.05           # 5pt minimum edge
MIN_MODEL_AGREEMENT = 2        # 2+ models must agree
MIN_ASK_DEPTH = 20             # 20 contracts minimum on ask side
MAX_ASK_PRICE = 0.50           # Don't buy >50¢
ORDER_TIMEOUT_SECONDS = 300    # Cancel unfilled limit orders after 5 min

CYCLE_INTERVAL_SECONDS = 1800  # 30 minutes between cycles

# ══════════════════════════════════════════════════════
# BOT MODE
# ══════════════════════════════════════════════════════
DRY_RUN = True  # True = log everything but don't place orders. False = live trading.
BOT_PAUSED = False  # Can be toggled via Telegram commands

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
    },
}

# Total model count per city (for model agreement denominator)
for city_cfg in CITIES.values():
    city_cfg["total_models"] = len(city_cfg["ensemble_models"]) + (
        1 if city_cfg["synthetic_model"] else 0
    )

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

# Polymarket CLOB
POLYMARKET_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet
