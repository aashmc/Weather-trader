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
BANKROLL_FALLBACK = 9.0                # Fallback if wallet query fails
MIN_ORDER_SIZE = 1.0                   # Polymarket minimum order $1
KELLY_FRACTION = 0.33          # ⅓ Kelly (before concentration scaling)
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
    if balance <= 0:
        return  # Keep previous values
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
    },
}

# Total model count per city
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

POLYMARKET_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet
