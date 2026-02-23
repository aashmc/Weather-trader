"""
Weather Trader Bot — Ensemble Forecasting & Bias Correction
Fetches multi-model ensemble forecasts from Open-Meteo,
applies Monte Carlo bias correction using METAR-calibrated parameters.
"""

import math
import logging
import httpx
from config import ENSEMBLE_API, FORECAST_API, MC_SAMPLES, MC_SEED

log = logging.getLogger("ensemble")


def c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def ens_to_market(temp_c: float, unit: str) -> float:
    """Convert ensemble temp (always °C) to market units."""
    return c_to_f(temp_c) if unit == "F" else temp_c


def mulberry32(seed: int):
    """Deterministic PRNG matching the dashboard."""
    a = seed

    def next_val():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t ^ (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return next_val


def _model_tag(member_key: str) -> str:
    """Extract model name from Open-Meteo member key."""
    key = member_key.lower()
    if "ecmwf_ifs" in key:
        return "ifs"
    if "ecmwf_aifs" in key:
        return "aifs"
    if "icon" in key:
        return "ico"
    if "gem" in key:
        return "gem"
    if "ukmo" in key:
        return "ukmo"
    if "bom" in key:
        return "bom"
    if "gfs" in key or "ncep" in key:
        return "gfs"
    return "x"


def _model_family(tag: str) -> str:
    """Collapse related models into dependence-adjusted families."""
    if tag in ("ifs", "aifs"):
        return "ecmwf"
    if tag == "ico":
        return "icon"
    if tag in ("gem", "ukmo", "gfs", "kma"):
        return tag
    return tag


async def fetch_ensemble(city: dict, date_str: str) -> dict:
    """
    Fetch ensemble forecasts for a city and date.
    Returns:
        {
            "max_temps": [float],       # all member daily max temps in market units
            "mean": float,              # ensemble mean
            "count": int,               # total members
            "model_votes": {int_deg: int},  # how many distinct models predict each degree
            "family_votes": {int_deg: int}, # how many distinct model families predict each degree
            "raw_dist": {int_deg: float},   # raw probability distribution by degree
        }
    """
    models_str = ",".join(city["ensemble_models"])
    url = (
        f"{ENSEMBLE_API}?latitude={city['lat']}&longitude={city['lon']}"
        f"&hourly=temperature_2m&models={models_str}"
        f"&forecast_days=7&timezone={city['tz']}"
    )

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    member_keys = [k for k in hourly if k.startswith("temperature_2m_member")]

    if not member_keys:
        raise ValueError(f"No ensemble members returned for {city['name']}")

    # Find indices for target date
    target_idx = [i for i, t in enumerate(times) if t and t.startswith(date_str)]
    if not target_idx:
        raise ValueError(f"No data for {date_str} in ensemble response")

    max_temps = []
    model_votes_per_deg = {}   # {degree: {model_tag: True}}
    family_votes_per_deg = {}  # {degree: {family_tag: True}}

    for mk in member_keys:
        vals = hourly[mk]
        mx = max(
            (vals[i] for i in target_idx if vals[i] is not None),
            default=None,
        )
        if mx is None:
            continue

        mkt_temp = ens_to_market(mx, city["unit"])
        max_temps.append(mkt_temp)

        tag = _model_tag(mk)
        fam = _model_family(tag)
        rd = round(mkt_temp)
        model_votes_per_deg.setdefault(rd, set()).add(tag)
        family_votes_per_deg.setdefault(rd, set()).add(fam)

    # Synthetic model (e.g., KMA for Seoul)
    if city.get("synthetic_model"):
        try:
            syn_temp = await _fetch_synthetic(city, date_str)
            if syn_temp is not None:
                mkt_temp = ens_to_market(syn_temp, city["unit"])
                # Add 10 synthetic members with tiny noise
                import random

                rng = random.Random(42)
                for _ in range(10):
                    jitter = (rng.random() - 0.5) * 0.6  # ±0.3°
                    t = mkt_temp + jitter
                    max_temps.append(t)
                    rd = round(t)
                    model_votes_per_deg.setdefault(rd, set()).add("kma")
                    family_votes_per_deg.setdefault(rd, set()).add("kma")
                log.info(
                    f"{city['name']}: KMA synthetic {mkt_temp:.1f} (10 members added)"
                )
        except Exception as e:
            log.warning(f"{city['name']}: KMA synthetic failed: {e}")

    if not max_temps:
        raise ValueError(f"No valid max temps for {city['name']} on {date_str}")

    # Convert model_votes_per_deg sets to counts
    model_votes = {deg: len(models) for deg, models in model_votes_per_deg.items()}
    family_votes = {deg: len(fams) for deg, fams in family_votes_per_deg.items()}

    # Build raw distribution
    mean_temp = sum(max_temps) / len(max_temps)
    raw_dist = _build_distribution(max_temps)

    log.info(
        f"{city['name']}: {len(max_temps)} members, "
        f"mean {mean_temp:.1f}{'°F' if city['unit'] == 'F' else '°C'}"
    )

    return {
        "max_temps": max_temps,
        "mean": mean_temp,
        "count": len(max_temps),
        "model_votes": model_votes,
        "family_votes": family_votes,
        "raw_dist": raw_dist,
    }


async def _fetch_synthetic(city: dict, date_str: str) -> float | None:
    """Fetch deterministic forecast for synthetic model (e.g., KMA)."""
    url = (
        f"{FORECAST_API}?latitude={city['lat']}&longitude={city['lon']}"
        f"&daily=temperature_2m_max&models={city['synthetic_model']}"
        f"&forecast_days=7&timezone={city['tz']}"
    )
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    dates = data.get("daily", {}).get("time", [])
    maxes = data.get("daily", {}).get("temperature_2m_max", [])

    if date_str in dates:
        idx = dates.index(date_str)
        return maxes[idx]
    return None


def _build_distribution(max_temps: list[float]) -> dict[int, float]:
    """Build probability distribution from ensemble max temps."""
    counts = {}
    for t in max_temps:
        rd = round(t)
        counts[rd] = counts.get(rd, 0) + 1
    total = len(max_temps)
    return {deg: cnt / total for deg, cnt in counts.items()}


def bias_correct(raw_dist: dict[int, float], bias_mean: float, bias_sd: float) -> dict[int, float]:
    """
    Monte Carlo bias correction.
    Shifts raw ensemble distribution by METAR-calibrated bias.
    Returns corrected distribution by degree.
    """
    if bias_sd == 0:
        return dict(raw_dist)

    rng = mulberry32(MC_SEED)

    # Determine range
    all_degs = sorted(raw_dist.keys())
    if not all_degs:
        return {}
    range_min = min(all_degs) - 5
    range_max = max(all_degs) + 5

    counts = {i: 0 for i in range(range_min, range_max + 1)}

    # Cumulative distribution for sampling
    cum_dist = []
    cum = 0
    for deg in sorted(raw_dist.keys()):
        cum += raw_dist[deg]
        cum_dist.append((cum, deg))
    med_deg = all_degs[len(all_degs) // 2]

    for _ in range(MC_SAMPLES):
        # Sample from raw distribution
        r = rng()
        base_deg = med_deg
        for cum_val, deg in cum_dist:
            if r <= cum_val:
                base_deg = deg
                break

        # Add uniform sub-degree noise
        base_t = base_deg + (rng() - 0.5)

        # Box-Muller for normal bias
        u1 = rng()
        u2 = rng()
        z = math.sqrt(-2 * math.log(u1 + 1e-10)) * math.cos(2 * math.pi * u2)
        metar_t = base_t + bias_mean + bias_sd * z
        rounded = round(metar_t)

        if range_min <= rounded <= range_max:
            counts[rounded] += 1

    total = sum(counts.values())
    if total == 0:
        return {}

    return {deg: cnt / total for deg, cnt in counts.items() if cnt > 0}


def map_to_brackets(dist: dict[int, float], brackets: list[dict]) -> dict[str, float]:
    """
    Map a degree-keyed distribution to bracket-label-keyed probabilities.
    brackets: list of {"label": str, "type": str, "low": int, "high": int, "val": int}
    """
    result = {}
    for b in brackets:
        label = b["label"]
        btype = b["type"]
        prob = 0.0

        if btype == "below":
            prob = sum(v for k, v in dist.items() if k <= b["high"])
        elif btype == "above":
            prob = sum(v for k, v in dist.items() if k >= b["low"])
        elif btype == "single":
            prob = dist.get(b["val"], 0.0)
        elif btype == "range":
            prob = sum(v for k, v in dist.items() if b["low"] <= k <= b["high"])

        result[label] = prob

    return result
