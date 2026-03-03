"""
NYC probabilistic model from multi-source daily-max forecasts.

Forward-test scope:
- Uses NYC ingestion source max forecasts (NBM/GEFS/ECMWF ENS)
- Builds a weighted mixture distribution over final daily max (°F)
- Applies observed-high floor from METAR (if available)
"""

from __future__ import annotations

import math

import config
from ensemble import map_to_brackets


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    s = max(0.05, float(sigma))
    z = (x - mu) / (s * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _normalize_degree_dist(raw_dist: dict) -> dict[int, float]:
    clean = {}
    for k, v in (raw_dist or {}).items():
        try:
            deg = int(k)
            p = float(v)
        except (TypeError, ValueError):
            continue
        if p <= 0:
            continue
        clean[deg] = clean.get(deg, 0.0) + p
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in sorted(clean.items())}


def _gaussian_degree_dist(mean_f: float, sigma_f: float) -> dict[int, float]:
    lo = int(math.floor(mean_f - 12.0))
    hi = int(math.ceil(mean_f + 12.0))
    out = {}
    for deg in range(lo, hi + 1):
        a = deg - 0.5
        b = deg + 0.5
        p = max(0.0, _norm_cdf(b, mean_f, sigma_f) - _norm_cdf(a, mean_f, sigma_f))
        if p > 0:
            out[deg] = p
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def _available_sources(ingestion: dict) -> list[dict]:
    rows = []
    src_map = (ingestion or {}).get("sources") or {}
    source_order = list(config.NYC_INGESTION_ACTIVE_SOURCES) or list(src_map.keys())
    for name in source_order:
        src = src_map.get(name) or {}
        if str(src.get("status", "")).lower() != "ok":
            continue
        m = _safe_float(src.get("max_temp_f"))
        if m is None:
            continue
        w = float(config.NYC_PROB_WEIGHTS.get(name, 0.0))
        if w <= 0:
            continue
        sigma = float(config.NYC_PROB_SIGMA_F.get(name, 1.5))
        dist_f = _normalize_degree_dist(src.get("max_temp_dist_f") or {})
        if not dist_f:
            dist_f = _gaussian_degree_dist(float(m), max(0.05, sigma))
        rows.append(
            {
                "name": str(name),
                "mean_f": float(m),
                "weight": w,
                "sigma_f": max(0.05, sigma),
                "dist_f": dist_f,
                "dist_member_count": int(src.get("dist_members", 0) or 0),
                "as_of_utc": src.get("as_of_utc"),
            }
        )
    return rows


def _normalize_weights(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    total = sum(float(r.get("weight", 0.0)) for r in rows)
    if total <= 0:
        equal = 1.0 / len(rows)
        for r in rows:
            r["weight"] = equal
        return rows
    for r in rows:
        r["weight"] = float(r.get("weight", 0.0)) / total
    return rows


def _degree_distribution(rows: list[dict]) -> dict[int, float]:
    if not rows:
        return {}
    out = {}
    for r in rows:
        w = float(r["weight"])
        src_dist = r.get("dist_f") or {}
        for deg, p in src_dist.items():
            out[int(deg)] = out.get(int(deg), 0.0) + (w * float(p))

    s = sum(out.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in sorted(out.items())}


def _apply_observed_floor(dist: dict[int, float], observed_high_f: float | None) -> tuple[dict[int, float], int | None]:
    if not dist:
        return {}, None
    if observed_high_f is None:
        return dist, None
    floor_deg = int(round(float(observed_high_f)))
    clipped = {k: v for k, v in dist.items() if int(k) >= floor_deg}
    s = sum(clipped.values())
    if s <= 0:
        return {}, floor_deg
    return {k: v / s for k, v in clipped.items()}, floor_deg


def _normalize_bracket_probs(bracket_probs: dict[str, float]) -> dict[str, float]:
    clean = {}
    for k, v in (bracket_probs or {}).items():
        fv = _safe_float(v)
        if fv is None or fv <= 0:
            continue
        clean[str(k)] = float(fv)
    s = sum(clean.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in clean.items()}


def build_nyc_probability_snapshot(
    *,
    ingestion: dict,
    brackets: list[dict],
) -> dict:
    """
    Build NYC bracket probabilities from multi-source max forecasts.
    """
    rows = _normalize_weights(_available_sources(ingestion))
    if not rows:
        return {
            "ok": False,
            "reason": "no_valid_sources",
            "source_count": 0,
            "sources": [],
            "degree_dist_f": {},
            "bracket_probs": {},
            "predicted_bracket": None,
            "expected_max_f": None,
            "observed_floor_f": None,
        }

    raw_dist = _degree_distribution(rows)
    metar = (ingestion or {}).get("metar") or {}
    observed_high_f = _safe_float(metar.get("day_high_market"))
    dist_f, floor_deg = _apply_observed_floor(raw_dist, observed_high_f)
    if not dist_f:
        return {
            "ok": False,
            "reason": "empty_distribution_after_floor",
            "source_count": len(rows),
            "sources": rows,
            "degree_dist_f": {},
            "bracket_probs": {},
            "predicted_bracket": None,
            "expected_max_f": None,
            "observed_floor_f": floor_deg,
        }

    bracket_probs = _normalize_bracket_probs(map_to_brackets(dist_f, brackets))
    predicted = max(bracket_probs, key=bracket_probs.get) if bracket_probs else None
    expected = sum(float(k) * float(v) for k, v in dist_f.items())

    return {
        "ok": True,
        "reason": "ok",
        "source_count": len(rows),
        "sources": rows,
        "degree_dist_f": {str(k): float(v) for k, v in sorted(dist_f.items())},
        "bracket_probs": bracket_probs,
        "predicted_bracket": predicted,
        "expected_max_f": round(expected, 3),
        "observed_floor_f": floor_deg,
    }
