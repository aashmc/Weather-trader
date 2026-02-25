"""
Weather Trader Bot — Strategy v5
Full signal pipeline:
  1. Market maturity check
  2. Identify market favorite
  3. Candidates = favorite ±1
  4. Adaptive edge hurdle (1c base + spread/depth/time adders)
  5. Concentration check + Kelly scaling
  6. Kelly scaling by concentration (capped at 75%)
  7. Model vs market divergence warning (non-blocking)
"""

import logging
import config
from config import (
    KELLY_FRACTION, KELLY_CAP,
    MIN_ASK_DEPTH, MAX_ASK_PRICE,
    MATURITY_MIN_FAV_PRICE, MATURITY_MAX_FAV_SPREAD, MATURITY_HARD_MAX_FAV_SPREAD, MATURITY_MIN_LIQUID_BRACKETS,
    CONCENTRATION_MIN, CONCENTRATION_TIERS,
    MAX_DIVERGENCE_BRACKETS,
)
from market import bracket_degree

log = logging.getLogger("strategy")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _top_ask_depth(book: dict, ask: float) -> float:
    """Depth available immediately at the current best ask."""
    if not book:
        return 0.0
    depth = 0.0
    for level in book.get("asks", []):
        try:
            px = float(level.get("price", 0))
            sz = float(level.get("size", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        if abs(px - ask) <= 1e-9:
            depth += max(0.0, sz)
        elif px > ask:
            break
    return depth


def _estimate_execution(book: dict | None, ask: float, contracts: int) -> dict:
    """
    Estimate expected fill quality over the 5-minute order timeout window.
    """
    if not book or contracts <= 0:
        return {
            "top_ask_depth": 0.0,
            "immediate_fill_fraction": 1.0,
            "continuation_fill_prob": 1.0,
            "fill_fraction": 1.0,
            "slippage_penalty": 0.0,
        }

    spread = max(0.0, float(book.get("spread", 0.0)))
    ask_depth = max(0.0, float(book.get("ask_depth", 0.0)))
    top_ask_depth = _top_ask_depth(book, ask)

    immediate = _clip(top_ask_depth / max(1.0, float(contracts)), 0.0, 1.0)
    spread_ref = max(0.0001, config.EXECUTION_SPREAD_REF)
    spread_score = _clip(1.0 - spread / spread_ref, 0.0, 1.0)
    depth_score = _clip(ask_depth / max(1.0, float(contracts)), 0.0, 1.0)

    queue_penalty = _clip((spread / spread_ref) * config.EXECUTION_QUEUE_PENALTY, 0.0, 0.8)
    continuation = _clip(
        0.10 + 0.55 * spread_score + 0.35 * depth_score - queue_penalty,
        config.EXECUTION_MIN_FILL,
        config.EXECUTION_MAX_FILL,
    )

    fill_fraction = immediate + (1.0 - immediate) * continuation
    slippage_penalty = spread * (1.0 - immediate) * 0.50

    return {
        "top_ask_depth": top_ask_depth,
        "immediate_fill_fraction": immediate,
        "continuation_fill_prob": continuation,
        "fill_fraction": _clip(fill_fraction, config.EXECUTION_MIN_FILL, 1.0),
        "slippage_penalty": max(0.0, slippage_penalty),
    }


def _required_ask_depth(city_config: dict, contracts: int) -> float:
    """
    Dynamic depth requirement:
    - floor by city
    - scale with expected contracts (capped) for thin books
    """
    floor = float(city_config.get("min_ask_depth", MIN_ASK_DEPTH))
    ratio = float(city_config.get("min_depth_contract_ratio", 0.0))
    cap = float(city_config.get("min_depth_cap", floor))
    if contracts <= 0 or ratio <= 0:
        return floor
    scaled = min(cap, contracts * ratio)
    return max(floor, scaled)


def _agreement_multiplier(
    model_votes: int,
    family_votes: int,
    min_models: int,
    min_families: int,
) -> tuple[float, str | None]:
    """
    Scale bet-size by model/family agreement instead of hard blocking.
    Extreme disagreement (zero support) still blocks.
    """
    if model_votes <= 0 or family_votes <= 0:
        return 0.0, "extreme disagreement"

    model_ratio = model_votes / max(1, min_models)
    family_ratio = family_votes / max(1, min_families)
    support = min(model_ratio, family_ratio)
    floor = _clip(config.AGREEMENT_SCALE_FLOOR, 0.0, 1.0)
    min_mult = _clip(config.AGREEMENT_SCALE_MIN, 0.0, 1.0)
    mult = floor + (1.0 - floor) * _clip(support, 0.0, 1.0)
    return _clip(mult, min_mult, 1.0), None


def _adaptive_edge_threshold(
    city_config: dict,
    spread: float,
    ask_depth: float,
    required_depth: float,
    late_applies: bool,
    late_phase: str,
    is_favorite: bool,
    guard_edge_bump: float = 0.0,
) -> float:
    """
    Required edge after costs:
      base 1c floor + spread adder + depth adder + late-session adder + live guardrail bump.
    """
    base = float(city_config.get("min_edge", config.ADAPTIVE_EDGE_BASE))
    soft_spread = float(city_config.get("fav_soft_spread", MATURITY_MAX_FAV_SPREAD))
    spread_slope = float(city_config.get("adaptive_spread_edge_slope", config.ADAPTIVE_SPREAD_EDGE_SLOPE))
    depth_edge_max = float(city_config.get("adaptive_depth_edge_max", config.ADAPTIVE_DEPTH_EDGE_MAX))

    spread_excess = max(0.0, spread - soft_spread)
    spread_adder = max(0.0, spread_excess * spread_slope)

    depth_ratio = 0.0
    if required_depth > 0:
        depth_ratio = _clip((required_depth - ask_depth) / required_depth, 0.0, 1.0)
    depth_adder = depth_ratio * max(0.0, depth_edge_max)

    late_adder = 0.0
    if late_applies and late_phase in ("after_cutoff", "freeze"):
        late_adder += max(0.0, config.ADAPTIVE_LATE_EDGE_ADDER)
        if not is_favorite:
            late_adder += max(0.0, config.ADAPTIVE_NONFAV_LATE_EDGE_ADDER)

    return _clip(base + spread_adder + depth_adder + late_adder + max(0.0, guard_edge_bump), 0.0, 0.5)


# ══════════════════════════════════════════════════════
# STEP 1: MARKET MATURITY CHECK
# ══════════════════════════════════════════════════════

def check_market_maturity(brackets: list[dict], books: dict, city_config: dict | None = None) -> tuple[bool, str]:
    """
    Check if the market is mature enough to trade.
    Returns (mature, reason).
    """
    if not books:
        return False, "no order books"

    # Keep only the strict hard-cap maturity safety gate.
    hard_cap = MATURITY_HARD_MAX_FAV_SPREAD
    if city_config:
        hard_cap = float(city_config.get("maturity_hard_spread", hard_cap))
    fav_label = max(books, key=lambda k: books[k]["ba"] if books[k] else 0)
    fav_book = books[fav_label]
    if fav_book and fav_book.get("spread", 1) > hard_cap:
        return False, f"favorite spread {fav_book['spread']*100:.1f}¢ > hard cap {hard_cap*100:.0f}¢"

    return True, "mature"


# ══════════════════════════════════════════════════════
# STEP 2: IDENTIFY MARKET FAVORITE
# ══════════════════════════════════════════════════════

def find_market_favorite(brackets: list[dict], books: dict) -> tuple[str, int]:
    """
    Find the market favorite: bracket with the highest ask price.
    Returns (favorite_label, favorite_index_in_brackets).
    """
    best_label = None
    best_price = 0

    for b in brackets:
        label = b["label"]
        book = books.get(label)
        if book and book.get("ba", 0) > best_price:
            best_price = book["ba"]
            best_label = label

    # Find index in bracket list
    fav_idx = None
    for i, b in enumerate(brackets):
        if b["label"] == best_label:
            fav_idx = i
            break

    return best_label, fav_idx


# ══════════════════════════════════════════════════════
# STEP 3: GET CANDIDATE BRACKETS (favorite ±1)
# ══════════════════════════════════════════════════════

def get_candidate_brackets(brackets: list[dict], fav_idx: int) -> list[dict]:
    """
    Return favorite and adjacent ±1 brackets (max 3).
    """
    candidates = []
    for offset in [-1, 0, 1]:
        idx = fav_idx + offset
        if 0 <= idx < len(brackets):
            candidates.append(brackets[idx])
    return candidates


# ══════════════════════════════════════════════════════
# STEP 5: CONCENTRATION CHECK
# ══════════════════════════════════════════════════════

def compute_concentration(
    corrected_probs: dict[str, float],
    min_concentration: float = CONCENTRATION_MIN,
) -> tuple[float, float]:
    """
    Compute top-2 concentration and Kelly multiplier.
    Returns (concentration, kelly_multiplier).
    """
    sorted_probs = sorted(corrected_probs.values(), reverse=True)
    top2 = sum(sorted_probs[:2]) if len(sorted_probs) >= 2 else sum(sorted_probs)

    if top2 < config.CONCENTRATION_HARD_FLOOR:
        return top2, 0.0
    if top2 < min_concentration:
        kelly_mult = config.CONCENTRATION_BELOW_MIN_MULT
    else:
        kelly_mult = 0.0
        for threshold, mult in CONCENTRATION_TIERS:
            if top2 >= threshold:
                kelly_mult = mult
                break
        if kelly_mult <= 0:
            kelly_mult = CONCENTRATION_TIERS[-1][1] if CONCENTRATION_TIERS else 0.5

    # Cap at KELLY_CAP
    kelly_mult = min(kelly_mult, KELLY_CAP)
    return top2, kelly_mult


# ══════════════════════════════════════════════════════
# STEP 7: MODEL VS MARKET DIVERGENCE
# ══════════════════════════════════════════════════════

def check_divergence(
    brackets: list[dict],
    corrected_probs: dict[str, float],
    fav_idx: int,
) -> tuple[bool, int, str, str]:
    """
    Check if our model's top bracket diverges too far from market favorite.
    Returns (diverged, distance, model_top_label, market_fav_label).
    """
    # Find model's top bracket
    model_top_label = max(corrected_probs, key=corrected_probs.get)
    model_top_idx = None
    for i, b in enumerate(brackets):
        if b["label"] == model_top_label:
            model_top_idx = i
            break

    if model_top_idx is None or fav_idx is None:
        return False, 0, model_top_label, brackets[fav_idx]["label"] if fav_idx is not None else "?"

    distance = abs(model_top_idx - fav_idx)
    market_fav_label = brackets[fav_idx]["label"]
    diverged = distance > MAX_DIVERGENCE_BRACKETS

    return diverged, distance, model_top_label, market_fav_label


def _is_impossible_after_observed_high(bracket: dict, observed_high_market: float | None) -> bool:
    """
    If observed day-high already exceeds this bracket's top bound, the bracket
    can no longer win for today's highest-temperature market.
    """
    if observed_high_market is None:
        return False
    try:
        obs = float(observed_high_market)
    except (TypeError, ValueError):
        return False

    if bracket.get("type") == "below":
        return obs > float(bracket["high"])
    if bracket.get("type") == "single":
        return obs > float(bracket["val"])
    if bracket.get("type") == "range":
        return obs > float(bracket["high"])
    return False


def _contains_temp(bracket: dict, temp: float) -> bool:
    """True if a bracket contains this temperature value in market units."""
    btype = bracket.get("type")
    if btype == "below":
        return temp <= float(bracket["high"])
    if btype == "single":
        return temp == float(bracket["val"])
    if btype == "range":
        return float(bracket["low"]) <= temp <= float(bracket["high"])
    if btype == "above":
        return temp >= float(bracket["low"])
    return False


def condition_probs_on_observed_high(
    brackets: list[dict],
    probs_by_label: dict[str, float],
    observed_high_market: float | None,
) -> tuple[dict[str, float], dict]:
    """
    Condition bracket probabilities on observed day-high.
    Any bracket already impossible given observed high is set to zero, then
    remaining probability mass is renormalized.

    Returns:
      (conditioned_probs, meta)
    """
    if observed_high_market is None:
        return dict(probs_by_label), {"applied": False, "reason": "no_observed_high"}

    conditioned = {}
    impossible_labels = []
    possible_labels = []
    for b in brackets:
        label = b["label"]
        p = float(probs_by_label.get(label, 0.0) or 0.0)
        if _is_impossible_after_observed_high(b, observed_high_market):
            conditioned[label] = 0.0
            impossible_labels.append(label)
        else:
            conditioned[label] = max(0.0, p)
            possible_labels.append(label)

    total = sum(conditioned.values())
    fallback_label = None

    if total > 0:
        for label in conditioned:
            conditioned[label] /= total
    else:
        # Rare case: prior puts zero mass on all feasible buckets.
        # Fall back to the bracket containing observed high; if missing, use first feasible.
        try:
            obs = float(observed_high_market)
        except (TypeError, ValueError):
            obs = None
        if obs is not None:
            for b in brackets:
                label = b["label"]
                if label in possible_labels and _contains_temp(b, obs):
                    fallback_label = label
                    break
        if fallback_label is None and possible_labels:
            fallback_label = possible_labels[0]
        for label in conditioned:
            conditioned[label] = 0.0
        if fallback_label is not None:
            conditioned[fallback_label] = 1.0

    return conditioned, {
        "applied": True,
        "observed_high_market": observed_high_market,
        "impossible_count": len(impossible_labels),
        "possible_count": len(possible_labels),
        "fallback_label": fallback_label,
    }


# ══════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════

def analyze_brackets(
    brackets: list[dict],
    corrected_probs: dict[str, float],
    raw_probs: dict[str, float],
    market_probs: dict[str, float],
    books: dict[str, dict],
    model_votes: dict[int, int],
    family_votes: dict[int, int],
    total_models: int,
    total_families: int,
    existing_positions: set[str],
    current_exposure: float,
    city_config: dict,
    late_context: dict | None = None,
    live_guard: dict | None = None,
) -> dict:
    """
    Full v5 signal pipeline. Returns a dict with:
    - maturity: (bool, str)
    - favorite: str
    - concentration: float
    - kelly_multiplier: float
    - divergence: (bool, int, str, str)
    - signals: list[dict]  (all candidates with analysis)
    - actionable: list[dict]  (BUY signals only)
    Uses both model-level and family-level agreement filters.
    """
    result = {
        "maturity": (False, "not checked"),
        "favorite": None,
        "favorite_idx": None,
        "concentration": 0.0,
        "kelly_multiplier": 0.0,
        "divergence": (False, 0, "", ""),
        "signals": [],
        "actionable": [],
        "late_guard": late_context or {},
    }

    # ── Step 1: Market maturity ──
    mature, maturity_reason = check_market_maturity(brackets, books, city_config)
    result["maturity"] = (mature, maturity_reason)
    if not mature:
        log.info(f"  Market immature: {maturity_reason}")
        return result

    # ── Step 2: Identify market favorite ──
    fav_label, fav_idx = find_market_favorite(brackets, books)
    result["favorite"] = fav_label
    result["favorite_idx"] = fav_idx

    if fav_idx is None:
        log.warning("  Could not identify market favorite")
        return result

    log.info(f"  Market favorite: {fav_label} (idx {fav_idx})")

    guard = live_guard or {}
    guard_size_mult = _clip(float(guard.get("size_mult", 1.0) or 1.0), 0.0, 1.0)
    guard_edge_bump = max(0.0, float(guard.get("edge_bump", 0.0) or 0.0))

    # ── Step 5: Concentration check ──
    min_concentration = float(city_config.get("min_concentration", CONCENTRATION_MIN))
    concentration, kelly_mult = compute_concentration(
        corrected_probs,
        min_concentration=min_concentration,
    )
    result["concentration"] = concentration
    result["kelly_multiplier"] = kelly_mult
    result["min_concentration"] = min_concentration

    if kelly_mult <= 0:
        log.info(
            "  Concentration too low: %.1f%% (hard floor %.0f%%)",
            concentration * 100,
            config.CONCENTRATION_HARD_FLOOR * 100,
        )
        return result

    if concentration < min_concentration:
        log.info(
            "  Concentration below min: %.1f%% < %.0f%% (soft Kelly %.2fx)",
            concentration * 100,
            min_concentration * 100,
            kelly_mult,
        )
    else:
        log.info(f"  Concentration: {concentration*100:.1f}% → Kelly mult {kelly_mult:.2f}x")

    # ── Step 7: Divergence check ──
    diverged, div_dist, model_top, market_fav = check_divergence(
        brackets, corrected_probs, fav_idx
    )
    result["divergence"] = (diverged, div_dist, model_top, market_fav)

    if diverged:
        log.warning(
            f"  ⚠️ DIVERGENCE: model top={model_top} vs market fav={market_fav} "
            f"({div_dist} brackets apart) → size/edge penalties only"
        )

    # ── Step 3: Get candidate brackets (favorite ±1) ──
    candidates = get_candidate_brackets(brackets, fav_idx)
    result["candidates"] = candidates
    log.info(f"  Candidates: {[c['label'] for c in candidates]}")

    # ── Late-session guard context ──
    late = late_context or {}
    late_applies = (
        config.LATE_GUARD_ENABLED
        and bool(late.get("enabled", True))
        and bool(late.get("applies_today", False))
    )
    late_phase = late.get("phase", "inactive") if late_applies else "inactive"
    late_cutoff_hour = late.get("cutoff_hour")
    late_freeze_hour = late.get("freeze_hour")
    observed_day_high_market = late.get("observed_day_high_market")

    if late_applies:
        log.info(
            "  Late guard: %s (cutoff %s, freeze %s, obs_high=%s)",
            late_phase,
            late_cutoff_hour,
            late_freeze_hour,
            observed_day_high_market,
        )

    # ── Step 4 + 6: Analyze each candidate ──
    min_models = city_config.get("min_models", 2)
    min_families = city_config.get("min_families", min_models)
    remaining_budget = config.MAX_TOTAL_EXPOSURE - current_exposure
    total_families = max(1, total_families)

    signals = []
    for b in candidates:
        label = b["label"]
        raw_p = raw_probs.get(label, 0.0)
        cor_p = corrected_probs.get(label, 0.0)
        mkt_p = market_probs.get(label, 0.0)

        book = books.get(label)
        ask = book["ba"] if book else mkt_p
        bid = book["bb"] if book else 0.0
        spread = book["spread"] if book else 0.0
        bid_depth = book["bid_depth"] if book else 0.0
        ask_depth = book["ask_depth"] if book else 0.0

        raw_edge = cor_p - (bid + ask) / 2 if book else cor_p - mkt_p
        base_true_edge = cor_p - ask

        # Model votes for this bracket
        deg = bracket_degree(b)
        mv = model_votes.get(deg, 0)
        fv = family_votes.get(deg, 0)
        if b.get("type") == "range":
            for d in range(b["low"], b["high"] + 1):
                mv = max(mv, model_votes.get(d, 0))
                fv = max(fv, family_votes.get(d, 0))

        # Base thresholds used by agreement sizing and adaptive edge
        edge_threshold = float(city_config.get("min_edge", config.ADAPTIVE_EDGE_BASE))
        family_threshold = min_families
        if late_applies and late_phase in ("after_cutoff", "freeze"):
            family_threshold = min(
                total_families,
                family_threshold + max(0, config.LATE_GUARD_AFTER_CUTOFF_FAMILY_BUMP),
            )

        agreement_mult, agreement_block_reason = _agreement_multiplier(
            model_votes=mv,
            family_votes=fv,
            min_models=min_models,
            min_families=family_threshold,
        )
        if diverged:
            agreement_mult *= 0.75

        # Kelly with concentration scaling + agreement scaling + live guardrail scaling
        kelly_full = 0.0
        kelly_bet = 0.0
        expected_profit = 0.0
        contracts = 0
        fill_fraction = 1.0
        immediate_fill_fraction = 1.0
        continuation_fill_prob = 1.0
        fee_per_contract = 0.0
        gas_per_contract = 0.0
        slippage_penalty = 0.0
        effective_ask = ask
        edge_after_costs = base_true_edge
        true_edge = base_true_edge

        if ask > 0 and ask < 1 and cor_p > 0:
            # Pass 1: base-size estimate on raw ask
            b_odds = (1 - ask) / ask
            q = 1 - cor_p
            base_kelly = (b_odds * cor_p - q) / b_odds
            base_kelly_frac = max(
                0,
                base_kelly * KELLY_FRACTION * kelly_mult * agreement_mult * guard_size_mult,
            )
            base_bet = round(base_kelly_frac * config.BANKROLL, 2)
            base_bet = min(base_bet, config.MAX_BET_PER_BRACKET)
            base_bet = min(base_bet, max(0, remaining_budget))
            contracts_hint = max(1, int(base_bet / ask)) if base_bet > 0 else 1
            required_depth = _required_ask_depth(city_config, contracts_hint)

            edge_threshold = _adaptive_edge_threshold(
                city_config=city_config,
                spread=spread,
                ask_depth=ask_depth,
                required_depth=required_depth,
                late_applies=late_applies,
                late_phase=late_phase,
                is_favorite=(label == fav_label),
                guard_edge_bump=guard_edge_bump,
            )

            est = _estimate_execution(book, ask, contracts_hint)
            fill_fraction = est["fill_fraction"]
            immediate_fill_fraction = est["immediate_fill_fraction"]
            continuation_fill_prob = est["continuation_fill_prob"]
            slippage_penalty = est["slippage_penalty"] if config.EXECUTION_ADJUSTED_EDGE_ENABLED else 0.0

            fee_per_contract = ask * config.EXECUTION_FEE_RATE if config.EXECUTION_ADJUSTED_EDGE_ENABLED else 0.0
            gas_per_contract = (
                config.EXECUTION_GAS_USD_PER_ORDER / max(1, contracts_hint)
                if config.EXECUTION_ADJUSTED_EDGE_ENABLED else 0.0
            )
            effective_ask = min(0.99, ask + fee_per_contract + gas_per_contract + slippage_penalty)

            kelly_price = effective_ask if config.EXECUTION_ADJUSTED_EDGE_ENABLED else ask
            if kelly_price > 0 and kelly_price < 1:
                eff_b = (1 - kelly_price) / kelly_price
                kelly_full = (eff_b * cor_p - (1 - cor_p)) / eff_b
                fill_mult = fill_fraction if config.EXECUTION_ADJUSTED_EDGE_ENABLED else 1.0
                kelly_frac = max(
                    0,
                    kelly_full * KELLY_FRACTION * kelly_mult * fill_mult * agreement_mult * guard_size_mult,
                )
                kelly_bet = round(kelly_frac * config.BANKROLL, 2)

                # Apply caps
                kelly_bet = min(kelly_bet, config.MAX_BET_PER_BRACKET)
                kelly_bet = min(kelly_bet, max(0, remaining_budget))

                if kelly_bet > 0 and ask > 0:
                    contracts = int(kelly_bet / ask)

            # Pass 2: refresh execution metrics using final contract count
            if contracts > 0:
                est2 = _estimate_execution(book, ask, contracts)
                fill_fraction = est2["fill_fraction"]
                immediate_fill_fraction = est2["immediate_fill_fraction"]
                continuation_fill_prob = est2["continuation_fill_prob"]
                slippage_penalty = est2["slippage_penalty"] if config.EXECUTION_ADJUSTED_EDGE_ENABLED else 0.0
                fee_per_contract = ask * config.EXECUTION_FEE_RATE if config.EXECUTION_ADJUSTED_EDGE_ENABLED else 0.0
                gas_per_contract = (
                    config.EXECUTION_GAS_USD_PER_ORDER / contracts
                    if config.EXECUTION_ADJUSTED_EDGE_ENABLED else 0.0
                )
                effective_ask = min(0.99, ask + fee_per_contract + gas_per_contract + slippage_penalty)

            edge_after_costs = cor_p - effective_ask
            true_edge = edge_after_costs * fill_fraction if config.EXECUTION_ADJUSTED_EDGE_ENABLED else edge_after_costs
            expected_profit = round(contracts * fill_fraction * edge_after_costs, 2) if contracts > 0 else 0.0

        # Final adaptive edge threshold uses final contract/depth estimate when available.
        required_depth = _required_ask_depth(city_config, contracts if contracts > 0 else 1)
        edge_threshold = _adaptive_edge_threshold(
            city_config=city_config,
            spread=spread,
            ask_depth=ask_depth,
            required_depth=required_depth,
            late_applies=late_applies,
            late_phase=late_phase,
            is_favorite=(label == fav_label),
            guard_edge_bump=guard_edge_bump,
        )
        filter_reasons = []
        if true_edge < edge_threshold:
            filter_reasons.append(
                f"edge {true_edge*100:.1f}pt < {edge_threshold*100:.0f}pt"
            )
        if agreement_block_reason:
            filter_reasons.append(agreement_block_reason)
        if ask > MAX_ASK_PRICE:
            filter_reasons.append(
                f"ask {ask*100:.0f}¢ > {MAX_ASK_PRICE*100:.0f}¢"
            )
        if label in existing_positions:
            filter_reasons.append("already holding")
        if kelly_bet <= 0:
            filter_reasons.append("kelly ≤ 0")
        if remaining_budget <= 0:
            filter_reasons.append("max exposure")

        if _is_impossible_after_observed_high(b, observed_day_high_market):
            filter_reasons.append(
                f"observed high {observed_day_high_market} already above bracket"
            )

        passed = len(filter_reasons) == 0 and true_edge > 0
        signal = "BUY" if passed else ("FILTERED" if true_edge > 0 else "PASS")

        sig_dict = {
            "bracket": label,
            "signal": signal,
            "raw_prob": raw_p,
            "corrected_prob": cor_p,
            "market_prob": mkt_p,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "required_ask_depth": required_depth,
            "raw_edge": raw_edge,
            "base_true_edge": base_true_edge,
            "true_edge": true_edge,
            "edge_after_costs": edge_after_costs,
            "effective_ask": effective_ask,
            "fill_fraction": fill_fraction,
            "immediate_fill_fraction": immediate_fill_fraction,
            "continuation_fill_prob": continuation_fill_prob,
            "fee_per_contract": fee_per_contract,
            "gas_per_contract": gas_per_contract,
            "slippage_penalty": slippage_penalty,
            "execution_adjusted": config.EXECUTION_ADJUSTED_EDGE_ENABLED,
            "model_votes": mv,
            "family_votes": fv,
            "agreement_multiplier": agreement_mult,
            "total_models": total_models,
            "total_families": total_families,
            "kelly_full": kelly_full,
            "kelly_bet": kelly_bet,
            "kelly_multiplier": kelly_mult,
            "guardrail_multiplier": guard_size_mult,
            "concentration": concentration,
            "expected_profit": expected_profit,
            "contracts": contracts,
            "filter_reasons": filter_reasons,
            "is_favorite": (label == fav_label),
            "edge_threshold": edge_threshold,
            "family_threshold": family_threshold,
            "late_phase": late_phase,
            "late_applies": late_applies,
        }
        signals.append(sig_dict)

        if passed and kelly_bet > 0:
            remaining_budget -= kelly_bet

    # Guardrail: avoid weak non-favorite picks when favorite still has positive edge.
    # This reduces repeated contrarian stabs unless non-favorite edge is materially better.
    nonfav_adv_req = float(
        city_config.get(
            "nonfavorite_edge_advantage_required",
            config.NONFAVORITE_EDGE_ADVANTAGE_REQUIRED,
        )
    )
    if nonfav_adv_req > 0:
        favorite_signal = next((s for s in signals if s["is_favorite"]), None)
        favorite_edge = favorite_signal.get("true_edge", 0.0) if favorite_signal else 0.0
        if favorite_edge > 0:
            floor = favorite_edge + nonfav_adv_req
            for s in signals:
                if s["signal"] != "BUY" or s["is_favorite"]:
                    continue
                if s["true_edge"] < floor:
                    s["signal"] = "FILTERED"
                    s["filter_reasons"].append(
                        f"non-favorite edge {s['true_edge']*100:.1f}pt < favorite+{nonfav_adv_req*100:.0f}pt"
                    )

    # Sort by edge descending
    signals.sort(key=lambda r: r["true_edge"], reverse=True)
    result["signals"] = signals
    result["actionable"] = [s for s in signals if s["signal"] == "BUY"]

    return result


def get_actionable_signals(analysis_result: dict) -> list[dict]:
    """Return only BUY signals from analysis result."""
    return analysis_result.get("actionable", [])


def compute_totals(analysis_result: dict) -> dict:
    """Compute aggregate stats."""
    buys = analysis_result.get("actionable", [])
    total_bet = sum(a["kelly_bet"] for a in buys)
    total_ev = sum(a["expected_profit"] for a in buys)
    total_contracts = sum(a["contracts"] for a in buys)

    return {
        "buy_count": len(buys),
        "total_bet": total_bet,
        "total_ev": total_ev,
        "total_contracts": total_contracts,
        "roi_pct": (total_ev / total_bet * 100) if total_bet > 0 else 0,
    }
