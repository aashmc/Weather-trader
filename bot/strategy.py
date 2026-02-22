"""
Weather Trader Bot — Strategy v5
Full signal pipeline:
  1. Market maturity check
  2. Identify market favorite
  3. Candidates = favorite ±1
  4. Per-city filters (models, edge)
  5. Concentration check (top 2 ≥ 50%)
  6. Kelly scaling by concentration (capped at 75%)
  7. Model vs market divergence warning
"""

import logging
from config import (
    KELLY_FRACTION, KELLY_CAP, BANKROLL, MAX_BET_PER_BRACKET, MAX_TOTAL_EXPOSURE,
    MIN_ASK_DEPTH, MAX_ASK_PRICE,
    MATURITY_MIN_FAV_PRICE, MATURITY_MAX_FAV_SPREAD, MATURITY_MIN_LIQUID_BRACKETS,
    CONCENTRATION_MIN, CONCENTRATION_TIERS,
    MAX_DIVERGENCE_BRACKETS,
)
from market import bracket_degree

log = logging.getLogger("strategy")


# ══════════════════════════════════════════════════════
# STEP 1: MARKET MATURITY CHECK
# ══════════════════════════════════════════════════════

def check_market_maturity(brackets: list[dict], books: dict) -> tuple[bool, str]:
    """
    Check if the market is mature enough to trade.
    Returns (mature, reason).
    """
    if not books:
        return False, "no order books"

    # Check 1: Any bracket with ask ≥ 25¢?
    max_ask = 0
    for label, book in books.items():
        if book and book.get("ba", 0) > max_ask:
            max_ask = book["ba"]

    if max_ask < MATURITY_MIN_FAV_PRICE:
        return False, f"no bracket ≥{MATURITY_MIN_FAV_PRICE*100:.0f}¢ (max={max_ask*100:.1f}¢)"

    # Check 2: Favorite bracket spread ≤ 3¢
    fav_label = max(books, key=lambda k: books[k]["ba"] if books[k] else 0)
    fav_book = books[fav_label]
    if fav_book and fav_book.get("spread", 1) > MATURITY_MAX_FAV_SPREAD:
        return False, f"favorite spread {fav_book['spread']*100:.1f}¢ > {MATURITY_MAX_FAV_SPREAD*100:.0f}¢"

    # Check 3: At least 3 brackets with ask depth ≥ 10
    liquid_count = sum(
        1 for b in books.values()
        if b and b.get("ask_depth", 0) >= 10
    )
    if liquid_count < MATURITY_MIN_LIQUID_BRACKETS:
        return False, f"only {liquid_count} liquid brackets (need {MATURITY_MIN_LIQUID_BRACKETS})"

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

def compute_concentration(corrected_probs: dict[str, float]) -> tuple[float, float]:
    """
    Compute top-2 concentration and Kelly multiplier.
    Returns (concentration, kelly_multiplier).
    """
    sorted_probs = sorted(corrected_probs.values(), reverse=True)
    top2 = sum(sorted_probs[:2]) if len(sorted_probs) >= 2 else sum(sorted_probs)

    if top2 < CONCENTRATION_MIN:
        return top2, 0.0

    kelly_mult = 0.0
    for threshold, mult in CONCENTRATION_TIERS:
        if top2 >= threshold:
            kelly_mult = mult
            break

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
    total_models: int,
    existing_positions: set[str],
    current_exposure: float,
    city_config: dict,
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
    }

    # ── Step 1: Market maturity ──
    mature, maturity_reason = check_market_maturity(brackets, books)
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

    # ── Step 5: Concentration check ──
    concentration, kelly_mult = compute_concentration(corrected_probs)
    result["concentration"] = concentration
    result["kelly_multiplier"] = kelly_mult

    if kelly_mult <= 0:
        log.info(f"  Concentration too low: {concentration*100:.1f}% (need ≥{CONCENTRATION_MIN*100:.0f}%)")
        return result

    log.info(f"  Concentration: {concentration*100:.1f}% → Kelly mult {kelly_mult:.2f}x")

    # ── Step 7: Divergence check ──
    diverged, div_dist, model_top, market_fav = check_divergence(
        brackets, corrected_probs, fav_idx
    )
    result["divergence"] = (diverged, div_dist, model_top, market_fav)

    if diverged:
        log.warning(
            f"  ⚠️ DIVERGENCE: model top={model_top} vs market fav={market_fav} "
            f"({div_dist} brackets apart) → skipping"
        )
        return result

    # ── Step 3: Get candidate brackets (favorite ±1) ──
    candidates = get_candidate_brackets(brackets, fav_idx)
    candidate_labels = {c["label"] for c in candidates}
    log.info(f"  Candidates: {[c['label'] for c in candidates]}")

    # ── Step 4 + 6: Analyze each candidate ──
    min_models = city_config.get("min_models", 2)
    min_edge = city_config.get("min_edge", 0.05)
    remaining_budget = MAX_TOTAL_EXPOSURE - current_exposure

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
        true_edge = cor_p - ask

        # Model votes for this bracket
        deg = bracket_degree(b)
        mv = model_votes.get(deg, 0)
        if b.get("type") == "range":
            for d in range(b["low"], b["high"] + 1):
                mv = max(mv, model_votes.get(d, 0))

        # Kelly with concentration scaling
        kelly_full = 0.0
        kelly_bet = 0.0
        expected_profit = 0.0
        contracts = 0

        if ask > 0 and ask < 1 and cor_p > 0:
            b_odds = (1 - ask) / ask
            q = 1 - cor_p
            kelly_full = (b_odds * cor_p - q) / b_odds
            # Apply ⅓ Kelly × concentration multiplier
            kelly_frac = max(0, kelly_full * KELLY_FRACTION * kelly_mult)
            kelly_bet = round(kelly_frac * BANKROLL, 2)

            # Apply caps
            kelly_bet = min(kelly_bet, MAX_BET_PER_BRACKET)
            kelly_bet = min(kelly_bet, max(0, remaining_budget))

            if kelly_bet > 0 and ask > 0:
                contracts = int(kelly_bet / ask)
                expected_profit = round(kelly_bet * (cor_p / ask - 1), 2)

        # Filter checks
        filter_reasons = []
        if true_edge < min_edge:
            filter_reasons.append(
                f"edge {true_edge*100:.1f}pt < {min_edge*100:.0f}pt"
            )
        if mv < min_models:
            filter_reasons.append(
                f"models {mv}/{total_models} < {min_models}"
            )
        if ask_depth < MIN_ASK_DEPTH:
            filter_reasons.append(
                f"depth {ask_depth:.0f} < {MIN_ASK_DEPTH}"
            )
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
            "raw_edge": raw_edge,
            "true_edge": true_edge,
            "model_votes": mv,
            "total_models": total_models,
            "kelly_full": kelly_full,
            "kelly_bet": kelly_bet,
            "kelly_multiplier": kelly_mult,
            "concentration": concentration,
            "expected_profit": expected_profit,
            "contracts": contracts,
            "filter_reasons": filter_reasons,
            "is_favorite": (label == fav_label),
        }
        signals.append(sig_dict)

        if passed and kelly_bet > 0:
            remaining_budget -= kelly_bet

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
