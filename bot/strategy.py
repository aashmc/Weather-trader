"""
Weather Trader Bot — Strategy
Edge detection, Kelly criterion sizing, and signal generation.
"""

import logging
from config import (
    KELLY_FRACTION, BANKROLL, MAX_BET_PER_BRACKET, MAX_TOTAL_EXPOSURE,
    MIN_TRUE_EDGE, MIN_MODEL_AGREEMENT, MIN_ASK_DEPTH, MAX_ASK_PRICE,
)
from market import bracket_degree

log = logging.getLogger("strategy")


def analyze_brackets(
    brackets: list[dict],
    corrected_probs: dict[str, float],
    raw_probs: dict[str, float],
    market_probs: dict[str, float],
    books: dict[str, dict],
    model_votes: dict[int, int],
    total_models: int,
    existing_positions: set[str],  # bracket labels already holding a position
    current_exposure: float,       # total $ already wagered today
) -> list[dict]:
    """
    Analyze all brackets and generate trading signals.
    
    Returns list of signal dicts, sorted by edge descending:
    [{
        "bracket": str,
        "signal": "BUY" | "PASS" | "FILTERED",
        "raw_prob": float,
        "corrected_prob": float,
        "market_prob": float,
        "bid": float, "ask": float, "spread": float,
        "bid_depth": float, "ask_depth": float,
        "raw_edge": float, "true_edge": float,
        "model_votes": int, "total_models": int,
        "kelly_full": float, "kelly_bet": float,
        "expected_profit": float,
        "contracts": int,
        "filter_reasons": [str],
    }]
    """
    results = []
    remaining_budget = MAX_TOTAL_EXPOSURE - current_exposure

    for b in brackets:
        label = b["label"]
        raw_p = raw_probs.get(label, 0.0)
        cor_p = corrected_probs.get(label, 0.0)
        mkt_p = market_probs.get(label, 0.0)

        book = books.get(label)
        ask = book["ba"] if book else mkt_p
        bid = book["bb"] if book else 0.0
        mid = (bid + ask) / 2 if book else mkt_p
        spread = book["spread"] if book else 0.0
        bid_depth = book["bid_depth"] if book else 0.0
        ask_depth = book["ask_depth"] if book else 0.0

        raw_edge = cor_p - mid
        true_edge = cor_p - ask

        # Model agreement for this bracket
        deg = bracket_degree(b)
        mv = model_votes.get(deg, 0)
        # For range brackets, check all degrees in range
        if b["type"] == "range":
            for d in range(b["low"], b["high"] + 1):
                mv = max(mv, model_votes.get(d, 0))

        # Kelly calculation
        kelly_full = 0.0
        kelly_bet = 0.0
        expected_profit = 0.0
        contracts = 0

        if ask > 0 and ask < 1 and cor_p > 0:
            b_odds = (1 - ask) / ask  # decimal odds - 1
            q = 1 - cor_p
            kelly_full = (b_odds * cor_p - q) / b_odds
            kelly_frac = max(0, kelly_full * KELLY_FRACTION)
            kelly_bet = round(kelly_frac * BANKROLL, 2)

            # Apply caps
            kelly_bet = min(kelly_bet, MAX_BET_PER_BRACKET)
            kelly_bet = min(kelly_bet, max(0, remaining_budget))

            if kelly_bet > 0 and ask > 0:
                contracts = int(kelly_bet / ask)
                expected_profit = round(kelly_bet * (cor_p / ask - 1), 2)

        # Filter checks
        filter_reasons = []
        if true_edge < MIN_TRUE_EDGE:
            filter_reasons.append(
                f"Edge {true_edge * 100:.1f}pt < {MIN_TRUE_EDGE * 100:.0f}pt min"
            )
        if mv < MIN_MODEL_AGREEMENT:
            filter_reasons.append(
                f"Models {mv}/{total_models} < {MIN_MODEL_AGREEMENT} min"
            )
        if ask_depth < MIN_ASK_DEPTH and MIN_ASK_DEPTH > 0:
            filter_reasons.append(
                f"Depth {ask_depth:.0f} < {MIN_ASK_DEPTH} min"
            )
        if ask > MAX_ASK_PRICE:
            filter_reasons.append(
                f"Ask {ask * 100:.0f}¢ > {MAX_ASK_PRICE * 100:.0f}¢ max"
            )
        if label in existing_positions:
            filter_reasons.append("Already holding position")
        if kelly_bet <= 0:
            filter_reasons.append("Kelly bet ≤ 0")
        if remaining_budget <= 0:
            filter_reasons.append("Max exposure reached")

        passed = len(filter_reasons) == 0 and true_edge > 0

        signal = "BUY" if passed else ("FILTERED" if true_edge > 0 else "PASS")

        results.append({
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
            "expected_profit": expected_profit,
            "contracts": contracts,
            "filter_reasons": filter_reasons,
        })

        # If this will be a BUY, deduct from remaining budget
        if passed and kelly_bet > 0:
            remaining_budget -= kelly_bet

    # Sort by true edge descending
    results.sort(key=lambda r: r["true_edge"], reverse=True)
    return results


def get_actionable_signals(analysis: list[dict]) -> list[dict]:
    """Return only BUY signals from analysis."""
    return [a for a in analysis if a["signal"] == "BUY"]


def compute_totals(analysis: list[dict]) -> dict:
    """Compute aggregate stats from analysis."""
    buys = get_actionable_signals(analysis)
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
