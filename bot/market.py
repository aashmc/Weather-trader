"""
Weather Trader Bot — Polymarket Interface
Fetches market data (brackets, prices, order books) and places orders via CLOB API.
"""

import re
import json
import time
import logging
import asyncio
import httpx

log = logging.getLogger("market")

GAMMA_API = "https://gamma-api.polymarket.com/events"
CLOB_HOST = "https://clob.polymarket.com"

# Polygon USDC.e (Polymarket collateral)
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_DECIMALS = 6
POLYGON_RPC = "https://polygon-rpc.com"

# Minimal ERC-20 ABI for balanceOf
BALANCE_OF_SELECTOR = "0x70a08231"  # keccak256("balanceOf(address)")[:4]

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def build_slug(city: dict, date_str: str) -> str:
    """Build Polymarket event slug from city and date."""
    parts = date_str.split("-")
    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    return f"highest-temperature-in-{city['poly_city']}-on-{MONTHS[month - 1]}-{day}-{year}"


def parse_bracket(label: str) -> dict | None:
    """Parse bracket label into structured info."""
    is_f = "°F" in label
    unit = "F" if is_f else "C"

    if "or below" in label or "or less" in label:
        m = re.search(r"([-\d]+)", label)
        if m:
            return {"label": label, "type": "below", "high": int(m.group(1)), "unit": unit}

    if "or higher" in label or "or above" in label:
        m = re.search(r"([-\d]+)", label)
        if m:
            return {"label": label, "type": "above", "low": int(m.group(1)), "unit": unit}

    rm = re.search(r"([-\d]+)[\s]*[-–][\s]*([-\d]+)", label)
    if rm:
        return {
            "label": label,
            "type": "range",
            "low": int(rm.group(1)),
            "high": int(rm.group(2)),
            "unit": unit,
        }

    sm = re.search(r"([-\d]+)", label)
    if sm:
        return {"label": label, "type": "single", "val": int(sm.group(1)), "unit": unit}

    return None


def temp_to_bracket(temp: float, brackets: list[dict]) -> str | None:
    """Map a temperature to a bracket label."""
    rd = round(temp)
    for b in brackets:
        if b["type"] == "below" and rd <= b["high"]:
            return b["label"]
        if b["type"] == "above" and rd >= b["low"]:
            return b["label"]
        if b["type"] == "single" and rd == b["val"]:
            return b["label"]
        if b["type"] == "range" and b["low"] <= rd <= b["high"]:
            return b["label"]
    return brackets[0]["label"] if brackets else None


def bracket_degree(b: dict) -> int:
    """Get representative degree for a bracket (for model vote lookup)."""
    if "val" in b:
        return b["val"]
    if "low" in b:
        return b["low"]
    if "high" in b:
        return b["high"]
    return 0


async def fetch_market(city: dict, date_str: str) -> dict:
    """
    Fetch Polymarket event data for a city/date.
    Returns:
        {
            "brackets": [{"label", "type", "low", "high", "val", "unit"}],
            "prices": {"label": float},
            "token_ids": {"label": str},
            "slug": str,
            "active": bool,
            "resolved": bool,
            "winner": str or None,
        }
    """
    slug = build_slug(city, date_str)
    url = f"{GAMMA_API}?slug={slug}"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    if not data:
        raise ValueError(f"No event found for {slug}")

    event = data[0] if isinstance(data, list) else data
    markets = event.get("markets", [])
    if not markets:
        raise ValueError(f"No markets in event {slug}")

    # Validate slug matches city
    ev_slug = event.get("slug", "")
    if ev_slug and city["poly_city"] not in ev_slug:
        raise ValueError(f"Slug mismatch: got {ev_slug}, expected {city['poly_city']}")

    prices = {}
    token_ids = {}
    bracket_labels = []
    winner = None
    resolved = False

    for m in markets:
        title = m.get("groupItemTitle", "")
        if not title:
            continue

        # Prices
        try:
            op = json.loads(m.get("outcomePrices", "[]"))
            if op:
                yes_price = float(op[0])
                prices[title] = yes_price
                # Check resolution
                if yes_price >= 0.99:
                    winner = title
                    resolved = True
        except (json.JSONDecodeError, IndexError, ValueError):
            continue

        # Token IDs for CLOB
        try:
            tids = json.loads(m.get("clobTokenIds", "[]"))
            if tids:
                token_ids[title] = tids[0]  # YES token
        except (json.JSONDecodeError, IndexError):
            pass

        bracket_labels.append(title)

    # Parse and sort brackets
    brackets = []
    for label in bracket_labels:
        parsed = parse_bracket(label)
        if parsed:
            brackets.append(parsed)

    brackets.sort(key=lambda b: b.get("val", b.get("low", b.get("high", 0))))

    log.info(
        f"{city['name']}: {len(brackets)} brackets, "
        f"{len(token_ids)} CLOB IDs"
        + (f", RESOLVED → {winner}" if resolved else "")
    )

    return {
        "brackets": brackets,
        "prices": prices,
        "token_ids": token_ids,
        "slug": slug,
        "active": event.get("active", True),
        "resolved": resolved,
        "winner": winner,
    }


async def fetch_order_book(token_id: str) -> dict:
    """
    Fetch CLOB order book for a single token.
    Returns: {"bb": float, "ba": float, "spread": float, "bid_depth": float, "ask_depth": float}
    """
    url = f"{CLOB_HOST}/book?token_id={token_id}"

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    bids = sorted(data.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
    asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))

    bb = float(bids[0]["price"]) if bids else 0.0
    ba = float(asks[0]["price"]) if asks else 1.0
    bid_depth = sum(float(b["size"]) for b in bids)
    ask_depth = sum(float(a["size"]) for a in asks)

    return {
        "bb": bb,
        "ba": ba,
        "spread": ba - bb,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "bids": bids,
        "asks": asks,
    }


async def fetch_all_books(token_ids: dict) -> dict[str, dict]:
    """Fetch order books for all brackets. Returns {label: book_data}."""
    books = {}
    for label, tid in token_ids.items():
        try:
            books[label] = await fetch_order_book(tid)
        except Exception as e:
            log.warning(f"Book fetch failed for {label}: {e}")
    log.info(f"Order books: {len(books)}/{len(token_ids)}")
    return books


async def check_resolution(city: dict, date_str: str) -> dict:
    """
    Check if a market has resolved.
    Returns: {"resolved": bool, "winner": str or None}
    """
    try:
        market = await fetch_market(city, date_str)
        return {"resolved": market["resolved"], "winner": market["winner"]}
    except Exception as e:
        log.warning(f"Resolution check failed for {city['name']} {date_str}: {e}")
        return {"resolved": False, "winner": None}


# ══════════════════════════════════════════════════════
# ORDER PLACEMENT (via py-clob-client)
# ══════════════════════════════════════════════════════

_clob_client = None


def get_clob_client():
    """Lazy-initialize the CLOB client."""
    global _clob_client
    if _clob_client is not None:
        return _clob_client

    try:
        from py_clob_client.client import ClobClient
        from config import POLYGON_PRIVATE_KEY, POLYMARKET_HOST, CHAIN_ID
        import os

        if not POLYGON_PRIVATE_KEY:
            log.error("No POLYGON_PRIVATE_KEY set — orders disabled")
            return None

        proxy_wallet = os.getenv("PROXY_WALLET", "").strip()

        if proxy_wallet:
            # Magic/email wallet: sign with private key, funds in proxy
            _clob_client = ClobClient(
                POLYMARKET_HOST,
                key=POLYGON_PRIVATE_KEY,
                chain_id=CHAIN_ID,
                signature_type=1,
                funder=proxy_wallet,
            )
            log.info(f"CLOB client: signature_type=1, funder={proxy_wallet[:10]}...")
        else:
            # EOA direct wallet
            _clob_client = ClobClient(
                POLYMARKET_HOST,
                key=POLYGON_PRIVATE_KEY,
                chain_id=CHAIN_ID,
            )
            log.info("CLOB client: signature_type=0 (EOA direct)")

        # Derive or create API creds
        try:
            _clob_client.set_api_creds(_clob_client.create_or_derive_api_creds())
            log.info("CLOB client initialized with API creds")
        except Exception as e:
            log.warning(f"API creds derivation: {e}")

        return _clob_client
    except ImportError:
        log.error("py-clob-client not installed — orders disabled")
        return None
    except Exception as e:
        log.error(f"CLOB client init failed: {e}")
        return None


async def fetch_wallet_balance() -> float:
    """
    Fetch available USDC balance via Polymarket CLOB API.
    Returns balance in USD (float), or -1 on failure.
    """
    try:
        client = get_clob_client()
        if client is None:
            log.warning("No CLOB client — cannot fetch balance")
            return -1

        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

        result = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        log.info(f"CLOB balance raw response: {result}")

        # result may be dict or have balance as string
        raw_balance = result.get("balance", 0) if isinstance(result, dict) else 0
        raw_balance = float(raw_balance)

        # Check if already in USD or needs decimal conversion
        if raw_balance > 1_000_000:
            # Looks like raw units (6 decimals for USDC)
            balance = raw_balance / (10 ** USDC_DECIMALS)
        else:
            # Already in USD
            balance = raw_balance

        log.info(f"Wallet balance: ${balance:.2f} USDC")
        return balance

    except Exception as e:
        log.warning(f"CLOB balance fetch failed: {e}")
        return -1


async def place_limit_order(
    token_id: str,
    price: float,
    size: float,
    bracket_label: str,
) -> dict:
    """
    Place a limit buy order for YES shares.
    Returns: {"order_id": str, "success": bool, "error": str or None}
    """
    client = get_clob_client()
    if client is None:
        return {"order_id": None, "success": False, "error": "No CLOB client"}

    try:
        from py_clob_client.order_builder.constants import BUY
        from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=BUY,
        )

        options = PartialCreateOrderOptions(
            tick_size="0.01",
            neg_risk=False,
        )

        order = client.create_and_post_order(order_args, options)

        order_id = order.get("orderID") or order.get("id") or str(order)
        log.info(f"Order placed: {bracket_label} @ {price:.3f} x{size:.0f} → {order_id}")

        return {"order_id": order_id, "success": True, "error": None}

    except Exception as e:
        log.error(f"Order failed for {bracket_label}: {e}")
        return {"order_id": None, "success": False, "error": str(e)}


async def cancel_order(order_id: str) -> bool:
    """Cancel an open order."""
    client = get_clob_client()
    if client is None:
        return False

    try:
        client.cancel(order_id)
        log.info(f"Order cancelled: {order_id}")
        return True
    except Exception as e:
        log.warning(f"Cancel failed for {order_id}: {e}")
        return False


async def check_order_status(order_id: str) -> str:
    """Check if an order was filled. Returns 'filled', 'open', 'cancelled', or 'unknown'."""
    client = get_clob_client()
    if client is None:
        return "unknown"

    try:
        order = client.get_order(order_id)
        status = order.get("status", "unknown")
        size_matched = float(order.get("size_matched", 0))
        original_size = float(order.get("original_size", 1))

        if size_matched >= original_size * 0.95:  # 95% filled = filled
            return "filled"
        elif size_matched > 0:
            return "partial"
        elif status in ("cancelled", "expired"):
            return "cancelled"
        else:
            return "open"
    except Exception as e:
        log.warning(f"Status check failed for {order_id}: {e}")
        return "unknown"
