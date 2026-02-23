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


def _parse_json_maybe(value):
    """
    Parse API fields that may be JSON strings or already-parsed lists.
    Returns parsed value or None.
    """
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
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
            op = _parse_json_maybe(m.get("outcomePrices", "[]")) or []
            if op:
                yes_price = float(op[0])
                prices[title] = yes_price
                # Check resolution
                if yes_price >= 0.99:
                    winner = title
                    resolved = True
        except (IndexError, ValueError, TypeError):
            pass

        # Token IDs for CLOB
        try:
            tids = _parse_json_maybe(m.get("clobTokenIds", "[]")) or []
            if tids:
                token_ids[title] = tids[0]  # YES token
        except (IndexError, TypeError):
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
_clob_signature_type = None


def _normalize_private_key(key: str) -> str:
    """Normalize private key format expected by web3 tooling."""
    key = (key or "").strip()
    if key and not key.startswith("0x"):
        key = "0x" + key
    return key


def _normalize_wallet_address(addr: str) -> str:
    """Normalize hex wallet address string."""
    addr = (addr or "").strip()
    if addr and not addr.startswith("0x"):
        addr = "0x" + addr
    return addr


def _is_signature_error(msg: str) -> bool:
    text = (msg or "").lower()
    return (
        "invalid signature" in text
        or "invalid order signature" in text
        or "signature verification failed" in text
    )


def _primary_signature_type(proxy_wallet: str) -> int:
    """
    Decide the initial signature type.
    - default 2 when using proxy wallet (most common account setup)
    - default 0 for direct EOA
    - override via POLY_SIGNATURE_TYPE env
    """
    import os

    raw = os.getenv("POLY_SIGNATURE_TYPE", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 2 if proxy_wallet else 0


def _fallback_signature_types(primary: int, proxy_wallet: str) -> list[int]:
    """
    Signature-type fallbacks for invalid-signature retries.
    Override via POLY_SIGNATURE_TYPE_FALLBACKS, e.g. "2,1,0".
    """
    import os

    raw = os.getenv("POLY_SIGNATURE_TYPE_FALLBACKS", "").strip()
    vals = []
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(int(part))
            except ValueError:
                continue
    elif proxy_wallet:
        # Common migration path: proxy wallets sometimes require sig_type=2.
        vals = [2, 1]
    else:
        vals = [0]

    out = []
    for v in vals:
        if v == primary or v in out:
            continue
        out.append(v)
    return out


def _init_clob_client(signature_type: int | None = None):
    """Create and authenticate a new CLOB client instance."""
    from py_clob_client.client import ClobClient
    from config import POLYGON_PRIVATE_KEY, POLYMARKET_HOST, CHAIN_ID
    import os

    private_key = _normalize_private_key(POLYGON_PRIVATE_KEY)
    if not private_key:
        log.error("No POLYGON_PRIVATE_KEY set — orders disabled")
        return None, None, ""

    proxy_wallet = _normalize_wallet_address(os.getenv("PROXY_WALLET", ""))
    sig_type = signature_type if signature_type is not None else _primary_signature_type(proxy_wallet)
    if not proxy_wallet and sig_type in (1, 2):
        log.warning(
            f"POLY signature_type={sig_type} requires PROXY_WALLET; falling back to signature_type=0"
        )
        sig_type = 0

    kwargs = {
        "host": POLYMARKET_HOST,
        "key": private_key,
        "chain_id": CHAIN_ID,
        "signature_type": sig_type,
    }
    if proxy_wallet:
        kwargs["funder"] = proxy_wallet

    client = ClobClient(**kwargs)

    # Derive or create API creds
    try:
        client.set_api_creds(client.create_or_derive_api_creds())
    except Exception as e:
        log.warning(f"API creds derivation: {e}")

    try:
        signer_addr = client.builder.signer.address()
    except Exception:
        signer_addr = "?"

    if proxy_wallet:
        log.info(
            f"CLOB client initialized: sig_type={sig_type}, "
            f"signer={str(signer_addr)[:10]}..., funder={proxy_wallet[:10]}..."
        )
    else:
        log.info(
            f"CLOB client initialized: sig_type={sig_type}, signer={str(signer_addr)[:10]}... (EOA)"
        )

    return client, sig_type, proxy_wallet


def get_clob_client(signature_type: int | None = None, force_reinit: bool = False):
    """Lazy-initialize the CLOB client."""
    global _clob_client, _clob_signature_type
    if (
        not force_reinit
        and _clob_client is not None
        and (signature_type is None or signature_type == _clob_signature_type)
    ):
        return _clob_client

    try:
        _clob_client, _clob_signature_type, _ = _init_clob_client(signature_type=signature_type)
        return _clob_client
    except ImportError:
        log.error("py-clob-client not installed — orders disabled")
        return None
    except Exception as e:
        log.error(f"CLOB client init failed: {e}")
        return None


async def fetch_wallet_balance() -> float:
    """
    Fetch available cash balance via Polymarket accounting snapshot API.
    Returns cashBalance in USD (float), or -1 on failure.
    """
    import os, zipfile, io, csv

    proxy_wallet = os.getenv("PROXY_WALLET", "")
    if not proxy_wallet:
        log.warning("No PROXY_WALLET configured — cannot fetch balance")
        return -1

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"https://data-api.polymarket.com/v1/accounting/snapshot?user={proxy_wallet}"
            resp = await client.get(url)
            resp.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(resp.content))
        equity_csv = z.read("equity.csv").decode("utf-8").strip()
        reader = csv.DictReader(io.StringIO(equity_csv))
        row = next(reader, None)

        if row is None:
            log.warning("Accounting snapshot: equity.csv is empty")
            return -1

        cash = float(row.get("cashBalance", 0))
        positions_val = float(row.get("positionsValue", 0))
        equity = float(row.get("equity", 0))
        snap_time = row.get("valuationTime", "?")

        log.info(
            f"Wallet balance: ${cash:.2f} cash, "
            f"${positions_val:.2f} positions, "
            f"${equity:.2f} equity (snapshot {snap_time})"
        )
        return cash

    except Exception as e:
        log.warning(f"Accounting snapshot fetch failed: {e}")
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
        from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions, OrderType
        import os

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=BUY,
        )

        options = PartialCreateOrderOptions(
            tick_size="0.01",
            neg_risk=True,
        )

        def submit_with(active_client, active_options):
            # Avoid create_and_post_order() for neg-risk markets; use explicit path.
            if active_options is None:
                signed_order = active_client.create_order(order_args)
            else:
                signed_order = active_client.create_order(order_args, active_options)
            return active_client.post_order(signed_order, OrderType.GTC)

        order = submit_with(client, options)

        order_id = order.get("orderID") or order.get("id") or str(order)
        log.info(f"Order placed: {bracket_label} @ {price:.3f} x{size:.0f} → {order_id}")

        return {"order_id": order_id, "success": True, "error": None}

    except Exception as e:
        msg = str(e)
        if _is_signature_error(msg):
            proxy_wallet = _normalize_wallet_address(os.getenv("PROXY_WALLET", ""))
            primary_sig = (
                _clob_signature_type
                if _clob_signature_type is not None
                else _primary_signature_type(proxy_wallet)
            )
            sig_candidates = [primary_sig] + _fallback_signature_types(primary_sig, proxy_wallet)
            option_variants = [("neg_risk=true", options), ("auto_neg_risk", None)]
            tried = {(primary_sig, "neg_risk=true")}
            last_err = e

            for sig_type in sig_candidates:
                try:
                    retry_client = get_clob_client(signature_type=sig_type, force_reinit=True)
                    if retry_client is None:
                        continue

                    for option_name, retry_options in option_variants:
                        if (sig_type, option_name) in tried:
                            continue
                        tried.add((sig_type, option_name))

                        try:
                            log.warning(
                                "Retrying order after signature error: "
                                f"sig_type={sig_type}, {option_name}"
                            )
                            order_retry = submit_with(retry_client, retry_options)
                            order_id = (
                                order_retry.get("orderID")
                                or order_retry.get("id")
                                or str(order_retry)
                            )
                            log.info(
                                f"Order placed after signature fallback: {bracket_label} @ {price:.3f} "
                                f"x{size:.0f} → {order_id} (sig_type={sig_type}, {option_name})"
                            )
                            return {"order_id": order_id, "success": True, "error": None}
                        except Exception as e2:
                            last_err = e2
                            log.warning(
                                f"Fallback failed (sig_type={sig_type}, {option_name}): {e2}"
                            )
                except Exception as e2:
                    last_err = e2
                    log.warning(f"Fallback client init failed (sig_type={sig_type}): {e2}")

            e = last_err

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
