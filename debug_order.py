#!/usr/bin/env python3
"""
Diagnostic script: debug invalid signature error for Polymarket orders.
Run on VPS: cd /opt/weather-trader && python3 debug_order.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("POLYMARKET ORDER SIGNATURE DIAGNOSTIC")
print("=" * 60)

# 1. Check versions
print("\n--- 1. Package versions ---")
import py_clob_client, py_order_utils
try:
    print(f"  py-clob-client: {py_clob_client.__version__}")
except:
    import pkg_resources
    print(f"  py-clob-client: {pkg_resources.get_distribution('py-clob-client').version}")
try:
    print(f"  py-order-utils: {py_order_utils.__version__}")
except:
    import pkg_resources
    print(f"  py-order-utils: {pkg_resources.get_distribution('py-order-utils').version}")

# 2. Check env vars
print("\n--- 2. Environment variables ---")
private_key = os.getenv("POLYGON_PRIVATE_KEY", "")
proxy_wallet = os.getenv("PROXY_WALLET", "")
signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "2" if proxy_wallet else "0"))
print(f"  POLYGON_PRIVATE_KEY: {'set (' + str(len(private_key)) + ' chars)' if private_key else 'MISSING'}")
print(f"  PROXY_WALLET: {proxy_wallet if proxy_wallet else 'MISSING'}")
print(f"  POLY_SIGNATURE_TYPE: {signature_type}")

if not private_key or not proxy_wallet:
    print("\n  ERROR: Missing env vars. Cannot continue.")
    sys.exit(1)

# 3. Derive signer address from private key
print("\n--- 3. Signer address derivation ---")
from py_clob_client.signer import Signer
signer = Signer(private_key, 137)
signer_address = signer.address()
print(f"  Signer (EOA): {signer_address}")
print(f"  Funder (proxy): {proxy_wallet}")
print(f"  Same address? {signer_address.lower() == proxy_wallet.lower()} (should be False for Magic wallets)")

# 4. Initialize CLOB client
print("\n--- 4. CLOB client initialization ---")
from py_clob_client.client import ClobClient

client = ClobClient(
    "https://clob.polymarket.com",
    key=private_key,
    chain_id=137,
    signature_type=signature_type,
    funder=proxy_wallet,
)

# Check builder config
print(f"  Builder sig_type: {client.builder.sig_type}")
print(f"  Builder funder: {client.builder.funder}")
print(f"  Builder signer addr: {client.builder.signer.address()}")

# 5. Derive API creds
print("\n--- 5. API credentials ---")
try:
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    print(f"  API key: {creds.api_key[:16]}...")
    print(f"  API creds set: OK")
except Exception as e:
    print(f"  ERROR deriving creds: {e}")
    sys.exit(1)

# 6. Test L2 auth (cancel_all is L2)
print("\n--- 6. L2 auth test (cancel_all) ---")
try:
    result = client.cancel_all()
    print(f"  cancel_all(): {result}")
    print(f"  L2 auth: OK")
except Exception as e:
    print(f"  L2 auth FAILED: {e}")

# 7. Get a real token_id from a temperature market
print("\n--- 7. Fetching a temperature market token ---")
import requests
try:
    # London temperature market
    resp = requests.get("https://gamma-api.polymarket.com/events", params={
        "tag": "weather",
        "limit": 5,
        "active": "true",
    })
    events = resp.json()
    
    token_id = None
    market_info = None
    for event in events:
        for market in event.get("markets", []):
            if market.get("enableOrderBook") and market.get("active"):
                for token in market.get("clobTokenIds", "").split(","):
                    if token.strip():
                        token_id = token.strip()
                        market_info = {
                            "question": market.get("question", "")[:60],
                            "neg_risk": market.get("negRisk"),
                            "min_size": market.get("orderMinSize"),
                            "active": market.get("active"),
                        }
                        break
            if token_id:
                break
        if token_id:
            break
    
    if token_id:
        print(f"  Token ID: {token_id[:30]}...")
        print(f"  Market: {market_info}")
    else:
        print("  No active weather market found, using a known token_id")
        # Fallback - we'll need one
except Exception as e:
    print(f"  Error fetching market: {e}")

if not token_id:
    print("  Cannot continue without a token_id")
    sys.exit(1)

# 8. Check neg_risk resolution
print("\n--- 8. Neg risk resolution ---")
try:
    auto_neg_risk = client.get_neg_risk(token_id)
    print(f"  Auto-resolved neg_risk for token: {auto_neg_risk}")
except Exception as e:
    print(f"  get_neg_risk() failed: {e}")
    auto_neg_risk = None

# 9. Create a signed order (don't post yet)
print("\n--- 9. Create signed order ---")
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY

order_args = OrderArgs(
    price=0.01,  # Very low price - won't fill
    size=5.0,     # Minimum size
    side=BUY,
    token_id=token_id,
)

# Try with explicit neg_risk=True
print("\n  9a. With neg_risk=True:")
try:
    options = PartialCreateOrderOptions(tick_size="0.01", neg_risk=True)
    signed_order = client.create_order(order_args, options)
    order_dict = signed_order.dict()
    print(f"    maker: {order_dict.get('maker')}")
    print(f"    signer: {order_dict.get('signer')}")
    print(f"    signatureType: {order_dict.get('signatureType')}")
    print(f"    side: {order_dict.get('side')}")
    print(f"    makerAmount: {order_dict.get('makerAmount')}")
    print(f"    takerAmount: {order_dict.get('takerAmount')}")
    print(f"    signature: {order_dict.get('signature', '')[:30]}...")
    print(f"    tokenId: {str(order_dict.get('tokenId', ''))[:30]}...")
    
    # Check maker vs funder
    maker = order_dict.get('maker', '').lower()
    expected_funder = proxy_wallet.lower()
    print(f"    maker matches funder? {maker == expected_funder}")
    
    # Check signer
    order_signer = order_dict.get('signer', '').lower()
    expected_signer = signer_address.lower()
    print(f"    signer matches EOA? {order_signer == expected_signer}")
    
except Exception as e:
    print(f"    create_order FAILED: {e}")
    import traceback
    traceback.print_exc()
    signed_order = None

# 10. Try posting the order
if signed_order:
    print("\n--- 10. Post order (neg_risk=True, price=0.01) ---")
    from py_clob_client.clob_types import OrderType
    try:
        resp = client.post_order(signed_order, OrderType.GTC)
        print(f"  SUCCESS: {resp}")
    except Exception as e:
        print(f"  FAILED: {e}")
        
        # Try with auto-resolved neg_risk (no explicit options)
        print("\n--- 10b. Retry WITHOUT explicit neg_risk (let client auto-resolve) ---")
        try:
            signed_order2 = client.create_order(order_args)
            resp2 = client.post_order(signed_order2, OrderType.GTC)
            print(f"  SUCCESS with auto-resolve: {resp2}")
        except Exception as e2:
            print(f"  ALSO FAILED: {e2}")

# 11. Try with a different well-known market (non-weather, binary)
print("\n--- 11. Test with a known binary market ---")
try:
    # Fetch any active non-weather market
    resp = requests.get("https://gamma-api.polymarket.com/markets", params={
        "limit": 5,
        "active": "true",
        "closed": "false",
    })
    markets = resp.json()
    
    binary_token = None
    for m in markets:
        if m.get("enableOrderBook") and not m.get("negRisk"):
            tokens = m.get("clobTokenIds", "").split(",")
            if tokens and tokens[0].strip():
                binary_token = tokens[0].strip()
                print(f"  Found non-negRisk market: {m.get('question', '')[:50]}")
                break
    
    if binary_token:
        order_args_binary = OrderArgs(
            price=0.01,
            size=5.0,
            side=BUY,
            token_id=binary_token,
        )
        signed_binary = client.create_order(order_args_binary)
        try:
            resp_binary = client.post_order(signed_binary, OrderType.GTC)
            print(f"  Non-negRisk order: SUCCESS → {resp_binary}")
            print("  *** This means negRisk signing is the issue ***")
        except Exception as e:
            print(f"  Non-negRisk order also FAILED: {e}")
            print("  *** This means the issue is NOT negRisk-specific ***")
    else:
        print("  No non-negRisk market found")
        
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
