"""
Polymarket API -- WNBA Game Outcome Market Endpoint Explorer
=============================================================
Discovers and validates all WNBA game-winner (moneyline) market endpoints
on Polymarket. Outputs market IDs, slugs, prices, and CLOB token IDs.

APIs used:
  Gamma API  https://gamma-api.polymarket.com   (market discovery)
  CLOB API   https://clob.polymarket.com         (live orderbook / pricing)

NOTES (validated 2026-03-20):
  - WNBA tag_id = 100254
  - sportsMarketType = 'moneyline' for individual game markets, None for prop/futures
  - The 'order' and 'sports_market_types' query params cause 422 on /markets -- omit them
  - WNBA season runs ~May-Oct; markets will be closed during the off-season

Run with:
  /c/Users/arius/anaconda3/envs/kalshi-wnba/python.exe notebooks/polymarket/01_explore_wnba_endpoints.py
"""

import requests
import json

GAMMA     = "https://gamma-api.polymarket.com"
CLOB      = "https://clob.polymarket.com"
WNBA_TAG  = 100254
PAGE_SIZE = 100

# ──────────────────────────────────────────────────────────────────────────────
# 1. Verify WNBA tag via /sports
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("1. SPORTS METADATA  ->  GET /sports")
print("=" * 70)
resp = requests.get(f"{GAMMA}/sports")
resp.raise_for_status()
sports = resp.json()

wnba_entries = [s for s in sports if "wnba" in json.dumps(s).lower()]
print(f"WNBA entries: {len(wnba_entries)}")
for e in wnba_entries:
    print(" ", e)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Active WNBA events (nested child markets, good for in-season)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. ACTIVE WNBA EVENTS  ->  GET /events?tag_id=100254&active=true&closed=false")
print("=" * 70)

active_events = []
offset = 0
while True:
    r = requests.get(f"{GAMMA}/events", params=dict(
        tag_id = WNBA_TAG, active = "true", closed = "false",
        limit  = PAGE_SIZE, offset = offset,
    ))
    r.raise_for_status()
    batch = r.json()
    if not batch:
        break
    active_events.extend(batch)
    print(f"  offset={offset}: {len(batch)} events")
    if len(batch) < PAGE_SIZE:
        break
    offset += PAGE_SIZE

print(f"\nTotal active WNBA events: {len(active_events)}")
for ev in active_events[:10]:
    print(f"  event_id={ev.get('id')}  slug={ev.get('slug')}  title={ev.get('title')}")
    for mkt in ev.get("markets", []):
        print(f"    +- conditionId={mkt.get('conditionId')}  "
              f"type={mkt.get('sportsMarketType')}  "
              f"q={mkt.get('question')}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. All WNBA markets (paginated, includes closed historical)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. ALL WNBA MARKETS (incl. historical)  ->  GET /markets?tag_id=100254")
print("=" * 70)
print("  NOTE: 'order' and 'sports_market_types' params cause 422 -- omitted")

all_markets = []
offset = 0
while True:
    r = requests.get(f"{GAMMA}/markets", params=dict(
        tag_id = WNBA_TAG, limit = PAGE_SIZE, offset = offset,
    ))
    if not r.ok:
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
        break
    batch = r.json()
    if not batch:
        break
    all_markets.extend(batch)
    print(f"  offset={offset}: {len(batch)} markets")
    if len(batch) < PAGE_SIZE:
        break
    offset += PAGE_SIZE

# Split by type
moneyline_markets = [m for m in all_markets if m.get("sportsMarketType") == "moneyline"]
other_markets     = [m for m in all_markets if m.get("sportsMarketType") != "moneyline"]

print(f"\nTotal WNBA markets:          {len(all_markets)}")
print(f"  sportsMarketType=moneyline: {len(moneyline_markets)}  (individual game outcomes)")
print(f"  sportsMarketType=None:      {len(other_markets)}  (props / season futures)")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Game-winner moneyline markets -- summary table
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. MONEYLINE MARKET SAMPLE (most recent 20)")
print("=" * 70)

header = f"{'gameStartTime':<22}  {'active':<6}  {'closed':<6}  {'question':<30}  {'outcomes':<30}  prices"
print(header)
print("-" * len(header))
for m in moneyline_markets[:20]:
    outcomes = m.get("outcomes", "[]")
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    prices = m.get("outcomePrices", "[]")
    if isinstance(prices, str):
        prices = json.loads(prices)
    print(f"{str(m.get('gameStartTime','')):<22}  "
          f"{str(m.get('active','')):<6}  "
          f"{str(m.get('closed','')):<6}  "
          f"{str(m.get('question',''))[:29]:<30}  "
          f"{str(outcomes):<30}  "
          f"{prices}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Keyword search (useful when tag is uncertain)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. PUBLIC KEYWORD SEARCH  ->  GET /public-search?q=WNBA")
print("=" * 70)

r = requests.get(f"{GAMMA}/public-search", params=dict(
    q = "WNBA", keep_closed_markets = 0, limit_per_type = 50,
))
r.raise_for_status()
sr = r.json()
print(f"Events returned: {len(sr.get('events', []))}  |  pagination: {sr.get('pagination')}")
for ev in sr.get("events", [])[:5]:
    print(f"  {ev.get('slug')}  --  {ev.get('title')}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. CLOB pricing for the first moneyline market (demonstrates pricing endpoints)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. LIVE CLOB PRICING  (first moneyline market found)")
print("=" * 70)

sample_market = next(
    (m for m in moneyline_markets if m.get("active") and not m.get("closed")),
    moneyline_markets[0] if moneyline_markets else None,
)

if sample_market:
    clob_ids = sample_market.get("clobTokenIds", "[]")
    if isinstance(clob_ids, str):
        clob_ids = json.loads(clob_ids)
    outcomes = sample_market.get("outcomes", "[]")
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)

    print(f"Market : {sample_market.get('question')}")
    print(f"conditionId: {sample_market.get('conditionId')}")
    print(f"gameId     : {sample_market.get('gameId')}")
    print(f"active={sample_market.get('active')}  closed={sample_market.get('closed')}")

    for outcome, token_id in zip(outcomes, clob_ids):
        print(f"\n  Outcome '{outcome}'  token_id={token_id}")
        for path, extra_params in [
            ("price",            {"side": "BUY"}),
            ("midpoint",         {}),
            ("spread",           {}),
            ("last-trade-price", {}),
        ]:
            rr = requests.get(f"{CLOB}/{path}",
                              params={"token_id": token_id, **extra_params})
            status = f"HTTP {rr.status_code}" if not rr.ok else str(rr.json())
            print(f"    /{path:<20} -> {status}")

        rr = requests.get(f"{CLOB}/book", params={"token_id": token_id})
        if rr.ok:
            book = rr.json()
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            print(f"    /book                 -> "
                  f"best_bid={bids[0] if bids else 'N/A'}  "
                  f"best_ask={asks[0] if asks else 'N/A'}  "
                  f"last_trade={book.get('last_trade_price')}")
        else:
            print(f"    /book                 -> HTTP {rr.status_code}")
else:
    print("No moneyline markets found.")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Endpoint reference
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VALIDATED ENDPOINT REFERENCE")
print("=" * 70)

endpoints = [
    ("GET", f"{GAMMA}/sports",
     "All sports metadata; WNBA entry has tag_id=100254 in 'tags' field"),
    ("GET", f"{GAMMA}/events?tag_id=100254&active=true&closed=false",
     "Active WNBA events with nested child markets (primary in-season query)"),
    ("GET", f"{GAMMA}/events?tag_id=100254",
     "All WNBA events including historical (paginate with limit/offset)"),
    ("GET", f"{GAMMA}/markets?tag_id=100254",
     "All WNBA markets flat list (paginate; DO NOT add 'order' or 'sports_market_types' -- 422)"),
    ("GET", f"{GAMMA}/markets?tag_id=100254&limit=100&offset=0",
     "Paginated WNBA markets; filter sportsMarketType='moneyline' client-side for game-winners"),
    ("GET", f"{GAMMA}/public-search?q=WNBA&keep_closed_markets=0",
     "Keyword search; useful fallback; 372+ results, paginated via 'page' param"),
    ("GET", f"{CLOB}/price?token_id={{clobTokenId}}&side=BUY",
     "Live BUY price for one outcome token (from market.clobTokenIds[i])"),
    ("GET", f"{CLOB}/midpoint?token_id={{clobTokenId}}",
     "Midpoint probability for one outcome token"),
    ("GET", f"{CLOB}/spread?token_id={{clobTokenId}}",
     "Bid-ask spread for one outcome token"),
    ("GET", f"{CLOB}/last-trade-price?token_id={{clobTokenId}}",
     "Last trade price for one outcome token"),
    ("GET", f"{CLOB}/book?token_id={{clobTokenId}}",
     "Full L2 orderbook (bids/asks arrays) for one outcome token"),
]
for method, url, desc in endpoints:
    print(f"\n  {method}  {url}")
    print(f"    -> {desc}")

print(f"\n\nKey market fields for game-winner markets:")
key_fields = {
    "conditionId":       "Unique hex ID for the CTF contract condition",
    "question":          "Market question (e.g. 'Aces vs Liberty')",
    "sportsMarketType":  "'moneyline' for individual game markets; None for props/futures",
    "gameId":            "Sports game identifier hex",
    "gameStartTime":     "Scheduled tip-off time (UTC)",
    "endDateIso":        "Market resolution date (YYYY-MM-DD)",
    "active":            "True if market is accepting orders",
    "closed":            "True if market is resolved",
    "outcomes":          "JSON array of team names e.g. ['Aces', 'Liberty']",
    "outcomePrices":     "Implied probabilities e.g. ['0.45', '0.55']",
    "clobTokenIds":      "JSON array [yes_token_id, no_token_id] -- pass to CLOB endpoints",
}
for field, desc in key_fields.items():
    print(f"  {field:<20} {desc}")

print("\nDone.")
