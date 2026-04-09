# Kalshi API — Endpoints, Schemas & Gaps
*For the WNBA 2025 Backtest v1*

---

## Base URL

```
https://api.elections.kalshi.com/trade-api/v2
```

---

## Data Tier Architecture

Kalshi partitions all exchange data into a **live tier** and a **historical tier**.

| Boundary endpoint | `GET /historical/cutoff` |
|---|---|
| Response fields | `market_settled_ts`, `trades_created_ts`, `orders_updated_ts` |

- Markets settled **before** `market_settled_ts` → use `/historical/markets`
- Trades filled **before** `trades_created_ts` → use `/historical/trades`
- As of March 2026, `market_settled_ts = 2025-03-20T00:00:00Z` (covers only ~12 months back).
- **All 2025 WNBA markets** (settled May–October 2025) are **AFTER** the historical cutoff → use the **live tier**.
  - Trades: `GET /markets/trades?ticker=...`
  - Candles: `GET /series/{series_ticker}/markets/{ticker}/candlesticks`
  - Markets metadata: `GET /historical/markets` (historical tier still works for metadata)
- Kalshi is targeting **March 6, 2026** for removal of historical data from live endpoints.

---

## 1. Endpoints Used

### 1.1 Historical Cutoff

| | |
|---|---|
| **Path** | `GET /historical/cutoff` |
| **Auth** | None |
| **Params** | None |
| **Response** | `{ market_settled_ts, trades_created_ts, orders_updated_ts }` |

---

### 1.2 Market Metadata — Historical Tier

| | |
|---|---|
| **Path** | `GET /historical/markets` |
| **Auth** | None |
| **Pagination** | Cursor-based; `limit` max 1000 |
| **Filters** | `tickers` (comma-separated), `event_ticker`, `mve_filter` |
| **Note** | Does **not** support `series_ticker` filtering — must supply tickers explicitly |

**Response schema** (Market object, key fields):

| API field | Type | Normalized column | Notes |
|---|---|---|---|
| `ticker` | string | `market_ticker` | |
| `event_ticker` | string | `event_ticker` | |
| *(derived)* | — | `series_ticker` | `event_ticker.split("-")[0]` |
| `title` | string | `title` | Deprecated but still present |
| `yes_sub_title` | string | `yes_sub_title` | Team that YES resolves for |
| `no_sub_title` | string | `no_sub_title` | Team that NO resolves for |
| `status` | string | `status` | e.g. `finalized` |
| `market_type` | string | `market_type` | `binary` for game-winner markets |
| `open_time` | ISO datetime | `open_time` | UTC |
| `close_time` | ISO datetime | `close_time` | UTC |
| `expected_expiration_time` | ISO datetime | `expected_expiration_time` | Nullable |
| `settlement_ts` | ISO datetime | `settlement_ts` | Nullable; populated when settled |
| `rules_primary` | string | `rules_primary` | Full settlement rules text |
| `rules_secondary` | string | `rules_secondary` | Additional conditions |
| `expiration_value` | string | `expiration_value` | Winning team name at settlement |
| `can_close_early` | boolean | `can_close_early` | |
| `early_close_condition` | string | `early_close_condition` | Nullable |
| `result` | string | `result` | `"yes"` \| `"no"` \| `""` |
| `settlement_value_dollars` | FixedPointDollars | `settlement_value_dollars` | 1.0 for winner, 0.0 for loser |
| `last_price_dollars` | FixedPointDollars | `last_price_dollars` | Last traded price |
| `volume_fp` | FixedPointCount | `volume_fp` | Total contracts traded (lifetime) |
| `open_interest_fp` | FixedPointCount | `open_interest_fp` | Open contracts at close |
| `custom_strike` | object | `custom_strike` | JSON; contains SR team UUID for sports |

**Fields NOT available in market metadata:**
- `series_ticker` — not returned by API; derived from `event_ticker`
- Real-time order book state (historical)

---

### 1.3 Historical Market Candlesticks

| | |
|---|---|
| **Path** | `GET /historical/markets/{ticker}/candlesticks` |
| **Auth** | None |
| **Params (required)** | `start_ts` (int64 Unix), `end_ts` (int64 Unix), `period_interval` (1, 60, or 1440) |
| **Pagination** | None — all candles in range returned in one response |
| **Period interval** | `1` = 1 minute, `60` = 1 hour, `1440` = 1 day |
| **Note** | Time range limits not documented; ingestion code chunks into 30-day windows |

**Response schema** (MarketCandlestickHistorical):

| API field | Sub-field | Type | Normalized column |
|---|---|---|---|
| `end_period_ts` | — | int64 Unix | `end_period_ts` (UTC Timestamp) |
| `yes_bid` | `open_dollars` | FixedPointDollars | `yes_bid_open` |
| `yes_bid` | `high_dollars` | FixedPointDollars | `yes_bid_high` |
| `yes_bid` | `low_dollars` | FixedPointDollars | `yes_bid_low` |
| `yes_bid` | `close_dollars` | FixedPointDollars | `yes_bid_close` |
| `yes_ask` | `open_dollars` | FixedPointDollars | `yes_ask_open` |
| `yes_ask` | `high_dollars` | FixedPointDollars | `yes_ask_high` |
| `yes_ask` | `low_dollars` | FixedPointDollars | `yes_ask_low` |
| `yes_ask` | `close_dollars` | FixedPointDollars | `yes_ask_close` |
| `price` | `open_dollars` | FixedPointDollars | `price_open` |
| `price` | `high_dollars` | FixedPointDollars | `price_high` |
| `price` | `low_dollars` | FixedPointDollars | `price_low` |
| `price` | `close_dollars` | FixedPointDollars | `price_close` |
| `price` | `mean_dollars` | FixedPointDollars | `price_mean` |
| `price` | `previous_dollars` | FixedPointDollars | `price_previous` |
| `volume` | — | FixedPointCount | `volume` |
| `open_interest` | — | FixedPointCount | `open_interest` |

**Note on spread inference:**
Kalshi order books are bid-only. The ask can be inferred:
```
implied_yes_ask = 1.0 - best_no_bid_close
implied_no_ask  = 1.0 - best_yes_bid_close
```
`yes_ask_*` fields in candlesticks represent the best YES ask (limit sell offers),
which equals `1.0 - best_no_bid` at the inside.

---

### 1.4 Historical Trades

| | |
|---|---|
| **Path** | `GET /historical/trades` |
| **Auth** | None |
| **Params** | `ticker` (string), `min_ts` (int64), `max_ts` (int64) |
| **Pagination** | Cursor-based; `limit` max 1000 |

**Response schema** (Trade object):

| API field | Type | Normalized column | Notes |
|---|---|---|---|
| `trade_id` | string | `trade_id` | Unique trade identifier |
| `ticker` | string | `market_ticker` | |
| `created_time` | ISO datetime | `trade_ts` | UTC |
| `yes_price_dollars` | FixedPointDollars | `yes_price` | YES contract price [0,1] |
| `no_price_dollars` | FixedPointDollars | `no_price` | `1 - yes_price` |
| `count_fp` | FixedPointCount | `count` | Number of contracts traded |
| `taker_side` | enum | `taker_side` | `"yes"` or `"no"` |

---

### 1.5 Live Order Book (NOT USED — Historical Snapshots Unavailable)

| | |
|---|---|
| **Path** | `GET /markets/{ticker}/orderbook` |
| **Auth** | **Required** — RSA-PSS signed headers (`KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-SIGNATURE`, `KALSHI-ACCESS-TIMESTAMP`) |
| **Returns** | Current live order book only |
| **Historical** | **NOT AVAILABLE** |

**Structure** (live, current book only):
```json
{
  "orderbook_fp": {
    "yes_dollars": [["0.5500", "100.00"], ["0.5400", "250.00"], ...],
    "no_dollars":  [["0.4600", "80.00"],  ["0.4500", "300.00"], ...]
  }
}
```
Each price level is `[price_string, quantity_string]`.  Books are bid-only —
there are no ask entries.  Asks are implied from the opposite side.

**Decision:** `kalshi_orderbook_snapshots` table is **not implemented**.
No historical snapshot data exists for backtesting.

---

## 2. Output Tables

| Table | File | Source endpoint | Rows (est.) |
|---|---|---|---|
| `kalshi_markets` | `kalshi_markets.parquet` | `/historical/markets` | 596 |
| `kalshi_settlements` | `kalshi_settlements.parquet` | derived from markets | 596 |
| `kalshi_candles_1m` | `kalshi_candles_1m.parquet` | `/historical/markets/{t}/candlesticks` | ~2M+ |
| `kalshi_trades` | `kalshi_trades.parquet` | `/historical/trades` | ~200K+ |
| `kalshi_orderbook_snapshots` | *(not created)* | N/A | 0 |

---

## 3. Available vs Unavailable Fields

### Available ✓

| Dataset | Available fields |
|---|---|
| Markets | ticker, event_ticker, series_ticker (derived), title, yes/no sub_title, status, market_type, open/close/settlement timestamps, rules, result, settlement value, volume, open_interest, custom_strike |
| Settlements | All settlement fields: result, settlement_value_dollars, settlement_ts |
| Candles (1m) | OHLC for yes_bid, yes_ask, trade price; volume; open_interest; end_period_ts |
| Trades | trade_id, ticker, timestamp, yes_price, no_price, count, taker_side |

### Unavailable ✗

| Field | Reason |
|---|---|
| Historical order book snapshots | API only exposes live current book (auth required); no historical endpoint exists |
| `series_ticker` in market object | Not returned by API; derived from event_ticker prefix |
| Maker/taker identity in trades | Only taker_side is available; no counterparty info |
| Trade aggressor ID | Not exposed |

---

## 4. Data Type Conventions

| Kalshi type | Python/Pandas type | Notes |
|---|---|---|
| `FixedPointDollars` | `float64` | String like `"0.560000"`; parse with `float()` |
| `FixedPointCount` | `float64` | String like `"100.00"`; parse with `float()` |
| ISO datetime | `pd.Timestamp` (UTC) | All timestamps stored timezone-aware |
| Unix int64 timestamp | `pd.Timestamp` (UTC) | Converted via `pd.Timestamp(val, unit="s", tz="UTC")` |

---

## 5. Pagination

All list endpoints use **cursor-based pagination**:
1. First request: omit `cursor` param
2. Each response includes a `cursor` field
3. Pass the cursor in the next request
4. Stop when cursor is empty/absent or result list is empty
5. Max `limit` per page: **1000** (for both live and historical endpoints)

---

## 6. Authentication

| Endpoint category | Auth required |
|---|---|
| GET /historical/* (markets, candles, trades) | None |
| GET /markets, GET /markets/trades | None |
| GET /series/*/markets/*/candlesticks | None |
| GET /markets/{ticker}/orderbook | Yes — RSA-PSS signed headers |
| POST /portfolio/* (orders, fills) | Yes — RSA-PSS signed headers |

---

## 7. Rate Limits

Kalshi does not publish explicit rate limits for public (unauthenticated) read
endpoints in their official documentation.  The ingestion client enforces a
conservative **0.25-second minimum between requests** and exponential backoff
on HTTP 429 responses.

---

## 8. Known Quirks

- **SR date offset**: SportRadar stores game dates in UTC, shifting late-night
  games (e.g. 8 pm PT) by +1 day.  Kalshi event tickers encode local game date.
  Matching uses ±1 day window with tiebreak (prefer SR date ≥ ticker date).

- **Kalshi order book is bid-only**: `yes_dollars` and `no_dollars` arrays both
  contain *bids only*.  The implied ask on each side is `1.0 − best_bid_opposite`.

- **IND vs MIN Jul 1 market (KXWNBAGAME-25JUL01INDMIN)**: Exists in Kalshi
  (finalized, winner = Indiana) but has no corresponding game in SportRadar.
  Likely cancelled/rescheduled and absent from SR data.

- **All-Star game (KXWNBAGAME-25JUL19COLCLA)**: Team Collier vs Team Clark;
  no SR `team_id` or `game_id` can be assigned.

- **Historical candlesticks**: The path uses `/historical/markets/{ticker}/candlesticks`,
  NOT `/series/{series_ticker}/markets/{ticker}/candlesticks` (live only).
