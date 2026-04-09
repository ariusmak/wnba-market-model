# Polymarket Ingestion Specification
## WNBA 2025 Backtest — v1

---

## 1. Endpoints Found

### Gamma API — `https://gamma-api.polymarket.com`
| Endpoint | Method | Auth | Used For |
|---|---|---|---|
| `/sports` | GET | None | Confirm WNBA `tag_id=100254` |
| `/events?tag_id=100254` | GET | None | All WNBA events (paginated) |
| `/markets?tag_id=100254` | GET | None | All WNBA markets flat list (paginated) |
| `/markets?condition_ids=0x...` | GET | None | Single market lookup by condition ID |
| `/sports/market-types` | GET | None | Valid sports market type strings |

### CLOB API — `https://clob.polymarket.com`
| Endpoint | Method | Auth | Used For |
|---|---|---|---|
| `/prices-history` | GET | None | 1-min price history per token |
| `/book` | GET | None | Live order book snapshot (current only) |
| `/books` | POST | None | Bulk live order books |
| `/price` | GET | None | Live best price (BUY or SELL side) |
| `/midpoint` | GET | None | Live midpoint probability |
| `/spread` | GET | None | Live bid-ask spread |
| `/last-trade-price` | GET | None | Most recent executed price |
| `/tick-size` | GET | None | Minimum tick size |

### Data API — `https://data-api.polymarket.com`
| Endpoint | Method | Auth | Used For |
|---|---|---|---|
| `/trades` | GET | None | Public trade history per market or token |
| `/oi` | GET | None | Open interest per market |
| `/positions` | GET | None | Positions per user (user required) |
| `/activity` | GET | None | Activity feed per user (user required) |

---

## 2. Available vs Unavailable Fields

### Available

| Required Field | Source | API Field |
|---|---|---|
| Market ID | Gamma `/markets` | `id` (integer) |
| Condition ID | Gamma `/markets` | `conditionId` (0x hex) |
| Event ID | Gamma `/events` | `id` |
| Slug | Gamma `/markets` | `slug` |
| Question/title | Gamma `/markets` | `question` |
| Description/rules | Gamma `/markets` | `description` / `resolutionSource` |
| Active flag | Gamma `/markets` | `active` (bool) |
| Closed flag | Gamma `/markets` | `closed` (bool) |
| Archived flag | Gamma `/markets` | `archived` (bool) |
| Accepting orders flag | Gamma `/markets` | `acceptingOrders` (bool) |
| Enable order book flag | Gamma `/markets` | `enableOrderBook` (bool) |
| Clear book on start | Gamma `/markets` | `clearBookOnStart` (bool) |
| Market start timestamp | Gamma `/markets` | `startDate` (ISO 8601) |
| Market end timestamp | Gamma `/markets` | `endDate` (ISO 8601) |
| Game start timestamp | Gamma `/markets` | `gameStartTime` (ISO 8601, sports only) |
| Actual close timestamp | Gamma `/markets` | `closedTime` (ISO 8601 string) |
| CLOB token IDs | Gamma `/markets` | `clobTokenIds` (JSON array string) |
| Outcome labels | Gamma `/markets` | `outcomes` (JSON array string) |
| Outcome prices | Gamma `/markets` | `outcomePrices` (JSON array string) |
| Sports market type | Gamma `/markets` | `sportsMarketType` ('moneyline' \| null) |
| Sports game ID | Gamma `/markets` | `gameId` (hex string) |
| Tags / category | Gamma `/markets` | `tags` (objects or comma-delimited IDs) |
| Neg risk flag | Gamma `/markets` | `negRisk` (bool) |
| Volume / liquidity | Gamma `/markets` | `volumeNum`, `liquidityNum`, `volume24hr` |
| Price history (token-level) | CLOB `/prices-history` | `history[].t`, `history[].p` |
| Live order book | CLOB `/book` | `bids[]`, `asks[]`, `tick_size` |
| Live best bid/ask | CLOB `/price` | `price` (side=BUY → ask, side=SELL → bid) |
| Live midpoint | CLOB `/midpoint` | `mid` |
| Live spread | CLOB `/spread` | `spread` |
| Last trade price | CLOB `/last-trade-price` | `price` |
| Tick size | CLOB `/tick-size` | `minimum_tick_size` |
| Public trade history | Data `/trades` | `timestamp`, `price`, `size`, `side`, `asset`, `conditionId` |
| Settlement / winning outcome | Derived from Gamma `outcomePrices` | price=1.0 → winner |
| Settlement timestamp (proxy) | Derived from Gamma `closedTime` → `endDate` | — |

### Unavailable

| Required Field | Status |
|---|---|
| `resolvedAt` / `settlementTs` | **Not present** in Gamma schema. Use `closedTime` (if populated) then `endDate` as proxy. |
| Historical order book snapshots | **Not available** via any REST endpoint. Only live (current) snapshots via CLOB `/book`. WebSocket capture needed for future real-time collection. |
| Resolution metadata / notes | Not present. `resolvedBy` is a string authority only. |
| Trade IDs | Not exposed in Data API `/trades` response (transaction hash available instead). |
| CLOB order-level detail | CLOB `/trades` requires L2 auth. Maker-order breakdown not available publicly. |

---

## 3. Price History — Token-level or Market-level?

**Token-level.** The `/prices-history` endpoint takes one `token_id` (CLOB asset ID) at a time. Each market has two tokens (e.g., "Aces" and "Liberty"). Both must be fetched separately.

- `market` query parameter = CLOB token ID (confusingly named; this is NOT the condition ID)
- `fidelity` = minutes per bucket (1 = 1-minute candles, 60 = hourly)
- `startTs` / `endTs` = Unix timestamps in seconds

**Important quirk:** Using `interval=max` or `interval=all` returns empty data for resolved/closed markets. Always use explicit `startTs`/`endTs` instead. Chunking into ≤15-day windows is recommended. In practice (validated on 566 tokens from 283 WNBA 2025 markets) explicit timestamps return data for all resolved markets.

**Validated results:** 1,417,330 price points from 566 tokens; 0 empty responses.

---

## 4. Historical Order Book Snapshots

**Not available.**

The CLOB `/book` endpoint returns only the current live state of the order book. There is no REST endpoint for historical snapshots at any past timestamp.

**Implications for backtest:**
- Order book state at any past moment cannot be reconstructed from the API.
- Live snapshots can be collected going forward for the 2026 season.
- The `polymarket_orderbooks_live_snapshots` table captures only same-moment snapshots taken during ingestion.
- Spread/midpoint at historical times can be approximated from `polymarket_prices_history` (midpoint ≈ price, since prices are probabilities) or from `polymarket_trades` (last traded price).

---

## 5. Pregame Trading Window — Exact Field

**Use `game_start_ts` as the end of the pregame trading window.**

Rationale:
- `clearBookOnStart = true` on all WNBA game markets means all resting limit orders are cancelled at `game_start_ts` (the scheduled tip-off time). This is the hard pre/live boundary.
- `end_ts` (endDate) is the market resolution deadline, not the game start.
- `accepting_orders_flag` stays true after game start (live betting continues).
- `closed_flag` goes true only when the market resolves (after the game ends).

**Pregame window definition:**
```
market open:   start_ts (market creation)
pregame close: game_start_ts (order book cleared; no resting orders survive)
live window:   game_start_ts → closed_ts (in-game trading)
settled:       closed_ts (resolution)
```

**Caveat:** `game_start_ts` is the SCHEDULED tip-off. Actual game starts can shift (Polymarket documentation notes this). If precision matters, use the earliest trade timestamp after the order book clear as the actual game start proxy.

---

## 6. Sports Market Behavior at Game Start

From Polymarket documentation (confirmed):
> "Any outstanding limit orders are automatically cancelled once the game begins, clearing the entire order book at the official start time."
> "Sports markets enforce a 3-second delay on the placement of marketable orders."

The `clearBookOnStart` boolean in market metadata encodes this behavior:
- `clearBookOnStart = true` → all limit orders cancelled at game start; live trading continues
- The market does NOT close at game start; `acceptingOrders` remains true

---

## 7. Normalized Table Schemas

### polymarket_events
| Column | Type | Source |
|---|---|---|
| `event_id` | str | `events[].id` |
| `event_slug` | str | `events[].slug` |
| `event_ticker` | str | `events[].ticker` |
| `event_title` | str | `events[].title` |
| `tags` | str (pipe-delimited) | `events[].tags` |
| `start_ts` | Timestamp (UTC) | `events[].startDate` |
| `end_ts` | Timestamp (UTC) | `events[].endDate` |
| `active_flag` | bool | `events[].active` |
| `closed_flag` | bool | `events[].closed` |
| `archived_flag` | bool | `events[].archived` |
| `neg_risk_flag` | bool | `events[].negRisk` |
| `volume` | float | `events[].volume` |
| `liquidity` | float | `events[].liquidity` |
| `open_interest` | float | `events[].openInterest` |

### polymarket_markets
| Column | Type | Source | Notes |
|---|---|---|---|
| `market_id` | str | `markets[].id` | Gamma integer ID |
| `condition_id` | str | `markets[].conditionId` | Primary join key (0x hex) |
| `market_slug` | str | `markets[].slug` | |
| `question` | str | `markets[].question` | |
| `description` | str | `markets[].description` | May be empty |
| `resolved_by` | str | `markets[].resolvedBy` | Resolution authority |
| `active_flag` | bool | `markets[].active` | |
| `closed_flag` | bool | `markets[].closed` | |
| `archived_flag` | bool | `markets[].archived` | |
| `accepting_orders_flag` | bool | `markets[].acceptingOrders` | |
| `enable_order_book_flag` | bool | `markets[].enableOrderBook` | |
| `clear_book_on_start` | bool | `markets[].clearBookOnStart` | True for sports markets |
| `neg_risk_flag` | bool | `markets[].negRisk` | |
| `sports_market_type` | str\|None | `markets[].sportsMarketType` | 'moneyline' or null |
| `game_id` | str | `markets[].gameId` | Sports game hex ID |
| `start_ts` | Timestamp (UTC) | `markets[].startDate` | Market creation / open |
| `end_ts` | Timestamp (UTC) | `markets[].endDate` | Resolution deadline |
| `game_start_ts` | Timestamp (UTC) | `markets[].gameStartTime` | **Use as pregame boundary** |
| `closed_ts` | Timestamp (UTC) | `markets[].closedTime` | Actual resolution time |
| `outcomes` | str (pipe-delimited) | `markets[].outcomes` | e.g. "Aces\|Liberty" |
| `outcome_prices` | str (pipe-delimited) | `markets[].outcomePrices` | e.g. "0.45\|0.55" |
| `clob_token_ids` | str (pipe-delimited) | `markets[].clobTokenIds` | token0\|token1 |
| `volume` | float | `markets[].volumeNum` | |
| `liquidity` | float | `markets[].liquidityNum` | |
| `volume_24hr` | float | `markets[].volume24hr` | |
| `last_trade_price` | float | `markets[].lastTradePrice` | |
| `best_bid` | float | `markets[].bestBid` | Stale; use CLOB for live |
| `best_ask` | float | `markets[].bestAsk` | Stale; use CLOB for live |

### polymarket_tokens
| Column | Type | Source | Notes |
|---|---|---|---|
| `condition_id` | str | parent market | Join to polymarket_markets |
| `market_id` | str | parent market | |
| `token_id` | str | `markets[].clobTokenIds[i]` | Pass to CLOB endpoints |
| `outcome_label` | str | `markets[].outcomes[i]` | e.g. "Aces", "Yes" |
| `token_index` | int | derived | 0=first outcome, 1=second |
| `outcome_price` | float | `markets[].outcomePrices[i]` | Implied prob at fetch time |

### polymarket_prices_history
| Column | Type | Source | Notes |
|---|---|---|---|
| `token_id` | str | fetch param / embedded | Join to polymarket_tokens |
| `condition_id` | str | derived via token_map | |
| `ts` | Timestamp (UTC) | `history[].t` (Unix seconds) | Bucket start |
| `price` | float [0,1] | `history[].p` | 1-min default (fidelity=1) |

### polymarket_best_quotes (live only)
| Column | Type | Source | Notes |
|---|---|---|---|
| `token_id` | str | | |
| `condition_id` | str | derived | |
| `snapshot_ts` | Timestamp (UTC) | ingestion time | |
| `best_bid` | float | CLOB `/price?side=SELL` | |
| `best_ask` | float | CLOB `/price?side=BUY` | |
| `midpoint` | float | CLOB `/midpoint` | |
| `spread` | float | CLOB `/spread` | |
| `last_trade_price` | float | CLOB `/last-trade-price` | |
| `tick_size` | float | CLOB `/tick-size` | |

### polymarket_orderbooks_live_snapshots (live only)
| Column | Type | Source | Notes |
|---|---|---|---|
| `token_id` | str | | |
| `condition_id` | str | derived | |
| `snapshot_ts` | Timestamp (UTC) | ingestion time | |
| `side` | str | "bid" \| "ask" | |
| `price` | float | `book.bids[].price` / `asks[].price` | |
| `size` | float | `book.bids[].size` / `asks[].size` | USDC |
| `level` | int | derived | 1=best, 2=second-best, ... |
| `tick_size` | float | `book.tick_size` | |
| `neg_risk` | bool | `book.neg_risk` | |

### polymarket_trades
| Column | Type | Source | Notes |
|---|---|---|---|
| `token_id` | str | Data API `asset` | CLOB token (outcome) |
| `condition_id` | str | Data API `conditionId` | |
| `trade_ts` | Timestamp (UTC) | Data API `timestamp` (Unix seconds) | |
| `price` | float [0,1] | Data API `price` | |
| `size` | float | Data API `size` | USDC notional |
| `side` | str | Data API `side` | "BUY" \| "SELL" (taker) |
| `outcome` | str | Data API `outcome` | e.g. "Yes", "Aces" |
| `outcome_index` | int | Data API `outcomeIndex` | 0 or 1 |
| `tx_hash` | str | Data API `transactionHash` | On-chain ID |

### polymarket_settlements
| Column | Type | Source | Notes |
|---|---|---|---|
| `condition_id` | str | market `conditionId` | |
| `market_id` | str | market `id` | |
| `resolved_flag` | bool | market `closed` | |
| `winning_outcome` | str | derived from `outcomePrices` | outcome where price=1.0 |
| `settlement_ts` | Timestamp (UTC) | `closedTime` → `endDate` fallback | Proxy only; no true resolvedAt |
| `final_status` | str | derived | "finalized" \| "open" |
| `outcomes` | str (pipe-delimited) | market `outcomes` | |
| `outcome_prices` | str (pipe-delimited) | market `outcomePrices` | "1\|0" or "0\|1" |

---

## 8. Polymarket vs Kalshi Comparison

| Dimension | Kalshi | Polymarket |
|---|---|---|
| Market structure | One contract per team per game (Yes/No binary) | One market per game with 2 outcome tokens |
| Unique contracts | 596 (2 per game × 298 games) | 283 (1 per game) |
| Outcome tokens | 1 per contract | 2 per market (Yes/No or Team A/Team B) |
| Price history | 1-min OHLC candlesticks (bid/ask + trade) | 1-min midpoint price per token |
| Trade history | fill-level OHLC (yes_price, no_price, count) | flat fills (price, size, side, tx_hash) |
| Settlement field | `result` ("yes"/"no") + `settlement_value_dollars` | Derived from `outcomePrices` (1.0=winner) |
| Settlement timestamp | `settlement_ts` (explicit) | `closedTime` (often populated) or `endDate` fallback |
| Historical OB | Not available | Not available |
| Price history availability | All resolved markets ✓ | All resolved markets ✓ (explicit ts workaround) |
| Total trades (2025 WNBA) | ~551K | ~123K |
| Total price points (2025 WNBA) | ~497K 1-min candles | ~1.4M 1-min midpoints |
| Game start field | Not explicit; infer from market title | `gameStartTime` (explicit) |
| Book clear at game start | Not documented | `clearBookOnStart=true` (explicit) |

---

## 9. Rate Limits

| API / Endpoint | Published Limit | Code Delay Used |
|---|---|---|
| Gamma `/markets` | 300 req/10 s | 0.05 s between calls |
| Gamma `/events` | 500 req/10 s | 0.05 s between calls |
| CLOB `/book`, `/price`, `/midpoint` | 1500 req/10 s | 0.05 s between calls |
| Data API `/trades` | 200 req/10 s | 0.06 s between calls |

---

## 10. Assumptions and Gaps

1. **Settlement timestamp**: `closedTime` is used where available. For the WNBA 2025 dataset, `closedTime` was populated for all resolved markets. `endDate` is used as a fallback but may reflect the scheduled resolution deadline, not the actual settlement moment.

2. **Winning outcome derivation**: `outcomePrices == "1"` is treated as the winning outcome. This is standard Polymarket behavior (binary: winner resolves to 1, loser to 0). Ambiguous prices (not exactly 0 or 1) are treated as unresolved.

3. **Price history `fidelity=1`**: 1-minute resolution. Coarser resolution (e.g., `fidelity=60` for hourly) would reduce the 1.4M rows by 60x if storage is a concern.

4. **`order` and `sports_market_types` query params**: These cause HTTP 422 on the Gamma `/markets` endpoint. Client-side filtering by `sportsMarketType == 'moneyline'` is used instead.

5. **Data API offset ceiling**: Hard limit of 10,000 records per query window. Not hit for any individual WNBA 2025 game market (max observed was ~1,429 trades per market).

6. **Historical order books**: Confirmed unavailable. No workaround exists. The spread visible in trade data (last_trade_price series) is the best available proxy for historical book state.

7. **6 unmatched markets** (from Sportradar matching): 1 All-Star game (Jul 2), 1 missing from Sportradar (Sep 19 Valkyries/Lynx), 4 late playoff games after Sportradar data cutoff (Oct 11). These have `game_id = null` in `wnba_2025_game_markets_matched.csv`.
