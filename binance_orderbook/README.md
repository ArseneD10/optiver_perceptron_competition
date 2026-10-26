# Binance Orderbook Recorder

Real-time orderbook + trade recorder for Binance, written in **Rust**.

## What it does

| File | Content |
|---|---|
| `orderbook.csv` | best bid/ask price & quantity at the moment each trade arrives |
| `trades.csv` | trade price, quantity, side (`buy`/`sell`), timestamp |

**Both files are perfectly synchronised: line N in `orderbook.csv` corresponds to line N in `trades.csv`.**

### Architecture

```
Binance WebSocket (combined stream)
  ├─ btcusdt@depth5@100ms  →  updates local BestQuotes state (in-memory)
  └─ btcusdt@trade         →  snapshots BestQuotes + writes one row to each CSV
```

The depth stream runs at **100 ms** intervals on Binance's side.  
The orderbook state is always up-to-date; the CSV snapshot is triggered **only** on trade events.

## Quick start

```bash
# Build (release)
cargo build --release

# Run (Ctrl-C to stop)
./target/release/binance_orderbook

# Or with debug logging
RUST_LOG=debug ./target/release/binance_orderbook
```

## Configuration

Edit the constants at the top of `src/main.rs`:

| Constant | Default | Description |
|---|---|---|
| `SYMBOL` | `btcusdt` | Trading pair |
| `DEPTH_STREAM` | `btcusdt@depth5@100ms` | Depth stream name |
| `TRADE_STREAM` | `btcusdt@trade` | Trade stream name |
| `ORDERBOOK_CSV` | `orderbook.csv` | Output path for orderbook |
| `TRADES_CSV` | `trades.csv` | Output path for trades |

## Output format

### `orderbook.csv`
```
trade_time_ms,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty
1729956000123,67250.01,0.15,67250.02,0.30
...
```

### `trades.csv`
```
trade_id,trade_time_ms,price,quantity,side
123456789,1729956000123,67250.02,0.05,buy
...
```

**`side`** field:
- `buy`  → taker is the buyer  (aggressive buy order)
- `sell` → taker is the seller (aggressive sell order)

## Dependencies

- [`tokio`](https://crates.io/crates/tokio) — async runtime
- [`tokio-tungstenite`](https://crates.io/crates/tokio-tungstenite) — WebSocket client
- [`serde_json`](https://crates.io/crates/serde_json) — JSON deserialization
- [`csv`](https://crates.io/crates/csv) — CSV serialization
- [`rust_decimal`](https://crates.io/crates/rust_decimal) — lossless decimal arithmetic
- [`tracing`](https://crates.io/crates/tracing) — structured logging
