mod models;
mod recorder;

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use models::{BestQuotes, CombinedMessage, DepthUpdate, OrderbookRow, TradeEvent, TradeRow};
use recorder::Recorder;
use rust_decimal::Decimal;
use std::str::FromStr;
use tokio_tungstenite::connect_async;
use tracing::{error, info, warn};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Trading pair in lowercase (e.g. "btcusdt")
const SYMBOL: &str = "btcusdt";

/// Binance combined-stream endpoint
const WS_BASE: &str = "wss://stream.binance.com:9443/stream?streams=";

/// Orderbook depth snapshot stream (5 levels, 100 ms updates)
const DEPTH_STREAM: &str = "btcusdt@depth5@100ms";

/// Individual trade stream
const TRADE_STREAM: &str = "btcusdt@trade";

/// Output files
const ORDERBOOK_CSV: &str = "orderbook.csv";
const TRADES_CSV: &str = "trades.csv";

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a `&str` into `Decimal`, returning an error on failure.
fn parse_decimal(s: &str) -> Result<Decimal> {
    Decimal::from_str(s).with_context(|| format!("Cannot parse decimal: '{s}'"))
}

/// Extract the best bid (first element of bids array) from a `DepthUpdate`.
/// Returns `None` if the bids list is empty.
fn best_bid(depth: &DepthUpdate) -> Option<(Decimal, Decimal)> {
    depth.bids.first().and_then(|pair| {
        let price = parse_decimal(&pair[0]).ok()?;
        let qty = parse_decimal(&pair[1]).ok()?;
        Some((price, qty))
    })
}

/// Extract the best ask (first element of asks array) from a `DepthUpdate`.
/// Returns `None` if the asks list is empty.
fn best_ask(depth: &DepthUpdate) -> Option<(Decimal, Decimal)> {
    depth.asks.first().and_then(|pair| {
        let price = parse_decimal(&pair[0]).ok()?;
        let qty = parse_decimal(&pair[1]).ok()?;
        Some((price, qty))
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Main entry-point
// ──────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise structured logging (RUST_LOG env var controls level).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!(symbol = SYMBOL, "Starting Binance orderbook recorder");

    // Build WebSocket URL for the combined stream (depth + trades).
    let ws_url = format!("{WS_BASE}{DEPTH_STREAM}/{TRADE_STREAM}");

    info!(%ws_url, "Connecting to Binance WebSocket");
    let (ws_stream, _) = connect_async(ws_url.as_str())
        .await
        .context("WebSocket connection failed")?;
    info!("WebSocket connected");

    let mut ws = ws_stream;

    // CSV recorder
    let mut recorder = Recorder::new(ORDERBOOK_CSV, TRADES_CSV)
        .context("Failed to initialise CSV recorder")?;

    // Local state: last known best quotes (updated on every depth event)
    let mut quotes = BestQuotes::default();
    let mut quotes_ready = false; // true once we have received at least one depth snapshot

    // ──────────────────────────────────────────────────────────────────────────
    // Main event loop
    // ──────────────────────────────────────────────────────────────────────────
    while let Some(msg_result) = ws.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                error!("WebSocket error: {e}");
                break;
            }
        };

        // Only handle text frames
        let text = match msg {
            tokio_tungstenite::tungstenite::Message::Text(t) => t,
            tokio_tungstenite::tungstenite::Message::Close(_) => {
                info!("WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        // Deserialise the combined-stream envelope
        let combined: CombinedMessage = match serde_json::from_str(&text) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to parse message: {e}\nRaw: {text}");
                continue;
            }
        };

        // Route by stream name
        if combined.stream.ends_with("@depth5@100ms") {
            // ── Depth snapshot ─────────────────────────────────────────────
            let depth: DepthUpdate = match serde_json::from_value(combined.data) {
                Ok(d) => d,
                Err(e) => {
                    warn!("Failed to parse depth update: {e}");
                    continue;
                }
            };

            match (best_bid(&depth), best_ask(&depth)) {
                (Some((bp, bq)), Some((ap, aq))) => {
                    quotes.best_bid_price = bp;
                    quotes.best_bid_qty = bq;
                    quotes.best_ask_price = ap;
                    quotes.best_ask_qty = aq;
                    quotes_ready = true;
                }
                _ => {
                    warn!("Depth snapshot missing bid or ask side");
                }
            }
        } else if combined.stream.ends_with("@trade") {
            // ── Trade event ────────────────────────────────────────────────
            if !quotes_ready {
                // We haven't received a depth snapshot yet — skip this trade
                // to avoid writing a row with zeroed quotes.
                warn!("Trade received before first depth snapshot — skipping");
                continue;
            }

            let trade: TradeEvent = match serde_json::from_value(combined.data) {
                Ok(t) => t,
                Err(e) => {
                    warn!("Failed to parse trade event: {e}");
                    continue;
                }
            };

            let price = match parse_decimal(&trade.price) {
                Ok(p) => p,
                Err(e) => {
                    warn!("Bad trade price '{}': {e}", trade.price);
                    continue;
                }
            };

            let quantity = match parse_decimal(&trade.quantity) {
                Ok(q) => q,
                Err(e) => {
                    warn!("Bad trade quantity '{}': {e}", trade.quantity);
                    continue;
                }
            };

            // Determine the trade side from the buyer-maker flag:
            //   is_buyer_maker = true  → buyer is the passive side (maker)
            //                          → the aggressor is the SELLER → "sell"
            //   is_buyer_maker = false → buyer is the aggressor        → "buy"
            let side = if trade.is_buyer_maker { "sell" } else { "buy" }.to_string();

            // Build the two CSV rows
            let ob_row = OrderbookRow {
                trade_time_ms: trade.trade_time,
                best_bid_price: quotes.best_bid_price,
                best_bid_qty: quotes.best_bid_qty,
                best_ask_price: quotes.best_ask_price,
                best_ask_qty: quotes.best_ask_qty,
            };

            let trade_row = TradeRow {
                trade_id: trade.trade_id,
                trade_time_ms: trade.trade_time,
                price,
                quantity,
                side,
            };

            // Log progress every 100 rows
            if recorder.rows_written % 100 == 0 {
                info!(
                    rows = recorder.rows_written,
                    bid = %quotes.best_bid_price,
                    ask = %quotes.best_ask_price,
                    trade_price = %price,
                    "Progress"
                );
            }

            if let Err(e) = recorder.record(&ob_row, &trade_row) {
                error!("Failed to write CSV row: {e}");
                bail!("CSV write error — aborting");
            }
        } else {
            warn!("Unknown stream: {}", combined.stream);
        }
    }

    info!(
        total_rows = recorder.rows_written,
        "Session ended. Both CSV files contain the same number of rows."
    );
    Ok(())
}
