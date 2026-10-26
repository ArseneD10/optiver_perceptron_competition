use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Binance WebSocket raw messages
// ──────────────────────────────────────────────────────────────────────────────

/// Partial book depth snapshot (e.g. <symbol>@depth5@100ms)
/// Returns top-N bids/asks directly — no local maintenance required.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct DepthUpdate {
    /// Last update ID on Binance side (for ordering)
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,

    /// List of [price, qty] pairs – descending order (best bid first)
    pub bids: Vec<[String; 2]>,

    /// List of [price, qty] pairs – ascending order (best ask first)
    pub asks: Vec<[String; 2]>,
}

/// Individual trade stream event (<symbol>@trade)
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct TradeEvent {
    /// Event type ("trade")
    #[serde(rename = "e")]
    pub event_type: String,

    /// Event time (ms since epoch)
    #[serde(rename = "E")]
    pub event_time: u64,

    /// Symbol
    #[serde(rename = "s")]
    pub symbol: String,

    /// Trade ID
    #[serde(rename = "t")]
    pub trade_id: u64,

    /// Price (as string from Binance)
    #[serde(rename = "p")]
    pub price: String,

    /// Quantity (as string from Binance)
    #[serde(rename = "q")]
    pub quantity: String,

    /// Buyer order ID
    #[serde(rename = "b")]
    pub buyer_order_id: u64,

    /// Seller order ID
    #[serde(rename = "a")]
    pub seller_order_id: u64,

    /// Trade time (ms since epoch)
    #[serde(rename = "T")]
    pub trade_time: u64,

    /// Is the buyer the market maker?
    /// true  → sell-initiated trade (taker is seller)
    /// false → buy-initiated trade  (taker is buyer)
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

// ──────────────────────────────────────────────────────────────────────────────
// Combined stream envelope
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CombinedMessage {
    /// Stream name, e.g. "btcusdt@trade" or "btcusdt@depth5@100ms"
    pub stream: String,
    /// Raw JSON value of the payload
    pub data: serde_json::Value,
}

// ──────────────────────────────────────────────────────────────────────────────
// Local order-book state (best bid / best ask only)
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct BestQuotes {
    pub best_bid_price: Decimal,
    pub best_bid_qty: Decimal,
    pub best_ask_price: Decimal,
    pub best_ask_qty: Decimal,
}

// ──────────────────────────────────────────────────────────────────────────────
// CSV row types
// ──────────────────────────────────────────────────────────────────────────────

/// One row written to orderbook.csv (one per trade)
#[derive(Debug, Serialize)]
pub struct OrderbookRow {
    /// Timestamp of the trade that triggered this snapshot (ms)
    pub trade_time_ms: u64,
    pub best_bid_price: Decimal,
    pub best_bid_qty: Decimal,
    pub best_ask_price: Decimal,
    pub best_ask_qty: Decimal,
}

/// One row written to trades.csv
#[derive(Debug, Serialize)]
pub struct TradeRow {
    /// Trade ID from Binance
    pub trade_id: u64,
    /// Trade timestamp (ms since epoch)
    pub trade_time_ms: u64,
    /// Trade price
    pub price: Decimal,
    /// Trade quantity
    pub quantity: Decimal,
    /// "buy" if taker is buyer, "sell" if taker is seller
    pub side: String,
}
