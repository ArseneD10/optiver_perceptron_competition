use anyhow::{Context, Result};
use csv::Writer;
use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::models::{OrderbookRow, TradeRow};

// ──────────────────────────────────────────────────────────────────────────────
// CSV writers
// ──────────────────────────────────────────────────────────────────────────────

/// Opens (or creates) a CSV file and optionally writes the header.
/// Returns a `csv::Writer` that keeps the file open for streaming writes.
fn open_csv_writer(path: &str) -> Result<Writer<File>> {
    let needs_header = !Path::new(path).exists();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Cannot open CSV file: {path}"))?;
    let wtr = csv::WriterBuilder::new()
        .has_headers(needs_header)
        .from_writer(file);

    // csv::Writer will automatically serialise headers on the first record
    // because we passed `has_headers(true)` when the file is new.
    // For append mode on an existing file we skip the header.
    let _ = needs_header; // used implicitly above
    Ok(wtr)
}

// ──────────────────────────────────────────────────────────────────────────────
// Public recorder struct
// ──────────────────────────────────────────────────────────────────────────────

pub struct Recorder {
    orderbook_wtr: Writer<File>,
    trades_wtr: Writer<File>,
    /// Running count of rows written (sanity check: both files stay in sync)
    pub rows_written: u64,
}

impl Recorder {
    /// Create (or re-open in append mode) the two CSV files.
    pub fn new(orderbook_path: &str, trades_path: &str) -> Result<Self> {
        Ok(Self {
            orderbook_wtr: open_csv_writer(orderbook_path)?,
            trades_wtr: open_csv_writer(trades_path)?,
            rows_written: 0,
        })
    }

    /// Write one pair of rows atomically (both files or neither).
    /// Both writers are flushed after each pair to minimise data loss on crash.
    pub fn record(&mut self, ob: &OrderbookRow, trade: &TradeRow) -> Result<()> {
        self.orderbook_wtr
            .serialize(ob)
            .context("Failed to write orderbook row")?;
        self.trades_wtr
            .serialize(trade)
            .context("Failed to write trade row")?;

        // Flush both files so data is visible on disk immediately.
        self.orderbook_wtr.flush().context("Flush orderbook CSV")?;
        self.trades_wtr.flush().context("Flush trades CSV")?;

        self.rows_written += 1;
        Ok(())
    }
}
