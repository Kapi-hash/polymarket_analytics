"""Typed column contracts and DuckDB DDL for the local analytics warehouse."""

from __future__ import annotations

from typing import Final, TypedDict

import polars as pl

# ---------------------------------------------------------------------------
# Row-level TypedDict contracts (documentation + static checking)
# ---------------------------------------------------------------------------


class TradeRow(TypedDict, total=False):
    trade_id: str
    condition_id: str
    token_id: str
    side: str
    price: float
    size: float
    notional: float
    outcome_label: str | None
    outcome_index: int | None
    wallet: str | None
    tx_hash: str | None
    traded_at: str  # ISO-8601 UTC
    ingest_at: str
    source_file: str


class MarketRow(TypedDict, total=False):
    condition_id: str
    question: str | None
    slug: str | None
    token_yes: str | None
    token_no: str | None
    winning_token_id: str | None
    winning_outcome: str | None
    resolved: bool
    resolved_at: str | None
    closed_at: str | None
    end_date: str | None
    volume: float | None
    liquidity: float | None
    neg_risk: bool
    ingest_at: str


class MarketTokenRow(TypedDict):
    condition_id: str
    token_id: str
    outcome_label: str | None
    outcome_index: int | None


# ---------------------------------------------------------------------------
# Polars schemas (canonical on-disk / in-memory contracts)
# ---------------------------------------------------------------------------

TRADES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "trade_id": pl.Utf8,
    "condition_id": pl.Utf8,
    "token_id": pl.Utf8,
    "side": pl.Categorical,
    "price": pl.Float64,
    "size": pl.Float64,
    "notional": pl.Float64,
    "outcome_label": pl.Utf8,
    "outcome_index": pl.Int8,
    "wallet": pl.Utf8,
    "tx_hash": pl.Utf8,
    "traded_at": pl.Datetime("us", "UTC"),
    "ingest_at": pl.Datetime("us", "UTC"),
    "source_file": pl.Utf8,
}

MARKETS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "condition_id": pl.Utf8,
    "question": pl.Utf8,
    "slug": pl.Utf8,
    "token_yes": pl.Utf8,
    "token_no": pl.Utf8,
    "winning_token_id": pl.Utf8,
    "winning_outcome": pl.Utf8,
    "resolved": pl.Boolean,
    "resolved_at": pl.Datetime("us", "UTC"),
    "closed_at": pl.Datetime("us", "UTC"),
    "end_date": pl.Datetime("us", "UTC"),
    "volume": pl.Float64,
    "liquidity": pl.Float64,
    "neg_risk": pl.Boolean,
    "ingest_at": pl.Datetime("us", "UTC"),
}

MARKET_TOKENS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "condition_id": pl.Utf8,
    "token_id": pl.Utf8,
    "outcome_label": pl.Utf8,
    "outcome_index": pl.Int8,
}

# Phase 2 / 2.5: per-trade rolling / bucket / TTR / composite features
FEATURES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "trade_id": pl.Utf8,
    "condition_id": pl.Utf8,
    "token_id": pl.Utf8,
    "traded_at": pl.Datetime("us", "UTC"),
    "price": pl.Float64,
    "notional": pl.Float64,
    "momentum_1h": pl.Float64,
    "momentum_6h": pl.Float64,
    "momentum_24h": pl.Float64,
    "volatility_1h": pl.Float64,
    "volatility_6h": pl.Float64,
    "volatility_24h": pl.Float64,
    "volume_1h": pl.Float64,
    "volume_6h": pl.Float64,
    "volume_24h": pl.Float64,
    "volume_spike_1h_24h": pl.Float64,
    "price_bucket": pl.Utf8,
    "time_to_resolution_hours": pl.Float64,
    # Phase 2.5 composites
    "whale_ratio": pl.Float64,
    "decay_adjusted_velocity": pl.Float64,
    "price_volume_divergence": pl.Boolean,
    "feature_at": pl.Datetime("us", "UTC"),
}

WHALE_RATIO_DIVERGENCE_THRESHOLD: Final[float] = 3.0
DECAY_TTR_FLOOR: Final[float] = 0.1

TRADES_COLUMNS: Final[list[str]] = list(TRADES_SCHEMA.keys())
MARKETS_COLUMNS: Final[list[str]] = list(MARKETS_SCHEMA.keys())
MARKET_TOKENS_COLUMNS: Final[list[str]] = list(MARKET_TOKENS_SCHEMA.keys())
FEATURES_COLUMNS: Final[list[str]] = list(FEATURES_SCHEMA.keys())

# Non-linear probability buckets (finer near 0 and 1).
# Polars cut() interior breakpoints → len(breaks)+1 labels covering (-inf, +inf).
PRICE_BUCKET_BREAKS: Final[list[float]] = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
]
PRICE_BUCKET_LABELS: Final[list[str]] = [
    "0.00-0.05",
    "0.05-0.10",
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.85",
    "0.85-0.90",
    "0.90-0.95",
    "0.95-1.00",
]

# Rolling window periods used by Phase 2
FEATURE_WINDOWS: Final[tuple[tuple[str, str], ...]] = (
    ("1h", "1h"),
    ("6h", "6h"),
    ("24h", "24h"),
)

# Source-field aliases → canonical names (Data API / Gamma dumps)
TRADE_ALIASES: Final[dict[str, str]] = {
    "asset": "token_id",
    "tokenId": "token_id",
    "token_id": "token_id",
    "conditionId": "condition_id",
    "condition_id": "condition_id",
    "proxyWallet": "wallet",
    "wallet": "wallet",
    "transactionHash": "tx_hash",
    "transaction_hash": "tx_hash",
    "tx_hash": "tx_hash",
    "side": "side",
    "price": "price",
    "size": "size",
    "timestamp": "traded_at",
    "traded_at": "traded_at",
    "outcome": "outcome_label",
    "outcome_label": "outcome_label",
    "outcomeIndex": "outcome_index",
    "outcome_index": "outcome_index",
    "trade_id": "trade_id",
    "id": "trade_id",
}

MARKET_ALIASES: Final[dict[str, str]] = {
    "conditionId": "condition_id",
    "condition_id": "condition_id",
    "question": "question",
    "title": "question",
    "slug": "slug",
    "token_yes": "token_yes",
    "tokenYes": "token_yes",
    "clobTokenIds": "clob_token_ids",
    "token_no": "token_no",
    "tokenNo": "token_no",
    "winning_token_id": "winning_token_id",
    "winningTokenId": "winning_token_id",
    "winning_outcome": "winning_outcome",
    "winningOutcome": "winning_outcome",
    "resolved": "resolved",
    "closed": "closed",
    "resolved_at": "resolved_at",
    "resolvedAt": "resolved_at",
    "closed_at": "closed_at",
    "closedTime": "closed_at",
    "end_date": "end_date",
    "endDate": "end_date",
    "endDateIso": "end_date",
    "volume": "volume",
    "liquidity": "liquidity",
    "neg_risk": "neg_risk",
    "negRisk": "neg_risk",
    "outcomePrices": "outcome_prices",
    "outcomes": "outcomes",
}

# Seed partition date used only so empty lakes still register in DuckDB
EMPTY_TRADES_SEED_DATE: Final[str] = "1970-01-01"


def duckdb_ddl() -> str:
    """CREATE TABLE DDL for optional materialization into DuckDB."""
    return """
CREATE TABLE IF NOT EXISTS trades (
    trade_id VARCHAR NOT NULL,
    condition_id VARCHAR NOT NULL,
    token_id VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    price DOUBLE NOT NULL,
    size DOUBLE NOT NULL,
    notional DOUBLE NOT NULL,
    outcome_label VARCHAR,
    outcome_index TINYINT,
    wallet VARCHAR,
    tx_hash VARCHAR,
    traded_at TIMESTAMPTZ NOT NULL,
    ingest_at TIMESTAMPTZ NOT NULL,
    source_file VARCHAR
);

CREATE TABLE IF NOT EXISTS markets (
    condition_id VARCHAR PRIMARY KEY,
    question VARCHAR,
    slug VARCHAR,
    token_yes VARCHAR,
    token_no VARCHAR,
    winning_token_id VARCHAR,
    winning_outcome VARCHAR,
    resolved BOOLEAN NOT NULL,
    resolved_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    end_date TIMESTAMPTZ,
    volume DOUBLE,
    liquidity DOUBLE,
    neg_risk BOOLEAN,
    ingest_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS market_tokens (
    condition_id VARCHAR NOT NULL,
    token_id VARCHAR NOT NULL,
    outcome_label VARCHAR,
    outcome_index TINYINT,
    PRIMARY KEY (condition_id, token_id)
);
""".strip()


def parquet_view_sql(
    trades_glob: str,
    markets_glob: str,
    tokens_glob: str,
    features_glob: str | None = None,
) -> str:
    """Views over the Parquet lake (hive partitioning for trades)."""
    sql = f"""
CREATE OR REPLACE VIEW v_trades AS
SELECT * EXCLUDE (date)
FROM read_parquet('{trades_glob}', hive_partitioning := true, union_by_name := true);

CREATE OR REPLACE VIEW v_markets AS
SELECT *
FROM read_parquet('{markets_glob}', union_by_name := true);

CREATE OR REPLACE VIEW v_market_tokens AS
SELECT *
FROM read_parquet('{tokens_glob}', union_by_name := true);

CREATE OR REPLACE VIEW v_resolved_trades AS
SELECT
    t.*,
    m.question,
    m.slug,
    m.winning_token_id,
    m.winning_outcome,
    m.resolved_at,
    (t.token_id = m.winning_token_id) AS token_won
FROM v_trades t
INNER JOIN v_markets m
    ON t.condition_id = m.condition_id
WHERE m.resolved = TRUE
  AND m.winning_token_id IS NOT NULL;
"""
    if features_glob:
        sql += f"""
CREATE OR REPLACE VIEW v_features AS
SELECT *
FROM read_parquet('{features_glob}', union_by_name := true);

CREATE OR REPLACE VIEW v_trade_features AS
SELECT
    r.*,
    f.momentum_1h,
    f.momentum_6h,
    f.momentum_24h,
    f.volatility_1h,
    f.volatility_6h,
    f.volatility_24h,
    f.volume_1h,
    f.volume_6h,
    f.volume_24h,
    f.volume_spike_1h_24h,
    f.price_bucket,
    f.time_to_resolution_hours,
    f.whale_ratio,
    f.decay_adjusted_velocity,
    f.price_volume_divergence,
    f.feature_at
FROM v_resolved_trades r
INNER JOIN v_features f
    ON r.trade_id = f.trade_id;
"""
    return sql.strip()
