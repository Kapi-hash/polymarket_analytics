"""Unit tests for Phase 1 ingest + DuckDB resolved join."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from polymarket_analytics.ingest import (
    normalize_markets,
    normalize_trades,
    run_ingest,
    write_trades_parquet,
)
from polymarket_analytics.store import bootstrap_warehouse, connect, warehouse_status

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


@pytest.fixture
def sample_trades_raw() -> pl.DataFrame:
    return pl.DataFrame(json.loads((FIXTURES / "sample_trades.json").read_text()))


@pytest.fixture
def sample_markets_raw() -> pl.DataFrame:
    return pl.DataFrame(json.loads((FIXTURES / "sample_markets.json").read_text()))


def test_alias_mapping_and_price_filter(sample_trades_raw: pl.DataFrame) -> None:
    out = normalize_trades(sample_trades_raw, source_file="fixtures/sample_trades.json")
    assert "token_id" in out.columns
    assert "condition_id" in out.columns
    assert "wallet" in out.columns
    # Invalid price 1.5 dropped
    assert out.filter(pl.col("tx_hash") == "0xtx_invalid_price").is_empty()
    assert out["price"].min() >= 0.0
    assert out["price"].max() <= 1.0
    assert set(out["side"].cast(pl.Utf8).unique().to_list()) <= {"BUY", "SELL"}


def test_trade_id_stability(sample_trades_raw: pl.DataFrame) -> None:
    a = normalize_trades(sample_trades_raw, source_file="a.json")
    b = normalize_trades(sample_trades_raw, source_file="b.json")
    # Same logical trades → same ids (source_file differs but id inputs do not)
    merged = a.select("tx_hash", "trade_id").join(
        b.select("tx_hash", pl.col("trade_id").alias("trade_id_b")),
        on="tx_hash",
        how="inner",
    )
    assert (merged["trade_id"] == merged["trade_id_b"]).all()


def test_notional_and_epoch_ms(sample_trades_raw: pl.DataFrame) -> None:
    out = normalize_trades(sample_trades_raw, source_file="fixtures")
    row = out.filter(pl.col("tx_hash") == "0xtx_yes_ms")
    assert row.height == 1
    assert abs(row["notional"][0] - row["price"][0] * row["size"][0]) < 1e-9
    # 1700000000000 ms → same instant as 1700000000 s
    sec_row = out.filter(pl.col("tx_hash") == "0xtx_yes_1")
    assert row["traded_at"][0] == sec_row["traded_at"][0]


def test_markets_resolution_tokens(sample_markets_raw: pl.DataFrame) -> None:
    markets, tokens = normalize_markets(sample_markets_raw)
    assert markets.height == 3
    yes = markets.filter(pl.col("condition_id") == "0xcond_yes_wins")
    assert yes["resolved"][0] is True
    assert yes["winning_token_id"][0].startswith("1111")
    no = markets.filter(pl.col("condition_id") == "0xcond_no_wins")
    assert no["winning_token_id"][0].startswith("4444")
    open_m = markets.filter(pl.col("condition_id") == "0xcond_open")
    assert open_m["resolved"][0] is False
    assert open_m["winning_token_id"][0] is None
    assert tokens.height == 6


def test_partition_write(tmp_path: Path, sample_trades_raw: pl.DataFrame) -> None:
    trades = normalize_trades(sample_trades_raw, source_file="t.json")
    n = write_trades_parquet(trades, tmp_path)
    assert n == trades.height
    parts = list((tmp_path / "trades").rglob("*.parquet"))
    assert parts
    assert any("date=" in str(p) for p in parts)


def test_end_to_end_ingest_and_resolved_join(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "parquet"
    wh = tmp_path / "warehouse.duckdb"
    (raw / "trades").mkdir(parents=True)
    (raw / "markets").mkdir(parents=True)
    (raw / "trades" / "sample_trades.json").write_text(
        (FIXTURES / "sample_trades.json").read_text()
    )
    (raw / "markets" / "sample_markets.json").write_text(
        (FIXTURES / "sample_markets.json").read_text()
    )

    stats = run_ingest(raw, out, bootstrap_warehouse=True, warehouse_path=wh)
    assert stats["trades"] >= 18  # one invalid price dropped from 20
    assert stats["markets"] == 3

    status = warehouse_status(wh)
    assert status["trades"] == stats["trades"]
    assert status["resolved_markets"] == 2
    assert status["resolved_trades"] > 0

    con = connect(wh)
    try:
        rows = con.execute(
            """
            SELECT token_won, COUNT(*) AS n
            FROM v_resolved_trades
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()
        assert rows
        # Both won and lost tokens should appear for resolved markets
        won_flags = {r[0] for r in rows}
        assert True in won_flags
        assert False in won_flags
    finally:
        con.close()


def test_bootstrap_empty_warehouse(tmp_path: Path) -> None:
    parquet = tmp_path / "parquet"
    wh = tmp_path / "wh.duckdb"
    bootstrap_warehouse(parquet, wh)
    status = warehouse_status(wh)
    assert status["trades"] == 0
    assert status["markets"] == 0
