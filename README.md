# Phase 1: Polymarket Data Ingestion & Storage

Local analytics warehouse for historical Polymarket trade logs and market outcomes.

**Stack:** Polars (vectorized ingest) → partitioned Parquet lake → DuckDB SQL views.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Ingest sample fixtures into data/parquet + data/warehouse.duckdb
python -m polymarket_analytics ingest --with-fixtures

# Warehouse summary
python -m polymarket_analytics status
```

Drop your own dumps under:

- `data/raw/trades/` — CSV / JSON / NDJSON / Parquet (Data API trade shape)
- `data/raw/markets/` — market outcome / Gamma-style resolution files

Then run `python -m polymarket_analytics ingest`.

## Schema

| Table | Grain | Purpose |
|-------|-------|---------|
| `trades` | one fill | price, size, side, token, condition, timestamp |
| `markets` | one condition_id | resolution outcome, tokens, volume |
| `market_tokens` | condition × token | outcome label map |

DuckDB views: `v_trades`, `v_markets`, `v_market_tokens`, `v_resolved_trades` (joined with `token_won`).

## Tests

```bash
pytest -q
```
