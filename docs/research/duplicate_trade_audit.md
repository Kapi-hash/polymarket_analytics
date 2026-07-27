# Duplicate trade audit

## Lake (`v_trades`) findings

| Metric | Value |
|---|---:|
| Rows | 100,888 |
| Unique `trade_id` | 99,336 |
| Duplicate `trade_id`s | 1,158 |
| Exact ingestion duplicates | **1,158** |
| Divergent duplicates (same id, differing fields) | **0** |
| Rows involved in dups | 2,710 |

**Classification:** all lake duplicates are exact ingestion/pagination copies (same `tx_hash`, `token_id`, `price`, `size`, `traded_at`, `side`).

## Deduplication rule

Stable fill identity (in priority order of available columns):

`tx_hash | log_index | token_id | price | size | traded_at | side`

- Keep first row after sorting by `ingest_at`, then `traded_at`, then `trade_id`.
- Assign `fill_id = sha256(identity)[:32]`.
- Raw parquet partitions are **not** modified; curated layer is written separately.

## HF sample (canonical outcome lake)

| Metric | Value |
|---|---:|
| Rows | 100,000 |
| Fill identity | `(transaction_hash, log_index)` unique |
| Duplicate trade_ids | 0 |

## Artifacts

- `data/quality/duplicate_trade_audit.parquet`
- `data/quality/duplicate_trade_detail.parquet` (lake detail)
- `data/curated/trades_canonical.parquet` (HF-expanded, resolved)

## Impact

Whale ratio / volume spike / event weights on the **raw lake feature path** were inflated by exact dups. Outcome research uses the curated HF layer with unique fills.
