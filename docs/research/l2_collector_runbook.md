# L2 collector runbook

Prospective order-book collector for **future** swing / microstructure research.
Does **not** unlock historical L2 backtests.

## Module

`polymarket_analytics/collectors/book_collector.py`

## Smoke test (verified)

```bash
python3 -m polymarket_analytics collect-books --duration 25 --n-tokens 3
```

Example result (2026-07-26):

- Connected: true
- Messages: 30
- Gaps: 0
- Reconnects: 0
- Health: `data/books/collector_health.json`
- Raw: `data/books/raw/session_*.jsonl`
- Normalized: `data/books/normalized/session_*.parquet`

## Continuous local collection

```bash
# Long-running local capture (no paid infra)
python3 -m polymarket_analytics collect-books --duration 86400 --n-tokens 20
```

Or pin tokens:

```bash
python3 -m polymarket_analytics collect-books \
  --duration 3600 \
  --token-id <TOKEN_A> \
  --token-id <TOKEN_B>
```

## Notes

- Gamma discovery requires a browser-like `User-Agent` (403 without it).
- After reconnect, collector continues; gap counter increments on sequence jumps when present.
- Store raw JSONL **before** normalization.
- Do not claim historical maker/queue realism until months of continuous books accumulate.

## Still blocked historically

OFI, queue imbalance, microprice, book resilience, maker fill probability, depth walking, capacity-from-depth.
