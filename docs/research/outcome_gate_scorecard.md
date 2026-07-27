# Outcome gate scorecard

**Decision: PASS-BOUNDED → sweep ran; definitive robust candidates: 0**

More precisely: fee reconstruction is **PASS-DEFINITIVE** (exact zero-fee for 2022–2023).
Execution is **BOUNDED** (adverse mid-print + slippage sensitivity; no L2).
Overall authorization: **PASS-BOUNDED**.

## Gate checklist

| # | Requirement | Result |
|---|---|---|
| 1 | Canonical trades deduplicated + tested | **PASS** — HF sample has unique `(tx, log_index)`; lake audit: 1158 exact ingestion dups, 0 divergent |
| 2 | Source state frozen | **PASS** — see `source_checkpoint.md` |
| 3 | CLI purged event-grouped WF | **PASS** — `sweep-outcomes` |
| 4 | Final test frozen | **PASS** — train `< 2023-06-01`, event-purged test |
| 5 | Historical fees exact / evidenced / bounded | **PASS-DEFINITIVE** — zero-fee era through 2026-01-05 |
| 6 | Execution supported by trade data | **PASS-BOUNDED** — mid-print + slip; next-print path available; no maker/queue |
| 7 | Placeholder stats removed | **PASS** — FDR unavailable without p-values; bootstrap real; PBO shape caveat noted |
| 8 | Event coverage meaningful | **PASS** — 708 independent events (was 81) |
| 9 | Tests pass | **PASS** — 79 tests |

## Critical caveats

- No historical L2 → no maker fills, queue, OFI, book capacity.
- Slippage stress is modeled; latency does not delay quote selection on mid-print path.
- Top locked-test EV configs fail leave-5-largest-winners → correctly **fragile**.

## Artifacts

- `data/curated/trades_canonical.parquet`
- `data/curated/trade_features_canonical.parquet`
- `data/research/outcome_*.parquet` / `outcome_backtest_manifest.json`
