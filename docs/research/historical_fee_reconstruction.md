# Historical fee reconstruction

## Decision for this lake (2022-11 → 2023-08)

| Period | Exchange | Maker | Taker | Formula | Confidence | Evidence |
|---|---|---:|---:|---|---|---|
| 1970-01-01 → 2026-01-05 | CLOB V1 / pre-fee era | 0 | 0 | `fee = 0` | **exact** | Multi-year zero-fee policy; fees began 2026-01-06 on crypto 15m markets, then phased category rollout |
| ≥ 2026-01-06 | CLOB V2 category schedule | 0 | category rate | `C × feeRate × p × (1−p)` | strongly_evidenced | Current docs / category table — **not applied** to this lake |

## Evidence summary

1. Contemporary reporting and Polymarket documentation: platform was fee-free for years; first taker fees on 15-minute crypto markets **2026-01-06**.
2. Broader category fees (sports Feb 2026; politics/finance/etc Mar 30 2026) post-date the lake.
3. Therefore applying `2026-07-polymarket-v1` crypto `feeRate=0.07` to 2022–2023 trades was **incorrect**.

## Implementation

- Module: `polymarket_analytics/research/historical_fees.py`
- Version pin: `clob-v1-historical-2026-07-27`
- Outcome sweep labels every result with `fee_regime=zero_fee_historical`, `fee_confidence=exact`

## Remaining uncertainty

- On-chain `feeRateBps` on individual V1 orders may have been non-zero in edge cases; no lake evidence of charged fees in this sample.
- Gas / bridging costs are **not** modeled (explicitly out of scope for CLOB trading fee reconstruction).
- Post-2026 research must use category PIT schedule, not the zero-fee regime.
