# Staged full-sweep design (DO NOT RUN automatically)

This document designs a controlled sweep. The representative backtest script must remain the only automated evaluation path in CI/local default flows.

## Stage 0 — Freeze baselines
- Pin `FEE_MODEL_VERSION` on all runs.
- Snapshot code defaults via `research.config.existing_code_defaults_snapshot()`.
- Separate `existing_code_grids` from `recommended_priors.feature_sweep`.

## Stage 1 — Leakage-safe harness only
- Grouped purged walk-forward by `condition_id`, embargo 1–2 days.
- Fit calibrators on train folds only.
- Record DSR / PBO proxy / BH-FDR / neighborhood stability (scaffolding already in `research.validation`).

## Stage 2 — Fee & execution stress (small grid)
Axes (recommended, not existing):
- `spread_slippage_bps ∈ {25, 50, 100}`
- `latency_ms ∈ {0, 50, 250}`
- `role ∈ {taker, maker}` where data allows
- Category fee from PIT market metadata (not swept as a free knob)

## Stage 3 — Core signal families (controlled)
Priority families for first controlled sweep (see deliverable #21):
1. Whale mid-bucket edge (`min_whale_ratio`, bucket)
2. Volume spike + momentum sign
3. TTR / information hazard gates
4. Logit-edge vs calibrated residual
5. RSI mean-reversion (swing)
6. Hurst + EMA momentum (swing)
7. Book imbalance / confluence
8. Complete-set residual (when YES/NO available)
9. Fee-aware net EV threshold
10. Risk/lifecycle caps (position %, cooldown, DD halt)

Keep Cartesian products tiny: fix all but 2–3 axes per stage.

## Stage 4 — Neighborhood stability
- For each winner, evaluate adjacent parameter neighbors.
- Require same-sign OOS EV and stability ≥ 0.6 before promotion.

## Stage 5 — Paper incubator confirmation
- Promote only to multi-profile paper journals (existing GHA Paper Trader).
- No live capital; no claim of alpha.

## Explicit non-goals until data arrives
- Multi-level OFI / book resilience / queue-ahead realism without L2/L3 history.
- Exogenous news/macro/onchain features without PIT providers.
- Cross-market lead-lag without related-market graph.
