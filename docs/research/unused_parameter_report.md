# Declared-but-unused / partially wired

## Unused
- `PaperTrader.matches.min_volume_spike`: If set, matches() always returns False — LiveFeatures lacks volume_spike
- `PaperTrader.matches.max_time_to_resolution_hours`: If set, matches() always returns False — LiveFeatures lacks TTR
- `exogenous.*`: Provider stubs only

## Partially wired
- `PaperConfig.fee_bps`: Active only with --flat-fees
- `StrategyParams.require_price_volume_divergence`: Field exists but iter_grid_params never sets True — not in default grid
- `momentum_6h grid`: Axis declared but default grid freezes momentum_6h at any
- `ExecutionConfig.latency_ms`: Present on PaperConfig and FillResult.meta, but does not delay quote selection or change fill price (inert until delayed-book model exists).
- `PaperConfig.use_book_walk`: Opt-in L1 ask cross; CLI does not expose the flag; no historical L2