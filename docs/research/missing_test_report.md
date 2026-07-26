# Missing-test report

## Covered (pre-expansion)
- Phase 1 ingest, Phase 2 features, Phase 3 backtest/edges, OOS split
- Phase 4 paper feed, Phase 5 fees/resolution/TP-SL
- Profiles + swing trader unit tests

## Gaps addressed by new tests
- Logit clamp / logit edge
- Fee model version + maker vs taker + rounding + fee-free
- Complete-set residual when YES/NO present
- OFI helper (synthetic levels)
- PIT calibrator fit-on-train-only
- Event-grouped purge / embargo folds
- TP/SL same-tick ordering
- Inventory detectors (CLI defaults, conflicts, unused, hardcoded)

## Still thin / blocked
- Full L2 OFI on historical lake (no depth data)
- Exogenous providers (interfaces only)
- Cross-market lead-lag
- End-to-end PaperTrader latency/book-walk path (execution helpers unit-tested only)
