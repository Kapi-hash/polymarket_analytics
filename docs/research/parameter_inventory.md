# Parameter inventory (Phase 1)

Code-defined defaults and existing sweep ranges are taken from the repo.
Recommended ranges are labeled separately and are **not** claimed as existing.

Total parameters: **69**

## A — Signals, entries, exits, microstructure

### `WHALE_RATIO_DIVERGENCE_THRESHOLD`
- Category: signal/feature
- Declaration: `schema.py:WHALE_RATIO_DIVERGENCE_THRESHOLD`
- Current default: `3.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[2.0, 2.5, 3.0, 4.0, 5.0]`
- Usage: features.compute_composite_features
- Status: **Active**
- Notes: Used for price_volume_divergence boolean

### `DECAY_TTR_FLOOR`
- Category: signal/feature
- Declaration: `schema.py:DECAY_TTR_FLOOR`
- Current default: `0.1`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.05, 0.1, 0.25, 0.5]`
- Usage: features.compute_composite_features
- Status: **Active**

### `FEATURE_WINDOWS`
- Category: signal/feature
- Declaration: `schema.py:FEATURE_WINDOWS`
- Current default: `['1h', '6h', '24h']`
- Existing sweep range: `Not defined`
- Recommended sweep range: `['30m', '1h', '6h', '24h', '72h']`
- Usage: features.compute_rolling_features
- Status: **Active**

### `PRICE_BUCKET_BREAKS`
- Category: signal/feature
- Declaration: `schema.py:PRICE_BUCKET_BREAKS`
- Current default: `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]`
- Existing sweep range: `Not defined (labels used as grid axis)`
- Recommended sweep range: `Keep; optionally finer near 0.45-0.55`
- Usage: features.assign_price_bucket, backtest.iter_grid_params
- Status: **Active**

### `_EPS_HOURS`
- Category: signal/feature
- Declaration: `features.py:_EPS_HOURS`
- Current default: `1e-06`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Not defined — numerical floor`
- Usage: features._rolling_window_frame
- Status: **Hardcoded**
- Notes: Module private constant

### `DEFAULT_SPIKE_THRESHOLDS`
- Category: entry
- Declaration: `backtest.py:DEFAULT_SPIKE_THRESHOLDS`
- Current default: `[None, 1.5, 2.0, 3.0]`
- Existing sweep range: `[None, 1.5, 2.0, 3.0]`
- Recommended sweep range: `[None, 1.25, 1.5, 2.0, 3.0, 4.0]`
- Usage: backtest.iter_grid_params, backtest.find_edges
- Status: **Active**

### `DEFAULT_WHALE_THRESHOLDS`
- Category: entry
- Declaration: `backtest.py:DEFAULT_WHALE_THRESHOLDS`
- Current default: `[None, 2.0, 3.0]`
- Existing sweep range: `[None, 2.0, 3.0]`
- Recommended sweep range: `[None, 1.5, 2.0, 3.0, 4.0, 5.0]`
- Usage: backtest.iter_grid_params
- Status: **Active**

### `DEFAULT_MOMENTUM_SIGNS`
- Category: entry
- Declaration: `backtest.py:DEFAULT_MOMENTUM_SIGNS`
- Current default: `['any', 'pos', 'neg']`
- Existing sweep range: `['any', 'pos', 'neg']`
- Recommended sweep range: `['any', 'pos', 'neg']`
- Usage: backtest.iter_grid_params
- Status: **Active**

### `PaperConfig.min_oos_ev_pct`
- Category: entry
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `10.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[5.0, 8.0, 10.0, 15.0]`
- Usage: PaperTrader signal gate, cli --min-ev
- Status: **Active**

### `PaperConfig.take_profit_pct`
- Category: exit
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `None`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[None, 0.1, 0.2, 0.3]`
- Usage: PaperTrader early exit
- Status: **Active**

### `PaperConfig.stop_loss_pct`
- Category: exit
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `None`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[None, 0.05, 0.1, 0.15]`
- Usage: PaperTrader early exit
- Status: **Active**

### `SwingConfig.min_liquidity_usd`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `50000.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[10000, 25000, 50000, 100000]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.min_ev_pct`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `8.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[5.0, 8.0, 10.0, 12.0]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.rsi_period`
- Category: signal/feature
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `14`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[7, 14, 21]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.rsi_oversold`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `25.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[20, 25, 30, 35]`
- Usage: SwingTrader, profiles.py
- Status: **Active**
- Notes: profiles RSI use 25; confluence 30

### `SwingConfig.bb_period`
- Category: signal/feature
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `20`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[10, 20, 30]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.bb_std`
- Category: signal/feature
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `2.5`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[2.0, 2.5, 3.0]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.hurst_min`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `0.55`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.5, 0.55, 0.6, 0.65]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.ema_fast`
- Category: signal/feature
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `5`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[3, 5, 8]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.ema_slow`
- Category: signal/feature
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `20`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[15, 20, 30]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.momentum_min_move`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `0.1`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.05, 0.08, 0.1, 0.15]`
- Usage: SwingTrader, profiles.py
- Status: **Active**
- Notes: profiles momentum use 0.08

### `SwingConfig.whale_volume_ratio`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `2.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[1.5, 2.0, 3.0]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.book_imbalance_min`
- Category: microstructure
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `3.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[2.0, 2.5, 3.0, 4.0]`
- Usage: SwingTrader, profiles.py
- Status: **Conflicting**
- Notes: vs confluence_book_min=2.5

### `SwingConfig.book_price_lo`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `0.2`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.1, 0.2, 0.3]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.book_price_hi`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `0.8`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.7, 0.8, 0.9]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.take_profit_pct`
- Category: exit
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `0.2`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.1, 0.15, 0.2, 0.3]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.take_profit_atr_mult`
- Category: exit
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `2.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[1.5, 2.0, 3.0]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.stop_loss_pct`
- Category: exit
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `0.1`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.05, 0.1, 0.15]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.stop_loss_atr_mult`
- Category: exit
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `1.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.5, 1.0, 1.5]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.atr_period`
- Category: signal/feature
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `14`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[7, 14, 21]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.atr_stop_mult`
- Category: exit
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `2.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[1.5, 2.0, 3.0]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.use_bb_take_profit`
- Category: exit
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `True`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[True, False]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.history_len`
- Category: signal/feature
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `200`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[100, 200, 500]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.require_confluence`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `False`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[True, False]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.min_confluence`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `2`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[2, 3]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.confluence_rsi`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `30.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[25, 30, 35]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.confluence_volume_usd`
- Category: entry
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `25000.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[10000, 25000, 50000]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.confluence_book_min`
- Category: microstructure
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `2.5`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[2.0, 2.5, 3.0]`
- Usage: SwingTrader, profiles.py
- Status: **Conflicting**

### `StrategyParams.require_price_volume_divergence`
- Category: entry
- Declaration: `backtest.py:StrategyParams`
- Current default: `False`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[False, True]`
- Usage: apply_strategy_filter
- Status: **Partially wired**
- Notes: Field exists but iter_grid_params never sets True — not in default grid

### `momentum_6h grid`
- Category: entry
- Declaration: `backtest.py:iter_grid_params`
- Current default: `['any']`
- Existing sweep range: `['any']`
- Recommended sweep range: `['any', 'pos', 'neg']`
- Usage: find_edges
- Status: **Partially wired**
- Notes: Axis declared but default grid freezes momentum_6h at any

### `logit_half_tick`
- Category: signal/feature
- Declaration: `research/logit.py:DEFAULT_HALF_TICK`
- Current default: `0.005`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.001, 0.005, 0.01]`
- Usage: research.logit, feature_registry.logit_price
- Status: **Active**
- Notes: New foundation

### `PaperTrader.matches.min_volume_spike`
- Category: entry
- Declaration: `paper_trader.py:PaperTrader.matches`
- Current default: `None`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[None, 1.5, 2.0, 3.0]`
- Usage: PaperTrader.matches
- Status: **Declared but unused**
- Notes: If set, matches() always returns False — LiveFeatures lacks volume_spike

### `PaperTrader.matches.max_time_to_resolution_hours`
- Category: entry
- Declaration: `paper_trader.py:PaperTrader.matches`
- Current default: `None`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[None, 24, 48, 72]`
- Usage: PaperTrader.matches
- Status: **Declared but unused**
- Notes: If set, matches() always returns False — LiveFeatures lacks TTR

## B — Fees, execution, risk, lifecycle

### `DEFAULT_TTR_BOUNDS`
- Category: lifecycle
- Declaration: `backtest.py:DEFAULT_TTR_BOUNDS`
- Current default: `[None, 24.0, 48.0, 72.0]`
- Existing sweep range: `[None, 24.0, 48.0, 72.0]`
- Recommended sweep range: `[None, 6.0, 12.0, 24.0, 48.0, 72.0, 168.0]`
- Usage: backtest.iter_grid_params
- Status: **Active**

### `CATEGORY_FEE_RATES`
- Category: fee
- Declaration: `paper_trader.py:CATEGORY_FEE_RATES`
- Current default: `{'crypto': 0.07, 'sports': 0.05, 'geopolitics': 0.0}`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Use FeeModel schedule / PIT versions; do not sweep rates as strategy knobs`
- Usage: paper_trader.resolve_fee_rate, research.fees
- Status: **Active**
- Notes: Duplicated into research.fees.CATEGORY_TAKER_FEE_RATES with FEE_MODEL_VERSION

### `PaperConfig.use_dynamic_fees`
- Category: fee
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `True`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[True]`
- Usage: PaperTrader.compute_taker_fee
- Status: **Active**

### `PaperConfig.fee_category`
- Category: fee
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `crypto`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Market-specific PIT category; not a free sweep axis`
- Usage: PaperTrader, cli --fee-category
- Status: **Active**

### `PaperConfig.fee_bps`
- Category: fee
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `0.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Legacy only when use_dynamic_fees=False`
- Usage: PaperTrader.compute_taker_fee
- Status: **Partially wired**
- Notes: Active only with --flat-fees

### `PaperConfig.spread_slippage_bps`
- Category: execution
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `50.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[10, 25, 50, 100, 200]`
- Usage: PaperTrader.fill modeling
- Status: **Active**

### `PaperConfig.bankroll`
- Category: risk
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `10000.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Not a strategy parameter`
- Usage: PaperTrader, cli --bankroll
- Status: **Active**

### `PaperConfig.kelly_fraction`
- Category: risk
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `0.25`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.1, 0.25, 0.5]`
- Usage: kelly_fraction, cli --kelly-fraction
- Status: **Active**

### `PaperConfig.max_position_pct`
- Category: risk
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `0.05`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.02, 0.05, 0.1]`
- Usage: PaperTrader sizing
- Status: **Active**

### `PaperConfig.max_open_positions`
- Category: risk
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `25`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[10, 15, 25, 40]`
- Usage: PaperTrader
- Status: **Active**

### `PaperConfig.cooldown_sec`
- Category: lifecycle
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `60.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[30, 60, 120, 300]`
- Usage: PaperTrader
- Status: **Active**

### `PaperConfig.resolve_poll_sec`
- Category: lifecycle
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `60.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[30, 60, 120]`
- Usage: PaperTrader resolution poll
- Status: **Active**

### `SwingConfig.bankroll`
- Category: risk
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `10000.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `N/A`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.position_pct`
- Category: risk
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `0.05`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0.02, 0.05, 0.1]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.stall_hours`
- Category: lifecycle
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `36.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[24, 36, 48, 72]`
- Usage: SwingTrader, profiles.py
- Status: **Active**
- Notes: CLI default aligned to 36.0

### `SwingConfig.max_open_positions`
- Category: risk
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `15`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[10, 15, 25]`
- Usage: SwingTrader, profiles.py
- Status: **Active**

### `SwingConfig.cooldown_sec`
- Category: lifecycle
- Declaration: `swing_trader.py:SwingConfig`
- Current default: `300.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[60, 120, 300]`
- Usage: SwingTrader, profiles.py
- Status: **Conflicting**
- Notes: profiles use 120

### `cli.swing-trade.stall_hours`
- Category: lifecycle
- Declaration: `cli.py:swing-trade --stall-hours`
- Current default: `36.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[24, 36, 48, 72]`
- Usage: cli._cmd_swing_trade
- Status: **Active**
- Notes: Aligned with SwingConfig.stall_hours=36.0 (was 48.0)

### `FEE_MODEL_VERSION`
- Category: fee
- Declaration: `research/fees.py:FEE_MODEL_VERSION`
- Current default: `2026-07-polymarket-v1`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Version pin only`
- Usage: compute_fill_fee, run metadata
- Status: **Active**
- Notes: New foundation

### `ExecutionConfig.latency_ms`
- Category: execution
- Declaration: `research/execution.py:ExecutionConfig`
- Current default: `50.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[0, 50, 100, 250, 500]`
- Usage: simulate_aggressive_fill, PaperConfig.latency_ms
- Status: **Partially wired**
- Notes: Present on PaperConfig and FillResult.meta, but does not delay quote selection or change fill price (inert until delayed-book model exists).

### `PaperConfig.use_book_walk`
- Category: execution
- Declaration: `paper_trader.py:PaperConfig`
- Current default: `False`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[False, True]`
- Usage: PaperTrader.apply_fill_price
- Status: **Partially wired**
- Notes: Opt-in L1 ask cross; CLI does not expose the flag; no historical L2

## C — Validation, calibration, CLI, ingest

### `find_edges.min_samples`
- Category: backtest/validation
- Declaration: `backtest.py:find_edges`
- Current default: `5`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[20, 50, 100]`
- Usage: backtest.find_edges, cli.find-edges --min-samples
- Status: **Active**
- Notes: CLI default 5 is aggressive for inference; recommend >=20 OOS

### `run_oos_edge_validation.min_samples`
- Category: backtest/validation
- Declaration: `backtest.py:run_oos_edge_validation`
- Current default: `20`
- Existing sweep range: `Not defined`
- Recommended sweep range: `[20, 50, 100]`
- Usage: backtest.run_oos_edge_validation
- Status: **Active**

### `DEFAULT_CHUNK_ROWS`
- Category: data/ingest
- Declaration: `ingest.py:DEFAULT_CHUNK_ROWS`
- Current default: `500000`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Not a research parameter`
- Usage: ingest.run_ingest, cli --chunk-rows
- Status: **Active**

### `cli.paper-trade.duration`
- Category: CLI/config
- Declaration: `cli.py:paper-trade --duration`
- Current default: `120.0`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Ops only (GHA uses 18000)`
- Usage: cli._cmd_paper_trade
- Status: **Active**

## D — Exogenous / other

### `exogenous.*`
- Category: exogenous
- Declaration: `research/exogenous.py`
- Current default: `None`
- Existing sweep range: `Not defined`
- Recommended sweep range: `Blocked until PIT providers exist`
- Usage: —
- Status: **Declared but unused**
- Notes: Provider stubs only
