# Declared-but-unused / partially wired

## Unused
- `exogenous.*`: Provider stubs only

## Partially wired
- `PaperConfig.fee_bps`: Active only with --flat-fees
- `StrategyParams.require_price_volume_divergence`: Field exists but iter_grid_params never sets True — not in default grid
- `momentum_6h grid`: Axis declared but default grid freezes momentum_6h at any