# Outcome sweep report

**Authorization:** PASS-BOUNDED (exact historical zero fees; bounded mid-print execution).
**Robust candidates:** 0
**Microstructure / swing historical ranking:** not run (blocked without L2).

## Coverage

- Train rows: 57,907 | purged test rows: 32,250
- Independent events train/test: 466 / 242
- Attempts: 270 | sufficient sample: 190 | promoted to final: 91
- Fee: `zero_fee_historical` confidence=`exact` model=`clob-v1-historical-2026-07-27`
- Execution: `adverse_mid_print_slippage_sensitivity` select slip=50.0 bps
- Frozen train end: `2023-06-01T00:00:00+00:00`

## Robustness summary

- rejected: 80
- fragile: 11

## Top locked-test configurations (by net EV @ 50bps slip)

| Rank | Label | Test EV% | N | Events | Sharpe | MaxDD | Pos folds | Leave5 mean | Class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `bucket=0.50-0.60 AND spike>2 AND ttr<48h AND side=BUY` | 16.19 | 95 | 10 | 3.54 | 8.12 | 0.67 | -1.7074699999999996 | **fragile** |
| 2 | `bucket=0.50-0.60 AND spike>1.5 AND ttr<48h AND side=BUY` | 14.45 | 109 | 10 | 3.33 | 9.29 | 0.67 | -1.6180199999999996 | **fragile** |
| 3 | `bucket=0.50-0.60 AND spike>2 AND ttr<72h AND side=BUY` | 13.95 | 116 | 12 | 3.26 | 9.87 | 0.67 | -1.724985714285714 | **fragile** |
| 4 | `bucket=0.50-0.60 AND ttr<48h AND side=BUY` | 12.47 | 175 | 11 | 3.60 | 13.35 | 0.67 | -2.1029159966666664 | **fragile** |
| 5 | `bucket=0.30-0.40 AND whale>3 AND mom1h=pos AND side=BUY` | 10.01 | 22 | 10 | 0.94 | 1.66 | 0.67 | -0.6914399999999998 | **fragile** |
| 6 | `bucket=0.40-0.50 AND side=BUY` | 3.36 | 258 | 48 | 1.08 | 14.31 | 1.00 | -0.41714874372093 | **fragile** |
| 7 | `bucket=0.60-0.70 AND side=BUY` | 3.05 | 412 | 51 | 1.34 | 25.68 | 0.67 | -0.9260703026086954 | **fragile** |
| 8 | `bucket=0.30-0.40 AND spike>1.5 AND whale>3 AND mom1h=pos AND side=BUY` | 1.88 | 19 | 9 | 0.17 | 1.66 | 0.67 | -0.7688249999999999 | **rejected** |
| 9 | `bucket=0.30-0.40 AND spike>2 AND whale>3 AND mom1h=pos AND side=BUY` | 1.88 | 19 | 9 | 0.17 | 1.66 | 0.67 | -0.7688249999999999 | **rejected** |
| 10 | `bucket=0.40-0.50 AND spike>1.5 AND side=BUY` | 1.85 | 173 | 44 | 0.49 | 7.53 | 1.00 | -0.31100117897435875 | **fragile** |

## Why none are robust

Every promoted config with positive locked-test EV fails at least one robustness gate:

1. **Leave-five-largest-winners** flips mean event PnL negative (edge concentrated in ≤5 events).
2. Test event counts for headline configs are typically 9–12 (underpowered vs preferred ≥100 regime-local).
3. Worst walk-forward fold is largely negative while best fold is strongly positive (regime fragility).
4. Bootstrap `p_positive` is only ~0.63–0.68 (CI includes zero / negative).

## Slippage sensitivity (headline)

Example `bucket=0.50-0.60 AND spike>2 AND ttr<48h AND side=BUY`:
- ev_slip_0: 16.46%
- ev_slip_100: 15.91%
- ev_slip_200: 15.36%
- ev_slip_25: 16.33%
- ev_slip_50: 16.19%

## Multiple-testing notes

- FDR: unavailable (no parametric p-values) — correctly not fabricated.
- DSR / PBO computed but **not used for promotion**; PBO matrix orientation treats candidates as partitions in this run — interpret as diagnostic only.

## Verdict

No reliable outcome edge suitable for paper trading was found under honest fee + bounded execution assumptions.

