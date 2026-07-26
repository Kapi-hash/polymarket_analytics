# Previous vs new feature comparison

## Previous (legacy Phase 2 / 2.5 / swing)

| Feature | Status before |
|---|---|
| momentum / vol / volume 1h·6h·24h | Implemented |
| volume_spike_1h_24h | Implemented |
| whale_ratio + price_volume_divergence | Implemented |
| decay_adjusted_velocity | Implemented |
| price_bucket + TTR hours | Implemented |
| RSI / Bollinger / EMA / Hurst / ATR (swing) | Implemented |
| Book imbalance (live depth proxy) | Partial (live only) |
| Dynamic taker fee curve | Implemented (paper) |
| Maker fee / fee versioning / PIT schedule | Missing |
| Logit-space edges | Missing |
| Multi-level OFI | Missing |
| Complete-set / neg-risk residuals | Missing |
| Purged event walk-forward + DSR/PBO/FDR | Missing |
| Exogenous PIT series | Missing |

## New in this expansion

| Feature | Status now |
|---|---|
| logit_price / logit_edge_vs_half | Implemented |
| ttr_info_hazard | Implemented |
| complete_set_residual | Partial (needs YES/NO mids) |
| FeeModel + FEE_MODEL_VERSION + maker/taker/rounding | Implemented |
| Execution walk / queue-ahead / TP-SL ordering | Implemented (helpers) |
| FeatureSpec registry | Implemented |
| Hierarchical YAML recommended priors | Implemented (advisory) |
| Purged WF + calibrator + DSR/PBO/FDR scaffold | Implemented |
| Risk lifecycle gates | Implemented |
| multi_level_ofi / book_resilience / lead-lag / exogenous | Blocked / stub (honest) |

No alpha is claimed. Coverage is infrastructure + highest-priority foundations.
