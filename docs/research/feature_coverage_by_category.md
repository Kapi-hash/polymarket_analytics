# Feature coverage by market category

| Category | Fee schedule | Implemented signals usable | Blocked / stub |
|---|---|---|---|
| crypto | taker feeRate 0.07 | logit, whale, spike, TTR hazard, swing tech, fee-aware EV | L2 OFI, exogenous |
| sports | 0.05 | same + sports fee | sports schedule PIT, injuries |
| finance | 0.04 | same | macro / vol index PIT |
| politics | 0.04 | same | polling PIT |
| economics | 0.05 | same | macro PIT |
| culture / weather / other / general | 0.05 | same | weather / social PIT |
| mentions / tech | 0.04 | same | news sentiment PIT |
| geopolitics | fee-free (0.0) | same; fees correctly zero | lead-lag / related markets |

Coverage is infrastructure-level; no category-specific alpha is claimed.
Lake currently has historical trades without reliable per-market category tags for all rows — fee category must come from market metadata when available.
