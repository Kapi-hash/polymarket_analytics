# Feature / parameter attribution (restricted diagnostic only)

**Scope:** mid-fill trade prints + FALLBACK crypto fee + 50bps slip on 81-event lake.
**Not definitive.** No L2 execution; fee category unknown.

## Highest-value feature families (train, fallback net)

| Family | Observation |
|---|---|
| Whale ratio (>2–3) + mid buckets (0.30–0.50) | Dominates top train candidates |
| Volume spike (>1.5–2) | Often co-selected with whale; modest incremental lift |
| Momentum 1h = pos | Helps in 0.30–0.40 band; less decisive in 0.40–0.50 |
| TTR gates | Rarely needed for top mid-bucket whale configs |

## Features that added no incremental value here

- Registry logit / arcsine / logit-RSI (not in this restricted grid — lake path still legacy features only)
- Complete-set residual (no YES/NO mids in lake)
- OFI / book resilience / lead-lag / exogenous (blocked — no data)

## Features that disappear / reverse after costs

- Gross EV systematically higher than fallback net EV (fees + 50bps). Ranking order among whale mid-bucket configs was mostly preserved, but absolute expectancy compressed.
- High-price buckets (0.70–0.80) showed thin n/events — unstable under fees.

## Regime notes

| Regime | Result |
|---|---|
| Best price bands | 0.30–0.40 and 0.40–0.50 |
| Worst in this grid | Extreme buckets not swept; high buckets under-powered |
| Walk-forward | Fold 2 (~Mar 2023 window) strongly negative for all top-5 — **regime fragility** |
| Event count | Train 44 / purged test 37 — below preferred 100 |

## Parameter stability

- Whale threshold 2–3 and spike 1.5–2 form a **neighborhood** with same-sign train EV.
- Point estimates (exact spike=2 vs 1.5) are **not** uniquely identified.
- Momentum×whale interactions matter in 0.30–0.40; less so in 0.40–0.50.

## Maker vs taker / latency / queue

- Not estimable without books. Maker=0 fee assumption would inflate results; not claimed.
- Latency inert in codebase — sensitivity not identifiable on lake.

## Conclusion

Whale + mid-bucket filters show **gross and fallback-net positive slices**, but fold-level instability and missing execution/fee PIT metadata prevent attributing a durable edge.
