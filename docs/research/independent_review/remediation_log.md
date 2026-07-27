# Remediation log (independent review)

## 2026-07-26 — honesty / integrity fixes

1. **Feature registry status inflation**
   - `calibration_residual`, `inventory_risk_cap` → `partial` (stub compute)
   - `whale_ratio`, `volume_spike_1h_24h`, `decay_adjusted_velocity` → `partial` (pass-through)

2. **Inventory accuracy**
   - `ExecutionConfig.latency_ms` → Partially wired (meta only; inert on fill)
   - Added `PaperConfig.use_book_walk`, paper matches spike/TTR unused axes

3. **Fee-aware historical simulate**
   - `simulate_strategy(..., fee_category=, spread_slippage_bps=)` opt-in net PnL
   - Default path remains gross for backward compatibility
   - Unit test asserts fees+slippage reduce PnL

4. **Not remediated (requires data / methodology authority)**
   - Historical L2/L3 books
   - Per-market fee category + true fee schedule history
   - Wiring purged WF into CLI backtest as default
   - Swing fee + risk + stop_first alignment (material trading-policy change)
   - Expanding beyond 81-event sample lake

## Decision impact
Critical Gates 4, 5, 7, 8 remain FAIL → **BLOCKED** for designed full sweep.
