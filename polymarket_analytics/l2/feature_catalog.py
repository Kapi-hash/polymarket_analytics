"""L2 feature catalog registry with formula metadata and sweep ranges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FeatureStatus = Literal["implemented", "partial", "blocked"]


@dataclass(frozen=True)
class FeatureCatalogEntry:
    name: str
    family: str
    formula: str
    defaults: dict[str, Any] = field(default_factory=dict)
    sweep_ranges: dict[str, Any] = field(default_factory=dict)
    status: FeatureStatus = "implemented"
    notes: str = ""


def _entry(
    name: str,
    family: str,
    formula: str,
    *,
    defaults: dict[str, Any] | None = None,
    sweep_ranges: dict[str, Any] | None = None,
    status: FeatureStatus = "implemented",
    notes: str = "",
) -> FeatureCatalogEntry:
    return FeatureCatalogEntry(
        name=name,
        family=family,
        formula=formula,
        defaults=dict(defaults or {}),
        sweep_ranges=dict(sweep_ranges or {}),
        status=status,
        notes=notes,
    )


FEATURE_CATALOG: dict[str, FeatureCatalogEntry] = {}

# Book state
for spec in [
    _entry("abs_spread", "book_state", "ask - bid"),
    _entry("rel_spread", "book_state", "(ask-bid)/mid"),
    _entry("logit_spread", "book_state", "logit(ask)-logit(bid)"),
    _entry("mid", "book_state", "(bid+ask)/2"),
    _entry("microprice", "book_state", "(ask*bid_sz + bid*ask_sz)/(bid_sz+ask_sz)"),
    _entry("multi_level_imbalance", "book_state", "Σ(bid-ask)/Σ(bid+ask)", sweep_ranges={"levels": [1, 3, 5]}),
    _entry("distance_weighted_imbalance", "book_state", "exp-decay weighted imbalance", defaults={"decay": 0.5}),
    _entry("book_slope", "book_state", "OLS slope of depth vs price"),
    _entry("book_convexity", "book_state", "second derivative of depth curve", status="partial"),
    _entry("depth_concentration", "book_state", "top_level/total_depth"),
    _entry("depth_entropy", "book_state", "-Σ p_i log p_i"),
    _entry("tob_vs_deep_divergence", "book_state", "tob_imb - deep_imb"),
    _entry("liquidity_walls", "book_state", "max(level)/avg(level)", defaults={"threshold": 3.0}),
    _entry("boundary_adjusted_depth", "book_state", "depth * p(1-p)"),
]:
    FEATURE_CATALOG[spec.name] = spec

# Order flow
for spec in [
    _entry("ofi", "order_flow", "Δbid - Δask at touch"),
    _entry("multi_level_ofi", "order_flow", "sum level OFI"),
    _entry("distance_weighted_ofi", "order_flow", "decay-weighted OFI"),
    _entry("signed_trade_imbalance", "order_flow", "(buy-sell)/(buy+sell)"),
    _entry("arrival_intensity", "order_flow", "count/window_sec"),
    _entry("add_rate", "order_flow", "adds/window", status="partial"),
    _entry("cancel_rate", "order_flow", "cancels/window", status="partial"),
    _entry("cancel_trade_ratio", "order_flow", "cancels/trades"),
    _entry("refill_rate", "order_flow", "Δdepth/depth/dt"),
    _entry("depletion_velocity", "order_flow", "(depth_before-depth_after)/dt"),
    _entry("imbalance_persistence", "order_flow", "autocorr(imb, lag=1)", status="partial"),
    _entry("flow_autocorr", "order_flow", "autocorr(signed_flow)"),
    _entry("vpin_proxy", "order_flow", "|buy-sell|/total_vol"),
    _entry("kyle_lambda_proxy", "order_flow", "Δp/signed_vol"),
    _entry("amihud_proxy", "order_flow", "|ret|/dollar_vol"),
    _entry("large_trade_impact", "order_flow", "Δmid after large trade", status="partial"),
    _entry("price_response_signed_vol", "order_flow", "Δp per signed volume"),
]:
    FEATURE_CATALOG[spec.name] = spec

# Resilience
for spec in [
    _entry("spread_recovery_time", "resilience", "ticks to baseline spread"),
    _entry("imbalance_mr_time", "resilience", "ticks to half imbalance"),
    _entry("refill_half_life", "resilience", "time to 50% depth restore", status="partial"),
    _entry("post_trade_adverse_selection", "resilience", "markout after trade"),
    _entry("liquidity_withdrawal_before_vol", "resilience", "depth drop pre vol spike", status="partial"),
    _entry("cancel_bursts", "resilience", "cancel cluster detection", status="partial"),
    _entry("resilience_conditional_p", "resilience", "recovery | prob bucket", status="partial"),
    _entry("resilience_conditional_ttr", "resilience", "recovery | TTR bucket", status="partial"),
]:
    FEATURE_CATALOG[spec.name] = spec

# Probability-aware
for spec in [
    _entry("logit_return", "probability_aware", "Δlogit(p)"),
    _entry("logit_vol", "probability_aware", "std(logit returns)"),
    _entry("logit_rsi", "probability_aware", "RSI in logit space", status="partial"),
    _entry("logit_ema", "probability_aware", "EMA in logit space", status="partial"),
    _entry("logit_bollinger", "probability_aware", "Bollinger in logit space", status="partial"),
    _entry("logit_atr", "probability_aware", "ATR in logit space", status="partial"),
    _entry("vol_scaled_p_var", "probability_aware", "vol * p(1-p)"),
    _entry("imbalance_x_p_var", "probability_aware", "imb * p(1-p)"),
    _entry("ttr_x_boundary", "probability_aware", "TTR near 0/1 boundary", status="partial"),
    _entry("spread_norm_upside", "probability_aware", "spread / remaining upside"),
]:
    FEATURE_CATALOG[spec.name] = spec

# PM structure
for spec in [
    _entry("yes_no_complement_deviation", "pm_structure", "yes_mid + no_mid - 1"),
    _entry("complete_set_residual", "pm_structure", "1 - yes_mid - no_mid"),
    _entry(
        "related_outcome_consistency",
        "pm_structure",
        "cross-outcome coherence",
        status="blocked",
        notes="Requires PIT outcome map",
    ),
    _entry("cross_market_lead_lag", "pm_structure", "lead/lag vs related", status="blocked"),
    _entry("event_shared_flow", "pm_structure", "correlated flow within event", status="partial"),
    _entry("ttr_decay", "pm_structure", "exp(-ttr/half_life)"),
    _entry("deadline_accel", "pm_structure", "acceleration near resolution", status="partial"),
    _entry("post_news_drift", "pm_structure", "drift after news", status="partial", notes="stub"),
    _entry("resolution_proximity", "pm_structure", "1 - ttr/max_ttr"),
    _entry("creation_age", "pm_structure", "age since market creation", status="partial"),
    _entry("category_conditional_move", "pm_structure", "move | category", status="partial"),
    _entry("probability_bucket_regime", "pm_structure", "regime by prob bucket", status="partial"),
]:
    FEATURE_CATALOG[spec.name] = spec


def get_feature(name: str) -> FeatureCatalogEntry:
    if name not in FEATURE_CATALOG:
        raise KeyError(f"unknown L2 feature: {name}")
    return FEATURE_CATALOG[name]


def list_features(*, family: str | None = None, status: FeatureStatus | None = None) -> list[FeatureCatalogEntry]:
    out = list(FEATURE_CATALOG.values())
    if family is not None:
        out = [f for f in out if f.family == family]
    if status is not None:
        out = [f for f in out if f.status == status]
    return out


def catalog_summary() -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for f in FEATURE_CATALOG.values():
        by_status[f.status] = by_status.get(f.status, 0) + 1
        by_family[f.family] = by_family.get(f.family, 0) + 1
    return {
        "total": len(FEATURE_CATALOG),
        "by_status": by_status,
        "by_family": by_family,
    }
