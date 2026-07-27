"""Versioned FeatureSpec registry (Polars-first)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Final, Literal, Sequence

import polars as pl

from polymarket_analytics.research.logit import DEFAULT_HALF_TICK, attach_logit_columns, logit_expr
from polymarket_analytics.research.technical import attach_arcsine, attach_logit_technicals

FeatureCategory = Literal[
    "prediction_market",
    "microstructure",
    "technical",
    "risk",
    "exogenous",
    "legacy",
]

ComputeFn = Callable[..., pl.DataFrame]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str
    category: FeatureCategory
    strategy_families: tuple[str, ...]
    required_columns: tuple[str, ...]
    required_sources: tuple[str, ...]
    defaults: dict[str, Any] = field(default_factory=dict)
    sweep_parameters: dict[str, Any] = field(default_factory=dict)
    point_in_time_safe: bool = True
    status: Literal["implemented", "partial", "blocked", "stub"] = "implemented"
    notes: str = ""
    compute: ComputeFn | None = None

    def key(self) -> str:
        return f"{self.name}@{self.version}"


def _compute_logit_price(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    half_tick = float(kwargs.get("half_tick", DEFAULT_HALF_TICK))
    return attach_logit_columns(df, half_tick=half_tick)


def _compute_logit_edge_proxy(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    """Proxy: -logit(price) as long-edge vs fair-coin; real model probs need calibrator."""
    half_tick = float(kwargs.get("half_tick", DEFAULT_HALF_TICK))
    if "price" not in df.columns:
        return df
    out = df.with_columns(logit_expr(pl.col("price"), half_tick=half_tick, alias="_lp"))
    # Edge vs 0.5 in logit space (0.5 → logit 0)
    return out.with_columns((-pl.col("_lp")).alias("logit_edge_vs_half")).drop("_lp")


def _compute_whale_ratio(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    """Pass-through if already present; else leave unchanged."""
    return df


def _compute_volume_spike(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    return df


def _compute_decay_velocity(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    return df


def _compute_ttr_hazard(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    """Simple TTR information-hazard proxy: 1 / (1 + TTR_hours)."""
    if "time_to_resolution_hours" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("ttr_info_hazard"))
    ttr = pl.col("time_to_resolution_hours").fill_null(0.0).clip(lower_bound=0.0)
    return df.with_columns((1.0 / (1.0 + ttr)).alias("ttr_info_hazard"))


def _compute_complete_set_residual(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    """
    YES+NO mid residual vs 1.0 when both sides present.

    Requires columns yes_mid and no_mid (or price_yes/price_no). Otherwise null.
    """
    yes_col = "yes_mid" if "yes_mid" in df.columns else ("price_yes" if "price_yes" in df.columns else None)
    no_col = "no_mid" if "no_mid" in df.columns else ("price_no" if "price_no" in df.columns else None)
    if yes_col is None or no_col is None:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("complete_set_residual"))
    return df.with_columns(
        (pl.col(yes_col) + pl.col(no_col) - 1.0).alias("complete_set_residual")
    )


def _compute_arcsine_price(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    return attach_arcsine(df)


def _compute_logit_technicals(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    return attach_logit_technicals(
        df,
        rsi_period=int(kwargs.get("rsi_period", 14)),
        ema_fast=int(kwargs.get("ema_fast", 5)),
        ema_slow=int(kwargs.get("ema_slow", 20)),
        band_window=int(kwargs.get("band_window", 48)),
        band_k=float(kwargs.get("band_k", 2.5)),
        half_tick=float(kwargs.get("half_tick", DEFAULT_HALF_TICK)),
    )


def _stub_blocked(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    return df


_REGISTRY: dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> FeatureSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get_feature(name: str) -> FeatureSpec | None:
    return _REGISTRY.get(name)


def list_features(
    *,
    category: FeatureCategory | None = None,
    status: str | None = None,
) -> list[FeatureSpec]:
    out = list(_REGISTRY.values())
    if category is not None:
        out = [f for f in out if f.category == category]
    if status is not None:
        out = [f for f in out if f.status == status]
    return sorted(out, key=lambda f: (f.category, f.name))


def coverage_report() -> dict[str, Any]:
    feats = list(_REGISTRY.values())
    by_cat: dict[str, dict[str, int]] = {}
    for f in feats:
        bucket = by_cat.setdefault(f.category, {"implemented": 0, "partial": 0, "blocked": 0, "stub": 0, "total": 0})
        bucket[f.status] = bucket.get(f.status, 0) + 1
        bucket["total"] += 1
    return {
        "n_features": len(feats),
        "by_category": by_cat,
        "implemented": [f.name for f in feats if f.status == "implemented"],
        "partial": [f.name for f in feats if f.status == "partial"],
        "blocked": [f.name for f in feats if f.status in {"blocked", "stub"}],
    }


# ---------------------------------------------------------------------------
# Seed registry
# ---------------------------------------------------------------------------

register(
    FeatureSpec(
        name="logit_price",
        version="1.0.0",
        category="prediction_market",
        strategy_families=("edge", "calibration", "swing"),
        required_columns=("price",),
        required_sources=("trades",),
        defaults={"half_tick": DEFAULT_HALF_TICK},
        sweep_parameters={"half_tick": [0.001, 0.005, 0.01]},
        point_in_time_safe=True,
        status="implemented",
        notes="logit(p) with half-tick clamp",
        compute=_compute_logit_price,
    )
)
register(
    FeatureSpec(
        name="logit_edge_vs_half",
        version="1.0.0",
        category="prediction_market",
        strategy_families=("edge", "calibration"),
        required_columns=("price",),
        required_sources=("trades",),
        defaults={"half_tick": DEFAULT_HALF_TICK},
        sweep_parameters={},
        point_in_time_safe=True,
        status="implemented",
        notes="Proxy edge vs fair coin in logit space; replace with calibrated p_model",
        compute=_compute_logit_edge_proxy,
    )
)
register(
    FeatureSpec(
        name="complete_set_residual",
        version="1.0.0",
        category="prediction_market",
        strategy_families=("arb", "neg_risk"),
        required_columns=("yes_mid", "no_mid"),
        required_sources=("book_or_paired_trades",),
        defaults={},
        sweep_parameters={"entry_threshold": [0.01, 0.02, 0.03]},
        point_in_time_safe=True,
        status="partial",
        notes="Requires paired YES/NO mids; null if columns absent",
        compute=_compute_complete_set_residual,
    )
)
register(
    FeatureSpec(
        name="neg_risk_residual",
        version="1.0.0",
        category="prediction_market",
        strategy_families=("neg_risk",),
        required_columns=("neg_risk",),
        required_sources=("markets", "multi_outcome_mids"),
        defaults={},
        sweep_parameters={},
        point_in_time_safe=True,
        status="blocked",
        notes="Needs multi-outcome mid vector sum vs 1.0; no PIT multi-outcome book in lake",
        compute=_stub_blocked,
    )
)
register(
    FeatureSpec(
        name="ttr_info_hazard",
        version="1.0.0",
        category="prediction_market",
        strategy_families=("edge", "lifecycle"),
        required_columns=("time_to_resolution_hours",),
        required_sources=("trades", "markets"),
        defaults={},
        sweep_parameters={},
        point_in_time_safe=True,
        status="implemented",
        notes="Simple 1/(1+TTR) hazard proxy",
        compute=_compute_ttr_hazard,
    )
)
register(
    FeatureSpec(
        name="arcsine_price",
        version="1.0.0",
        category="technical",
        strategy_families=("swing", "edge"),
        required_columns=("price",),
        required_sources=("trades",),
        defaults={"half_tick": DEFAULT_HALF_TICK},
        sweep_parameters={},
        point_in_time_safe=True,
        status="implemented",
        notes="y=2 arcsin(sqrt(p)) variance-stabilizing transform",
        compute=_compute_arcsine_price,
    )
)
register(
    FeatureSpec(
        name="logit_rsi_ema_mad",
        version="1.0.0",
        category="technical",
        strategy_families=("swing",),
        required_columns=("price", "token_id"),
        required_sources=("trades",),
        defaults={
            "rsi_period": 14,
            "ema_fast": 5,
            "ema_slow": 20,
            "band_window": 48,
            "band_k": 2.5,
        },
        sweep_parameters={
            "rsi_period": [5, 8, 14, 21, 34],
            "ema_fast": [3, 5, 8, 12],
            "ema_slow": [10, 20, 24, 48],
            "band_k": [1.5, 2.0, 2.5, 3.0],
        },
        point_in_time_safe=True,
        status="implemented",
        notes="Logit-space RSI, EMA cross, median/MAD bands; panel-grouped",
        compute=_compute_logit_technicals,
    )
)
register(
    FeatureSpec(
        name="whale_ratio",
        version="1.0.0",
        category="technical",
        strategy_families=("paper", "edge"),
        required_columns=("size_mean_1h", "size_median_24h"),
        required_sources=("trades",),
        defaults={"divergence_threshold": 3.0},
        sweep_parameters={"min_whale_ratio": [None, 2.0, 3.0, 4.0, 5.0]},
        point_in_time_safe=True,
        status="partial",
        notes=(
            "Computed in features.py; registry compute is pass-through only. "
            "Threshold WHALE_RATIO_DIVERGENCE_THRESHOLD=3.0"
        ),
        compute=_compute_whale_ratio,
    )
)
register(
    FeatureSpec(
        name="volume_spike_1h_24h",
        version="1.0.0",
        category="technical",
        strategy_families=("edge", "paper"),
        required_columns=("volume_1h", "volume_24h"),
        required_sources=("trades",),
        defaults={},
        sweep_parameters={"min_volume_spike": [None, 1.5, 2.0, 3.0]},
        point_in_time_safe=True,
        status="partial",
        notes="Computed in features.py; registry compute is pass-through only",
        compute=_compute_volume_spike,
    )
)
register(
    FeatureSpec(
        name="decay_adjusted_velocity",
        version="1.0.0",
        category="technical",
        strategy_families=("edge", "swing"),
        required_columns=("price_delta_1h", "time_to_resolution_hours"),
        required_sources=("trades", "markets"),
        defaults={"ttr_floor": 0.1},
        sweep_parameters={"ttr_floor": [0.05, 0.1, 0.25]},
        point_in_time_safe=True,
        status="partial",
        notes="Computed in features.py; registry compute is pass-through only",
        compute=_compute_decay_velocity,
    )
)
register(
    FeatureSpec(
        name="multi_level_ofi",
        version="1.0.0",
        category="microstructure",
        strategy_families=("swing", "mm"),
        required_columns=("bid_size_l1", "ask_size_l1"),
        required_sources=("l2_book"),
        defaults={"levels": 5},
        sweep_parameters={"levels": [1, 5, 10]},
        point_in_time_safe=True,
        status="blocked",
        notes="No historical L2 depth snapshots in lake; see microstructure.compute_ofi_from_levels",
        compute=_stub_blocked,
    )
)
register(
    FeatureSpec(
        name="book_resilience",
        version="1.0.0",
        category="microstructure",
        strategy_families=("execution", "mm"),
        required_columns=("bid_depth", "ask_depth"),
        required_sources=("l2_book"),
        defaults={},
        sweep_parameters={},
        point_in_time_safe=True,
        status="blocked",
        notes="Requires depth recovery after trade events",
        compute=_stub_blocked,
    )
)
register(
    FeatureSpec(
        name="fill_prob_adverse_markout",
        version="1.0.0",
        category="microstructure",
        strategy_families=("execution",),
        required_columns=("mid", "fill_price"),
        required_sources=("trades", "quotes"),
        defaults={"markout_horizon_s": 60.0},
        sweep_parameters={"markout_horizon_s": [5, 30, 60, 300]},
        point_in_time_safe=True,
        status="partial",
        notes="Helpers in execution.py; full LOB queue model blocked without event feed",
        compute=_stub_blocked,
    )
)
register(
    FeatureSpec(
        name="cross_market_lead_lag",
        version="1.0.0",
        category="prediction_market",
        strategy_families=("cross_market",),
        required_columns=("condition_id", "price"),
        required_sources=("related_markets_graph",),
        defaults={},
        sweep_parameters={"lag_seconds": [30, 60, 300]},
        point_in_time_safe=True,
        status="blocked",
        notes="Needs curated related-market graph + aligned clocks",
        compute=_stub_blocked,
    )
)
register(
    FeatureSpec(
        name="calibration_residual",
        version="1.0.0",
        category="prediction_market",
        strategy_families=("calibration", "edge"),
        required_columns=("price", "token_won"),
        required_sources=("resolved_trades",),
        defaults={"n_bins": 10},
        sweep_parameters={"n_bins": [5, 10, 20]},
        point_in_time_safe=True,
        status="partial",
        notes=(
            "Calibrator helpers live in validation.py; registry compute is a no-op. "
            "Not attached by apply_features / lake compute-features."
        ),
        compute=_stub_blocked,
    )
)
register(
    FeatureSpec(
        name="inventory_risk_cap",
        version="1.0.0",
        category="risk",
        strategy_families=("paper", "swing"),
        required_columns=(),
        required_sources=("runtime"),
        defaults={"max_position_pct": 0.05, "max_open_positions": 25},
        sweep_parameters={"max_position_pct": [0.02, 0.05, 0.10]},
        point_in_time_safe=True,
        status="partial",
        notes=(
            "RiskLimits wired into PaperTrader only; registry compute is a no-op. "
            "SwingTrader does not call check_entry_allowed."
        ),
        compute=_stub_blocked,
    )
)
register(
    FeatureSpec(
        name="exogenous_news_sentiment",
        version="1.0.0",
        category="exogenous",
        strategy_families=("event_driven",),
        required_columns=(),
        required_sources=("news_provider",),
        defaults={},
        sweep_parameters={},
        point_in_time_safe=False,
        status="stub",
        notes="Provider interface only — no PIT news corpus",
        compute=_stub_blocked,
    )
)

FEATURE_REGISTRY: Final[dict[str, FeatureSpec]] = _REGISTRY


def apply_features(
    df: pl.DataFrame,
    names: Sequence[str],
    **kwargs: Any,
) -> pl.DataFrame:
    """Apply registered compute callables in order."""
    out = df
    for name in names:
        spec = get_feature(name)
        if spec is None or spec.compute is None:
            continue
        out = spec.compute(out, **{**spec.defaults, **kwargs})
    return out
