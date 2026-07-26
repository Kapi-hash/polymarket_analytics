"""Tests for research foundations: logit, fees, inventory, validation, execution."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from polymarket_analytics.research.execution import (
    BookLevel,
    order_tp_sl_tick,
    queue_ahead_fill_probability,
    simulate_aggressive_fill,
)
from polymarket_analytics.research.exogenous import provider_status
from polymarket_analytics.research.feature_registry import (
    apply_features,
    coverage_report,
    get_feature,
)
from polymarket_analytics.research.fees import (
    FEE_MODEL_VERSION,
    FeeModel,
    compute_fill_fee,
    round_fee,
    taker_fee,
)
from polymarket_analytics.research.inventory import (
    build_baseline_inventory,
    detect_cli_defaults,
    detect_conflicting_defaults,
    detect_declared_but_unused,
    detect_hardcoded_thresholds,
    write_inventory_artifacts,
)
from polymarket_analytics.research.logit import clamp_prob, logit, logit_edge, sigmoid_from_logit
from polymarket_analytics.research.microstructure import compute_ofi_from_levels
from polymarket_analytics.research.risk import RiskLimits, RiskState, check_entry_allowed
from polymarket_analytics.research.validation import (
    apply_bin_calibrator,
    benjamini_hochberg,
    fit_bin_calibrator,
    mask_event_leakage,
    purged_walk_forward_folds,
)


def test_logit_clamp_and_edge():
    assert clamp_prob(0.0) == pytest.approx(0.005)
    assert clamp_prob(1.0) == pytest.approx(0.995)
    z = logit(0.5)
    assert z == pytest.approx(0.0)
    assert sigmoid_from_logit(0.0) == pytest.approx(0.5)
    assert logit_edge(0.6, 0.4) > 0


def test_fee_taker_maker_fee_free_and_version():
    taker = compute_fill_fee(100, 0.5, role="taker", category="crypto")
    maker = compute_fill_fee(100, 0.5, role="maker", category="crypto")
    free = compute_fill_fee(100, 0.5, role="taker", category="geopolitics")
    assert taker["fee"] == pytest.approx(taker_fee(100, 0.5, fee_rate=0.07))
    assert maker["fee"] == 0.0
    assert free["fee"] == 0.0
    assert free["fee_free"] is True
    assert taker["fee_model_version"] == FEE_MODEL_VERSION
    assert round_fee(0.123456789, 6) == pytest.approx(0.123457)


def test_fee_model_pit_lookup():
    model = FeeModel()
    entry = model.lookup("crypto", as_of="2024-01-01")
    assert entry.taker_fee_rate == 0.07
    meta = model.metadata()
    assert meta["fee_model_version"] == FEE_MODEL_VERSION


def test_complete_set_residual_feature():
    df = pl.DataFrame({"yes_mid": [0.48, 0.60], "no_mid": [0.50, 0.45]})
    out = apply_features(df, ["complete_set_residual"])
    assert out["complete_set_residual"].to_list() == pytest.approx([-0.02, 0.05])


def test_ofi_multi_level():
    ofi = compute_ofi_from_levels([10, 5], [8, 5])
    assert ofi == pytest.approx((2 + 0) / (18 + 10))
    ofi2 = compute_ofi_from_levels([12, 5], [8, 4], prev_bid_sizes=[10, 5], prev_ask_sizes=[8, 5])
    assert ofi2 == pytest.approx((2 - 0) + (0 - (-1)))


def test_pit_calibrator_train_only():
    prices = [0.1, 0.2, 0.8, 0.9]
    outcomes = [0, 0, 1, 1]
    cal = fit_bin_calibrator(prices, outcomes, n_bins=2)
    # Apply to holdout without refitting
    assert apply_bin_calibrator(0.15, cal) is not None
    assert cal["counts"][0] == 2


def test_event_grouping_and_embargo_folds():
    folds = purged_walk_forward_folds(
        start="2023-01-01",
        end="2023-03-01",
        n_folds=2,
        train_days=14,
        test_days=7,
        embargo_days=2,
    )
    assert len(folds) >= 1
    assert folds[0].embargo_start == folds[0].train_end
    df = pl.DataFrame({"condition_id": ["a", "b", "a"], "x": [1, 2, 3]})
    purged = mask_event_leakage(df, {"a"})
    assert purged["condition_id"].to_list() == ["b"]


def test_tp_sl_same_tick_prefers_stop():
    # Both would hit if thresholds tiny relative to move
    decision = order_tp_sl_tick(
        entry=0.50,
        mark=0.60,
        take_profit_pct=0.10,
        stop_loss_pct=0.10,
        prefer="stop_first",
    )
    # Only TP hits here
    assert decision == "take_profit"
    both = order_tp_sl_tick(
        entry=0.50,
        mark=0.50,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        prefer="stop_first",
    )
    assert both == "stop_loss"


def test_queue_ahead_and_book_walk():
    assert queue_ahead_fill_probability(100, 50) == 0.0
    assert queue_ahead_fill_probability(100, 150) > 0
    levels = [BookLevel(0.51, 40), BookLevel(0.52, 60)]
    fill = simulate_aggressive_fill("buy", 50, levels)
    assert fill.filled_size == 50
    assert fill.levels_consumed == 2
    assert fill.meta["fee_model_version"] == FEE_MODEL_VERSION


def test_inventory_detectors(tmp_path: Path):
    cli_src = 'parser.add_argument("--bankroll", type=float, default=10_000.0)\n'
    defaults = detect_cli_defaults(cli_src)
    assert any(d["flag"] == "--bankroll" for d in defaults)

    hits = detect_hardcoded_thresholds("if x > 3.5:\n    pass\n", path="t.py")
    assert hits and hits[0]["literal"] == 3.5

    unused = detect_declared_but_unused(["a", "b", "c"], ["a", "c"])
    assert unused == ["b"]

    items = build_baseline_inventory()
    conflicts = detect_conflicting_defaults(items)
    assert conflicts
    arts = write_inventory_artifacts(tmp_path)
    assert arts["markdown"].exists()
    assert arts["json"].exists()


def test_feature_registry_coverage():
    cov = coverage_report()
    assert cov["n_features"] >= 10
    assert get_feature("logit_price") is not None
    assert get_feature("multi_level_ofi").status == "blocked"


def test_risk_gates():
    state = RiskState(
        equity=10_000,
        cash=10_000,
        peak_equity=12_000,
        n_open=0,
        gross_exposure=0,
        per_event_exposure={},
    )
    limits = RiskLimits(max_drawdown_halt_pct=10.0)
    ok, reason = check_entry_allowed(
        state, limits, notional=100, event_id="e1", now_ts=1000.0
    )
    assert ok is False and reason == "max_drawdown_halt"


def test_exogenous_providers_unavailable():
    status = provider_status()
    assert len(status) == 10
    assert all(not s["available"] for s in status)


def test_bh_fdr():
    flags = benjamini_hochberg([0.001, 0.01, 0.04, 0.20], alpha=0.05)
    assert flags[0] is True


def test_logit_technicals_and_arcsine():
    from polymarket_analytics.research.technical import arcsine_sqrt, logit_rsi, logit_ema_cross

    assert arcsine_sqrt(0.5) == pytest.approx(2.0 * math.asin(math.sqrt(0.5)))
    prices = [0.55 - 0.01 * i for i in range(40)]
    rsi = logit_rsi(prices, period=14)
    assert rsi is not None and 0 <= rsi <= 100
    cross = logit_ema_cross(prices, fast=5, slow=20)
    assert cross["ema_fast"] is not None and cross["ema_slow"] is not None

    rows = []
    for i in range(60):
        rows.append(
            {
                "token_id": "tok-a",
                "traded_at": f"2023-04-{(i // 24) + 10:02d}T{i % 24:02d}:00:00Z",
                "price": 0.40 + (i % 20) / 100.0,
            }
        )
    df = pl.DataFrame(rows).with_columns(
        pl.col("traded_at").str.to_datetime(time_zone="UTC")
    )
    out = apply_features(df, ["arcsine_price", "logit_rsi_ema_mad"])
    assert "arcsine_price" in out.columns
    assert "logit_rsi" in out.columns
    assert out["logit_rsi"].drop_nulls().len() > 0


def test_paper_trader_risk_and_tpsl_ordering():
    from polymarket_analytics.backtest import StrategyParams
    from polymarket_analytics.live_feed import LiveFeatures
    from polymarket_analytics.paper_trader import PaperConfig, PaperTrader

    trader = PaperTrader(
        config=PaperConfig(
            bankroll=10_000.0,
            max_drawdown_halt_pct=5.0,
            take_profit_pct=0.10,
            stop_loss_pct=0.10,
            tp_sl_prefer="stop_first",
            cooldown_sec=0.0,
            min_oos_ev_pct=0.0,
            require_persists=False,
        ),
        strategies=[
            (
                StrategyParams(price_bucket=None, min_whale_ratio=None, momentum_1h="any", side="BUY"),
                50.0,
                0.70,
            )
        ],
    )
    # Force drawdown halt via peak >> equity
    trader.peak_equity = 20_000.0
    trader.cash = 10_000.0
    feat = LiveFeatures(
        token_id="t1",
        condition_id="c1",
        ts=1_000.0,
        price=0.45,
        size=10.0,
        whale_ratio=5.0,
        momentum_1h=0.01,
        n_trades_1h=5,
        best_ask=0.46,
    )
    assert trader.on_features(feat) is None
    assert trader.risk_rejects >= 1

    # Reset for TP/SL same-tick preference
    trader2 = PaperTrader(
        config=PaperConfig(
            bankroll=10_000.0,
            take_profit_pct=0.0,
            stop_loss_pct=0.0,
            tp_sl_prefer="stop_first",
            cooldown_sec=0.0,
        ),
        strategies=[],
    )
    from polymarket_analytics.paper_trader import PaperPosition

    trader2.positions.append(
        PaperPosition(
            position_id="p1",
            token_id="t2",
            condition_id="c2",
            strategy_label="x",
            entry_ts=0.0,
            entry_price_raw=0.50,
            entry_price_fill=0.50,
            shares=10.0,
            notional=5.0,
            fees=0.0,
            mark_price=0.50,
        )
    )
    trader2.cash -= 5.0
    trader2._evaluate_stops(1.0)
    assert trader2.positions[0].status == "stop_loss"
