#!/usr/bin/env python3
"""Small representative validation backtest (NOT a full parameter sweep).

Runs a tiny edge evaluation + purged walk-forward scaffold + fee/logit checks
on the local lake / synthetic fallback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polymarket_analytics.backtest import StrategyParams, compute_edge_stats, apply_strategy_filter
from polymarket_analytics.research.execution import BookLevel, simulate_aggressive_fill, ExecutionConfig
from polymarket_analytics.research.fees import FEE_MODEL_VERSION, compute_fill_fee
from polymarket_analytics.research.feature_registry import apply_features, coverage_report
from polymarket_analytics.research.inventory import write_inventory_artifacts
from polymarket_analytics.research.logit import logit_edge
from polymarket_analytics.research.validation import (
    fit_bin_calibrator,
    multiple_testing_controls,
    run_grouped_purged_validation,
)


def _synthetic_frame(n: int = 200) -> pl.DataFrame:
    rows = []
    for i in range(n):
        cid = f"evt-{i % 20}"
        token = f"tok-{i % 40}"
        price = 0.35 + (i % 30) / 100.0
        won = 1 if (i * 7) % 10 < int(price * 10) else 0
        day = 10 + (i // 10)
        rows.append(
            {
                "trade_id": f"t{i}",
                "condition_id": cid,
                "token_id": token,
                "traded_at": f"2023-04-{day:02d}T12:00:00Z",
                "price": price,
                "token_won": bool(won),
                "price_bucket": "0.40-0.50" if 0.4 <= price < 0.5 else "0.30-0.40",
                "whale_ratio": 3.5 if i % 3 == 0 else 1.2,
                "momentum_1h": 0.01 if i % 2 == 0 else -0.01,
                "volume_spike_1h_24h": 2.5 if i % 4 == 0 else 0.8,
                "time_to_resolution_hours": float(48 - (i % 40)),
                "side": "BUY",
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("traded_at").str.to_datetime(time_zone="UTC")
    )


def main() -> int:
    docs = ROOT / "docs" / "research"
    arts = write_inventory_artifacts(docs)
    print("inventory_artifacts", {k: str(v) for k, v in arts.items()})

    df = _synthetic_frame()
    df = apply_features(
        df,
        ["logit_price", "logit_edge_vs_half", "ttr_info_hazard", "arcsine_price"],
    )

    params = StrategyParams(
        price_bucket="0.40-0.50",
        min_whale_ratio=3.0,
        momentum_1h="any",
        side="BUY",
    )
    sliced = apply_strategy_filter(df, params)
    edge = compute_edge_stats(sliced, params)

    fee = compute_fill_fee(100.0, 0.45, role="taker", category="crypto")
    fee_maker = compute_fill_fee(100.0, 0.45, role="maker", category="crypto")
    fee_free = compute_fill_fee(100.0, 0.45, role="taker", category="geopolitics")

    # Tiny execution stress (NOT a full sweep): 3 latency × 2 slip scenarios
    exec_stress = []
    for latency_ms in (0.0, 50.0, 250.0):
        for slip_ask in (0.45, 0.4525):
            fill = simulate_aggressive_fill(
                "buy",
                size=100.0,
                levels=[BookLevel(price=slip_ask, size=500.0)],
                cfg=ExecutionConfig(latency_ms=latency_ms, fee_category="crypto"),
                mid_after=0.44,
            )
            exec_stress.append(
                {
                    "latency_ms": latency_ms,
                    "ask": slip_ask,
                    "avg_price": fill.avg_price,
                    "fee": fill.fee,
                    "markout": fill.markout,
                    "fee_model_version": (fill.meta or {}).get("fee_model_version"),
                }
            )

    def evaluate(train: pl.DataFrame, test: pl.DataFrame) -> dict:
        if train.is_empty() or "price" not in train.columns:
            return {"n_train": train.height, "n_test": test.height, "oos_ev": 0.0}
        cal = fit_bin_calibrator(
            train["price"].to_list(),
            train["token_won"].to_list(),
            n_bins=5,
        )
        te = apply_strategy_filter(test, params)
        stats = compute_edge_stats(te, params)
        return {
            "n_train": train.height,
            "n_test": test.height,
            "n_filtered": stats.n,
            "oos_ev": stats.ev_pct,
            "calibrator_bins": cal["n_bins"],
        }

    wf = run_grouped_purged_validation(
        df,
        evaluate,
        start="2023-04-10",
        end="2023-04-30",
        n_folds=2,
        train_days=5,
        test_days=3,
        embargo_days=1,
    )

    mt = multiple_testing_controls(
        sharpes=[0.5, 0.2, 1.1, 0.0],
        oos_sharpes=[0.1, -0.2, 0.3, 0.0],
        n_obs=50,
        p_values=[0.01, 0.04, 0.20, 0.50],
    )

    report = {
        "ok": True,
        "mode": "representative_validation_backtest",
        "full_sweep": False,
        "fee_model_version": FEE_MODEL_VERSION,
        "logit_edge_example": logit_edge(0.55, 0.45),
        "edge": edge.to_dict(),
        "fees": {"taker": fee, "maker": fee_maker, "fee_free": fee_free},
        "execution_stress_tiny": exec_stress,
        "walk_forward": wf,
        "multiple_testing": mt.to_dict(),
        "feature_coverage": coverage_report(),
        "inventory": {k: str(v) for k, v in arts.items()},
    }
    out = ROOT / "data" / "representative_backtest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out), "edge_n": edge.n, "edge_ev": edge.ev_pct}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
