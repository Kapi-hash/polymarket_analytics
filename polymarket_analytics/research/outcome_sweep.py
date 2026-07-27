"""Staged outcome strategy sweep on canonical features."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Literal

import polars as pl

from polymarket_analytics.backtest import (
    StrategyParams,
    _max_drawdown,
    _sharpe_ratio,
    apply_strategy_filter,
    compute_edge_stats,
)
from polymarket_analytics.research.historical_fees import FEE_MODEL_VERSION, compute_historical_fill_fee
from polymarket_analytics.research.outcome_execution import OutcomeExecConfig
from polymarket_analytics.research.stats_valid import (
    bh_fdr,
    cluster_bootstrap_ci,
    deflated_sharpe_valid,
    event_clustered_returns,
    leave_five_largest_out,
    pbo_from_matrices,
)
from polymarket_analytics.research.validation import mask_event_leakage, purged_walk_forward_folds
from polymarket_analytics.schema import PRICE_BUCKET_LABELS

Robustness = Literal["robust_candidate", "promising_underpowered", "fragile", "rejected"]


def _fee_aware_pnl(won: bool, price: float, traded_at: Any, slip_bps: float) -> tuple[float, float, float]:
    """Return (pnl, fill_px, fee) for 1 share taker buy under historical fee regime."""
    slip = max(float(slip_bps), 0.0) / 10_000.0
    fill_px = min(max(float(price) * (1.0 + slip), 0.01), 0.99)
    fee_info = compute_historical_fill_fee(1.0, fill_px, as_of=traded_at, role="taker")
    fee = float(fee_info["fee"])
    pnl = (1.0 if won else 0.0) - fill_px - fee
    return pnl, fill_px, fee


def _simulate_candidate_ledger(
    df: pl.DataFrame,
    params: StrategyParams,
    *,
    slip_bps: float,
    event_col: str,
) -> pl.DataFrame:
    filtered = apply_strategy_filter(df, params)
    if filtered.is_empty() or "token_won" not in filtered.columns:
        return pl.DataFrame(
            schema={
                "trade_id": pl.Utf8,
                "condition_id": pl.Utf8,
                event_col: pl.Utf8,
                "traded_at": pl.Datetime("us", "UTC"),
                "price": pl.Float64,
                "fill_price": pl.Float64,
                "fee": pl.Float64,
                "pnl": pl.Float64,
                "token_won": pl.Boolean,
            }
        )

    rows: list[dict[str, Any]] = []
    for row in filtered.sort("traded_at").iter_rows(named=True):
        won = row.get("token_won")
        if won is None:
            continue
        pnl, fill_px, fee = _fee_aware_pnl(bool(won), float(row["price"]), row.get("traded_at"), slip_bps)
        rows.append(
            {
                "trade_id": row.get("trade_id"),
                "condition_id": row.get("condition_id"),
                event_col: row.get(event_col),
                "traded_at": row.get("traded_at"),
                "price": float(row["price"]),
                "fill_price": fill_px,
                "fee": fee,
                "pnl": pnl,
                "token_won": bool(won),
                "label": params.label(),
                "slip_bps": slip_bps,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _classify(
    *,
    n_events: int,
    pos_fold_frac: float,
    test_ev: float | None,
    bootstrap: dict[str, Any],
    leave5: dict[str, Any],
) -> Robustness:
    if test_ev is None or n_events < 5:
        return "rejected"
    p_pos = bootstrap.get("p_positive") if bootstrap.get("status") == "ok" else None
    leave5_mean = leave5.get("mean_trimmed") if leave5.get("status") == "ok" else None
    if test_ev <= 0 or (p_pos is not None and p_pos < 0.55):
        return "rejected"
    if pos_fold_frac < 0.5:
        return "fragile"
    if leave5_mean is not None and leave5_mean <= 0:
        return "fragile"
    if n_events < 30 or pos_fold_frac < 0.67:
        return "promising_underpowered"
    if p_pos is not None and p_pos >= 0.8 and test_ev > 0 and pos_fold_frac >= 0.67:
        return "robust_candidate"
    return "promising_underpowered"


def coarse_grid() -> list[StrategyParams]:
    """Controlled coarse grid — not a naïve full Cartesian."""
    mid_buckets = ["0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70"]
    out: list[StrategyParams] = []
    # Baselines: bucket-only
    for bucket in mid_buckets:
        out.append(StrategyParams(price_bucket=bucket, side="BUY"))
    # Whale / spike / mom / ttr interactions on mid buckets
    for bucket, whale, spike, mom, ttr in product(
        mid_buckets,
        [None, 2.0, 3.0],
        [None, 1.5, 2.0],
        ["any", "pos"],
        [None, 48.0, 72.0],
    ):
        # Skip null-null-any-null duplicates of baseline
        if whale is None and spike is None and mom == "any" and ttr is None:
            continue
        out.append(
            StrategyParams(
                price_bucket=bucket,
                min_whale_ratio=whale,
                min_volume_spike=spike,
                momentum_1h=mom,  # type: ignore[arg-type]
                max_time_to_resolution_hours=ttr,
                side="BUY",
            )
        )
    # Deduplicate labels
    seen: set[str] = set()
    uniq: list[StrategyParams] = []
    for p in out:
        if p.label() in seen:
            continue
        seen.add(p.label())
        uniq.append(p)
    return uniq


def run_outcome_sweep(
    features: pl.DataFrame,
    *,
    train_end_exclusive: str = "2023-06-01T00:00:00+00:00",
    slip_grid: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 200.0),
    latency_s: float = 1.0,
    out_dir: Path | str | None = None,
    min_train_n: int = 30,
    min_train_events: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Walk-forward purged sweep with frozen final test.

    Execution label: adverse mid-print + slippage sensitivity (no L2).
    Fee label: historical zero-fee for 2022–2023 (exact).
    """
    out_dir = Path(out_dir or Path("data/research"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if features.is_empty() or "token_won" not in features.columns:
        return {"status": "unavailable", "reason": "empty features or missing token_won"}

    # Normalize timezone
    features = features.with_columns(pl.col("traded_at").dt.convert_time_zone("UTC"))
    event_col = (
        "event_id"
        if "event_id" in features.columns and features["event_id"].null_count() < features.height * 0.5
        else "condition_id"
    )
    time_col = "traded_at"

    train_end = datetime.fromisoformat(train_end_exclusive.replace("Z", "+00:00"))
    train_df = features.filter(pl.col(time_col) < pl.lit(train_end))
    test_raw = features.filter(pl.col(time_col) >= pl.lit(train_end))
    train_events = set(train_df[event_col].drop_nulls().to_list())
    test_df = mask_event_leakage(test_raw, train_events, event_col=event_col)

    t_min = train_df[time_col].min()
    if t_min is None:
        return {"status": "unavailable", "reason": "empty train"}

    folds = purged_walk_forward_folds(
        start=t_min,
        end=train_end,
        n_folds=3,
        train_days=60,
        test_days=21,
        embargo_days=2,
    )

    grid = coarse_grid()
    attempts: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    wf_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    ledger_chunks: list[pl.DataFrame] = []
    robust_rows: list[dict[str, Any]] = []

    # Primary selection slip; stress others later
    select_slip = 50.0
    is_matrix: list[list[float]] = []
    oos_matrix: list[list[float]] = []

    for params in grid:
        train_slice = apply_strategy_filter(train_df, params)
        n_train = train_slice.height
        n_ev = int(train_slice[event_col].n_unique()) if n_train else 0
        attempt = {
            "label": params.label(),
            **asdict(params),
            "n_train": n_train,
            "n_train_events": n_ev,
            "status": "ok" if n_train >= min_train_n and n_ev >= min_train_events else "insufficient_sample",
        }
        attempts.append(attempt)
        if attempt["status"] != "ok":
            continue

        # Walk-forward folds
        fold_oos_ev: list[float] = []
        fold_is_sharpe: list[float] = []
        fold_oos_sharpe: list[float] = []
        for fold in folds:
            fold_train = train_df.filter(
                (pl.col(time_col) >= pl.lit(fold.train_start).str.to_datetime(time_zone="UTC"))
                & (pl.col(time_col) < pl.lit(fold.train_end).str.to_datetime(time_zone="UTC"))
            )
            fold_test = train_df.filter(
                (pl.col(time_col) >= pl.lit(fold.test_start).str.to_datetime(time_zone="UTC"))
                & (pl.col(time_col) < pl.lit(fold.test_end).str.to_datetime(time_zone="UTC"))
            )
            tev = set(fold_train[event_col].drop_nulls().to_list())
            fold_test = mask_event_leakage(fold_test, tev, event_col=event_col)
            tr_led = _simulate_candidate_ledger(fold_train, params, slip_bps=select_slip, event_col=event_col)
            te_led = _simulate_candidate_ledger(fold_test, params, slip_bps=select_slip, event_col=event_col)
            is_sh = _sharpe_ratio(tr_led["pnl"]) if tr_led.height >= 2 else 0.0
            oos_sh = _sharpe_ratio(te_led["pnl"]) if te_led.height >= 2 else 0.0
            oos_ev = float(te_led["pnl"].mean()) * 100.0 if te_led.height else 0.0
            fold_is_sharpe.append(is_sh)
            fold_oos_sharpe.append(oos_sh)
            fold_oos_ev.append(oos_ev)
            wf_rows.append(
                {
                    "label": params.label(),
                    "fold_id": fold.fold_id,
                    "n_test": te_led.height,
                    "oos_ev_pct": oos_ev,
                    "oos_sharpe": oos_sh,
                    "is_sharpe": is_sh,
                }
            )

        is_matrix.append(fold_is_sharpe)
        oos_matrix.append(fold_oos_sharpe)
        pos_frac = sum(1 for e in fold_oos_ev if e > 0) / max(len(fold_oos_ev), 1)

        # Only promote to final test if majority of folds positive
        if pos_frac < 0.5:
            results.append(
                {
                    "label": params.label(),
                    **asdict(params),
                    "slip_bps_select": select_slip,
                    "pos_fold_frac": pos_frac,
                    "wf_mean_oos_ev": sum(fold_oos_ev) / len(fold_oos_ev),
                    "promoted_to_final": False,
                    "robustness": "fragile",
                }
            )
            continue

        # Locked final test once
        final_led = _simulate_candidate_ledger(test_df, params, slip_bps=select_slip, event_col=event_col)
        if final_led.height:
            ledger_chunks.append(final_led)
        test_ev = float(final_led["pnl"].mean()) * 100.0 if final_led.height else None
        test_sharpe = _sharpe_ratio(final_led["pnl"]) if final_led.height >= 2 else None
        test_pnl = float(final_led["pnl"].sum()) if final_led.height else 0.0
        equity = final_led["pnl"].cum_sum().to_list() if final_led.height else []
        max_dd = _max_drawdown(equity)
        n_test_events = int(final_led[event_col].n_unique()) if final_led.height else 0
        event_pnls = (
            event_clustered_returns(final_led, event_col=event_col, pnl_col="pnl")["event_pnl"].to_list()
            if final_led.height
            else []
        )
        boot = cluster_bootstrap_ci(event_pnls, n_boot=1000, seed=seed)
        leave5 = leave_five_largest_out(event_pnls)
        robustness = _classify(
            n_events=n_test_events,
            pos_fold_frac=pos_frac,
            test_ev=test_ev,
            bootstrap=boot,
            leave5=leave5,
        )

        # Slippage sensitivity on final ledger prices (recompute)
        slip_sens: dict[str, float | None] = {}
        for slip in slip_grid:
            led = _simulate_candidate_ledger(test_df, params, slip_bps=slip, event_col=event_col)
            slip_sens[f"ev_slip_{int(slip)}"] = (
                float(led["pnl"].mean()) * 100.0 if led.height else None
            )

        edge_train = compute_edge_stats(train_slice, params)
        row = {
            "label": params.label(),
            **asdict(params),
            "slip_bps_select": select_slip,
            "pos_fold_frac": pos_frac,
            "wf_mean_oos_ev": sum(fold_oos_ev) / len(fold_oos_ev),
            "best_fold_ev": max(fold_oos_ev) if fold_oos_ev else None,
            "worst_fold_ev": min(fold_oos_ev) if fold_oos_ev else None,
            "promoted_to_final": True,
            "test_n": final_led.height,
            "test_n_events": n_test_events,
            "test_ev_pct": test_ev,
            "test_total_pnl": test_pnl,
            "test_sharpe": test_sharpe,
            "test_max_dd": max_dd,
            "test_hit_rate": float(final_led["token_won"].mean()) if final_led.height else None,
            "bootstrap": boot,
            "leave5": leave5,
            "robustness": robustness,
            "train_edge": edge_train.to_dict(),
            "fee_regime": "zero_fee_historical",
            "fee_confidence": "exact",
            "fee_model_version": FEE_MODEL_VERSION,
            "execution_label": "adverse_mid_print_slippage_sensitivity",
            **slip_sens,
        }
        results.append(row)
        final_rows.append(row)
        robust_rows.append(
            {
                "label": params.label(),
                "robustness": robustness,
                "test_ev_pct": test_ev,
                "test_n_events": n_test_events,
                "pos_fold_frac": pos_frac,
                "p_positive": boot.get("p_positive") if boot.get("status") == "ok" else None,
                "leave5_mean": leave5.get("mean_trimmed") if leave5.get("status") == "ok" else None,
            }
        )

    # Multiple testing on promoted configs only (real inputs)
    promoted_sharpes = [r["test_sharpe"] for r in final_rows if r.get("test_sharpe") is not None]
    dsr = (
        deflated_sharpe_valid(
            max(promoted_sharpes),
            n_obs=max(int(test_df.height), 2),
            n_trials=max(len(attempts), 1),
        )
        if promoted_sharpes
        else {"status": "unavailable", "reason": "no promoted sharpes"}
    )
    pbo = pbo_from_matrices(is_matrix, oos_matrix) if is_matrix and oos_matrix else {"status": "unavailable"}
    fdr = bh_fdr(None)  # no parametric p-values → unavailable

    attempts_df = pl.DataFrame(attempts)
    results_df = pl.DataFrame(results) if results else pl.DataFrame({"label": []})
    wf_df = pl.DataFrame(wf_rows) if wf_rows else pl.DataFrame({"label": []})
    final_df = pl.DataFrame(final_rows) if final_rows else pl.DataFrame({"label": []})
    robust_df = pl.DataFrame(robust_rows) if robust_rows else pl.DataFrame({"label": []})
    ledger_df = pl.concat(ledger_chunks, how="diagonal_relaxed") if ledger_chunks else pl.DataFrame()

    attempts_df.write_parquet(out_dir / "outcome_sweep_attempts.parquet")
    results_df.write_parquet(out_dir / "outcome_sweep_results.parquet")
    wf_df.write_parquet(out_dir / "outcome_walk_forward.parquet")
    final_df.write_parquet(out_dir / "outcome_final_test.parquet")
    robust_df.write_parquet(out_dir / "outcome_robustness.parquet")
    if not ledger_df.is_empty():
        ledger_df.write_parquet(out_dir / "outcome_trade_ledger.parquet")

    n_events_train = int(train_df[event_col].n_unique())
    n_events_test = int(test_df[event_col].n_unique())
    manifest = {
        "mode": "outcome_strategy_sweep",
        "NOT_microstructure": True,
        "fee_model_version": FEE_MODEL_VERSION,
        "fee_regime": "zero_fee_historical",
        "fee_confidence": "exact",
        "fee_scenario": "primary_historical_zero",
        "execution_label": "adverse_mid_print_slippage_sensitivity",
        "latency_s_recorded": latency_s,
        "latency_note": "Latency does not shift prints without next-print path; slip grid is primary stress",
        "train_end_exclusive": train_end_exclusive,
        "event_col": event_col,
        "n_train_rows": train_df.height,
        "n_test_rows_purged": test_df.height,
        "n_events_train": n_events_train,
        "n_events_test": n_events_test,
        "n_attempts": attempts_df.height,
        "n_sufficient": int(attempts_df.filter(pl.col("status") == "ok").height) if attempts_df.height else 0,
        "n_promoted_final": final_df.height,
        "n_robust_candidates": int(robust_df.filter(pl.col("robustness") == "robust_candidate").height)
        if robust_df.height and "robustness" in robust_df.columns
        else 0,
        "slip_grid": list(slip_grid),
        "select_slip_bps": select_slip,
        "folds": [asdict(f) if hasattr(f, "__dataclass_fields__") else f for f in folds],
        "multiple_testing": {"dsr": dsr, "pbo": pbo, "fdr": fdr},
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    # folds may be dataclasses
    from dataclasses import is_dataclass

    manifest["folds"] = [asdict(f) if is_dataclass(f) else f for f in folds]
    (out_dir / "outcome_backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    top = []
    if final_df.height and "test_ev_pct" in final_df.columns:
        top = final_df.sort("test_ev_pct", descending=True).head(10).to_dicts()

    return {
        "status": "ok",
        "manifest": manifest,
        "top": top,
        "out_dir": str(out_dir),
    }
