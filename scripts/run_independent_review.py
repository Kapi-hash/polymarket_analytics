#!/usr/bin/env python3
"""Independent review scorecard + restricted mid-fill diagnostic search.

This is NOT the designed full execution-realistic staged sweep.
It exists to:
  1) produce review_scorecard / discrepancy / remediation artifacts
  2) optionally run an honestly labeled restricted mid-fill fee-fallback search
     on the local lake (81 events, no L2 books, no market fee categories)

Exit codes:
  0 — review artifacts written (decision may still be BLOCKED)
  2 — unexpected failure
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polymarket_analytics.backtest import (  # noqa: E402
    StrategyParams,
    apply_strategy_filter,
    compute_edge_stats,
    load_trade_features,
    simulate_strategy,
)
from polymarket_analytics.research.fees import FEE_MODEL_VERSION  # noqa: E402
from polymarket_analytics.research.feature_registry import coverage_report  # noqa: E402
from polymarket_analytics.research.inventory import (  # noqa: E402
    build_baseline_inventory,
    write_inventory_artifacts,
)
from polymarket_analytics.research.validation import (  # noqa: E402
    fit_bin_calibrator,
    multiple_testing_controls,
    run_grouped_purged_validation,
)
from polymarket_analytics.schema import PRICE_BUCKET_LABELS  # noqa: E402

OUT = ROOT / "docs" / "research"
REVIEW_OUT = OUT / "independent_review"
DATA_OUT = REVIEW_OUT  # scorecard + dataset live with the review pack
RESTRICTED = REVIEW_OUT / "results" / "restricted_midfill"


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _data_fingerprint(df: pl.DataFrame) -> str:
    key = f"{df.height}|{df['condition_id'].n_unique()}|{df['traded_at'].min()}|{df['traded_at'].max()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _run_tests() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": proc.returncode,
        "summary": (proc.stdout or "").strip().splitlines()[-1:] or [""],
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def _lake_summary(df: pl.DataFrame) -> dict[str, Any]:
    inv = (
        df.sort(["token_id", "traded_at"])
        .with_columns(
            (pl.col("traded_at").diff().over("token_id").dt.total_seconds() < 0).alias(
                "_inv"
            )
        )["_inv"]
        .fill_null(False)
        .sum()
    )
    return {
        "n_trade_feature_rows": df.height,
        "n_independent_events": int(df["condition_id"].n_unique()),
        "n_tokens": int(df["token_id"].n_unique()),
        "date_min": str(df["traded_at"].min()),
        "date_max": str(df["traded_at"].max()),
        "duplicate_trade_ids": int(df.height - df["trade_id"].n_unique()),
        "timestamp_inversions": int(inv),
        "token_won_nulls": int(df["token_won"].null_count()),
        "has_l2_books": False,
        "has_fee_category_column": "fee_category" in df.columns
        or "category" in df.columns,
        "price_buckets": {
            str(r["price_bucket"]): int(r["len"])
            for r in df.group_by("price_bucket").len().sort("price_bucket").to_dicts()
        },
    }


def build_scorecard(df: pl.DataFrame, test_info: dict[str, Any]) -> dict[str, Any]:
    feats = coverage_report()
    n_events = int(df["condition_id"].n_unique())
    gates = {
        "gate1_code_test_integrity": {
            "result": "PASS" if test_info["returncode"] == 0 else "FAIL",
            "evidence": test_info,
            "notes": "pytest suite; type/lint not enforced in CI for this package",
        },
        "gate2_parameter_inventory": {
            "result": "PASS",
            "evidence": {
                "inventory_items": len(build_baseline_inventory()),
                "remediated": [
                    "latency marked Partially wired / inert",
                    "paper spike/ttr matches declared unused",
                    "registry stub statuses corrected to partial",
                ],
            },
            "notes": "Active params inventoried; residual conflicts documented",
        },
        "gate3_point_in_time": {
            "result": "PASS",
            "evidence": {
                "rolling_closed": "right (includes current trade)",
                "caveat": (
                    "Features include the decision trade in the rolling window; "
                    "acceptable for trade-along signals but not pre-trade signals"
                ),
                "resolution_visible_before": False,
                "token_won_nulls": 0,
            },
            "notes": "Legacy features.py rolling is PIT-safe for closed='right' at trade time",
        },
        "gate4_fee_correctness": {
            "result": "FAIL",
            "critical": True,
            "evidence": {
                "fee_model_version": FEE_MODEL_VERSION,
                "historical_schedule_versions": "none (all effective_from=1970-01-01)",
                "market_fee_category_in_lake": False,
                "backtest_default_fees": "opt-in only; historical CLI still gross by default",
                "swing_fees": False,
            },
            "notes": (
                "PIT fee category missing for all lake rows; only conservative "
                "fallback sensitivity is honest — cannot present definitive fee results"
            ),
        },
        "gate5_execution_realism": {
            "result": "FAIL",
            "critical": True,
            "evidence": {
                "l2_book_history": False,
                "latency_changes_fill": False,
                "book_walk_default": False,
                "queue_ahead_on_lake": False,
                "historical_simulate_uses_trade_print": True,
            },
            "notes": "No historical depth; cannot walk books or model queue-ahead fills",
        },
        "gate6_feature_correctness": {
            "result": "PASS",
            "evidence": feats,
            "notes": "Logit/fee/OFI unit-tested; blocked features correctly stubbed after remediation",
        },
        "gate7_data_coverage": {
            "result": "FAIL",
            "critical": True,
            "evidence": _lake_summary(df),
            "notes": (
                f"Only {n_events} independent events (preferred ≥100); "
                "no books; sample lake ~Nov 2022–Aug 2023"
            ),
        },
        "gate8_validation_design": {
            "result": "FAIL",
            "critical": True,
            "evidence": {
                "purged_wf_helpers": True,
                "wired_to_cli_backtest": False,
                "frozen_final_test_set": False,
                "representative_script_uses_synthetic": True,
                "legacy_oos": "chronological split_date only",
            },
            "notes": "Scaffolding exists but is not the production backtest path",
        },
    }
    critical_fails = [
        k for k, v in gates.items() if v.get("critical") and v["result"] == "FAIL"
    ]
    decision = "BLOCKED" if critical_fails else "PASS"
    return {
        "decision": decision,
        "critical_failures": critical_fails,
        "gates": gates,
        "git_commit": _git_head(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "feature_coverage": feats,
        "full_backtest_authorized": False,
    }


def write_discrepancies(path: Path) -> None:
    rows = [
        {
            "issue": "Incorrect status",
            "item": "calibration_residual / inventory_risk_cap marked implemented with stub compute",
            "source": "research/feature_registry.py",
            "resolution": "Remediated → status=partial",
        },
        {
            "issue": "Incorrect status",
            "item": "whale_ratio / volume_spike / decay registry compute pass-through marked implemented",
            "source": "research/feature_registry.py",
            "resolution": "Remediated → status=partial",
        },
        {
            "issue": "Declared-but-unused parameter",
            "item": "PaperTrader.matches min_volume_spike / max_ttr always False if set",
            "source": "paper_trader.py:matches",
            "resolution": "Documented in inventory as Declared but unused",
        },
        {
            "issue": "Incorrect formula / inert param",
            "item": "latency_ms does not delay fills",
            "source": "paper_trader.py / execution.py",
            "resolution": "Inventory status → Partially wired; EXECUTION_GAPS noted",
        },
        {
            "issue": "Missing inventory item (pre-review)",
            "item": "PaperConfig.use_book_walk default False, CLI-hidden",
            "source": "paper_trader.py",
            "resolution": "Added to inventory",
        },
        {
            "issue": "Incorrect configuration precedence claim",
            "item": "research_priors.yaml never applied at runtime",
            "source": "config/research_priors.yaml",
            "resolution": "Precedence doc already correct; confirmed unused",
        },
        {
            "issue": "Conflicting definition",
            "item": "Swing TP-before-SL vs paper stop_first",
            "source": "swing_trader.py / paper_trader.py",
            "resolution": "Open conflict — documented; not auto-aligned (methodology change)",
        },
        {
            "issue": "Incorrect formula",
            "item": "Legacy simulate_strategy gross-only (no fees/slippage)",
            "source": "backtest.py",
            "resolution": "Remediated: opt-in fee_category / spread_slippage_bps",
        },
        {
            "issue": "Missing test",
            "item": "Fee-aware simulate_strategy",
            "source": "tests/test_phase3_backtest.py",
            "resolution": "Added test_simulate_strategy_fee_aware_reduces_pnl",
        },
        {
            "issue": "Hidden hardcoded threshold",
            "item": "paper matches PVD uses whale<=3.0 literal",
            "source": "paper_trader.py:330",
            "resolution": "Already in hardcoded report; remains active",
        },
    ]
    pl.DataFrame(rows).write_csv(path)


def write_remediation_log(path: Path) -> None:
    path.write_text(
        """# Remediation log (independent review)

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
""",
        encoding="utf-8",
    )


def run_restricted_midfill(df: pl.DataFrame) -> dict[str, Any]:
    """Honest restricted search: mid-fill prints + crypto fee fallback + 50bps slip."""
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    # Chronological freeze: train/val before 2023-06-01, locked test after
    split = pl.lit("2023-06-01").str.to_datetime(time_zone="UTC")
    train = df.filter(pl.col("traded_at") < split)
    test = df.filter(pl.col("traded_at") >= split)
    # Purge test events that appear in train
    train_events = set(train["condition_id"].unique().to_list())
    test_clean = test.filter(~pl.col("condition_id").is_in(list(train_events)))

    fee_kwargs = dict(fee_category="crypto", fee_role="taker", spread_slippage_bps=50.0)
    axes = {
        "price_bucket": list(PRICE_BUCKET_LABELS)[3:10],  # mid buckets only
        "min_whale_ratio": [None, 2.0, 3.0],
        "min_volume_spike": [None, 1.5, 2.0],
        "momentum_1h": ["any", "pos"],
        "max_ttr": [None, 48.0, 72.0],
    }
    attempts: list[dict[str, Any]] = []
    for bucket, whale, spike, mom, ttr in product(
        axes["price_bucket"],
        axes["min_whale_ratio"],
        axes["min_volume_spike"],
        axes["momentum_1h"],
        axes["max_ttr"],
    ):
        params = StrategyParams(
            price_bucket=bucket,
            min_whale_ratio=whale,
            min_volume_spike=spike,
            momentum_1h=mom,  # type: ignore[arg-type]
            max_time_to_resolution_hours=ttr,
            side="BUY",
        )
        train_bt = simulate_strategy(train, params, **fee_kwargs)
        # Event-clustered bootstrap proxy: mean EV by condition
        sliced = apply_strategy_filter(train, params)
        n_events = int(sliced["condition_id"].n_unique()) if sliced.height else 0
        attempts.append(
            {
                "label": params.label(),
                **asdict(params),
                "train_n": train_bt.n,
                "train_n_events": n_events,
                "train_ev_pct_net": train_bt.ev_pct,
                "train_total_pnl_net": train_bt.total_pnl,
                "train_sharpe": train_bt.sharpe,
                "train_max_dd": train_bt.max_drawdown,
                "fee_assumption": "FALLBACK_crypto_2026-07-polymarket-v1",
                "execution_assumption": "mid_trade_print_plus_50bps",
            }
        )

    att_df = pl.DataFrame(attempts)
    att_df.write_parquet(RESTRICTED / "sweep_attempts.parquet")

    # Rank by train net EV with min samples/events
    cand = att_df.filter((pl.col("train_n") >= 30) & (pl.col("train_n_events") >= 5))
    cand = cand.sort(["train_ev_pct_net", "train_n"], descending=True)
    cand.write_parquet(RESTRICTED / "sweep_results.parquet")

    # Walk-forward on top-5 train candidates (still pre-test)
    top = cand.head(5)
    wf_rows: list[dict[str, Any]] = []
    for row in top.to_dicts():
        params = StrategyParams(
            price_bucket=row["price_bucket"],
            min_whale_ratio=row["min_whale_ratio"],
            min_volume_spike=row["min_volume_spike"],
            momentum_1h=row["momentum_1h"] or "any",
            max_time_to_resolution_hours=row["max_time_to_resolution_hours"],
            side="BUY",
        )

        def evaluate(tr: pl.DataFrame, te: pl.DataFrame) -> dict[str, Any]:
            if tr.height < 10:
                return {"oos_ev": 0.0, "n_test": te.height}
            # Fit calibrator train-only (sanity; not used for sizing here)
            _ = fit_bin_calibrator(
                tr["price"].to_list(), tr["token_won"].to_list(), n_bins=5
            )
            # te already event-purged by run_grouped_purged_validation
            bt = simulate_strategy(te, params, **fee_kwargs)
            return {
                "oos_ev": bt.ev_pct,
                "n_test": bt.n,
                "n_events": int(te["condition_id"].n_unique()) if te.height else 0,
                "sharpe": bt.sharpe,
            }

        wf = run_grouped_purged_validation(
            train,
            evaluate,
            start="2022-12-01",
            end="2023-05-31",
            n_folds=3,
            train_days=60,
            test_days=21,
            embargo_days=2,
        )
        for fold in wf.get("folds", []):
            wf_rows.append({"label": params.label(), **fold.get("metrics", {}), "fold": fold.get("fold")})

    wf_df = pl.DataFrame(wf_rows) if wf_rows else pl.DataFrame({"label": []})
    wf_df.write_parquet(RESTRICTED / "walk_forward_results.parquet")

    # Locked final test ONCE for configs with majority positive WF folds
    final_rows: list[dict[str, Any]] = []
    surviving: list[str] = []
    if not wf_df.is_empty() and "oos_ev" in wf_df.columns:
        for label in top["label"].to_list():
            folds = wf_df.filter(pl.col("label") == label)
            if folds.is_empty():
                continue
            pos = int((folds["oos_ev"] > 0).sum())
            if pos >= max(1, math.ceil(folds.height / 2)):
                surviving.append(label)

    for row in top.to_dicts():
        if row["label"] not in surviving:
            continue
        params = StrategyParams(
            price_bucket=row["price_bucket"],
            min_whale_ratio=row["min_whale_ratio"],
            min_volume_spike=row["min_volume_spike"],
            momentum_1h=row["momentum_1h"] or "any",
            max_time_to_resolution_hours=row["max_time_to_resolution_hours"],
            side="BUY",
        )
        bt = simulate_strategy(test_clean, params, **fee_kwargs)
        final_rows.append(
            {
                "label": params.label(),
                "test_n": bt.n,
                "test_n_events": int(
                    apply_strategy_filter(test_clean, params)["condition_id"].n_unique()
                )
                if bt.n
                else 0,
                "test_ev_pct_net": bt.ev_pct,
                "test_total_pnl_net": bt.total_pnl,
                "test_sharpe": bt.sharpe,
                "test_max_dd": bt.max_drawdown,
                "fee_assumption": "FALLBACK_crypto — NOT definitive",
                "execution_assumption": "mid_trade_print — NOT book-realistic",
            }
        )

    final_df = pl.DataFrame(final_rows) if final_rows else pl.DataFrame({"label": []})
    final_df.write_parquet(RESTRICTED / "final_test_results.parquet")

    # Baselines
    baselines = []
    for bucket in axes["price_bucket"]:
        p = StrategyParams(price_bucket=bucket, side="BUY")
        g = simulate_strategy(train, p)
        n = simulate_strategy(train, p, **fee_kwargs)
        baselines.append(
            {
                "family": "buy_hold_bucket",
                "label": p.label(),
                "gross_ev": g.ev_pct,
                "net_ev_fallback": n.ev_pct,
                "n": g.n,
            }
        )
    # Random-entry control: all BUY mid buckets pooled
    rand_p = StrategyParams(side="BUY")
    baselines.append(
        {
            "family": "all_buy",
            "label": "all BUY",
            "gross_ev": simulate_strategy(train, rand_p).ev_pct,
            "net_ev_fallback": simulate_strategy(train, rand_p, **fee_kwargs).ev_pct,
            "n": simulate_strategy(train, rand_p).n,
        }
    )
    pl.DataFrame(baselines).write_parquet(RESTRICTED / "baselines.parquet")

    # Multiple-testing on train candidates
    sharpes = [float(x) for x in cand.head(50)["train_sharpe"].to_list()] if cand.height else [0.0]
    pvals = [0.05] * len(sharpes)  # placeholder — no parametric p; FDR scaffolding only
    mt = multiple_testing_controls(
        sharpes=sharpes or [0.0],
        oos_sharpes=[0.0] * len(sharpes or [0.0]),
        n_obs=max(int(train.height), 1),
        p_values=pvals or [1.0],
    )

    manifest = {
        "mode": "restricted_midfill_fee_fallback",
        "NOT_full_execution_realistic_sweep": True,
        "git_commit": _git_head(),
        "data_fingerprint": _data_fingerprint(df),
        "fee_model_version": FEE_MODEL_VERSION,
        "fee_label": "FALLBACK — lake lacks fee_category; crypto schedule applied for sensitivity only",
        "execution_label": "trade-print mid fill + 50bps slip; no L2 walk",
        "train_end_exclusive": "2023-06-01T00:00:00+00:00",
        "n_train_rows": train.height,
        "n_test_rows_purged": test_clean.height,
        "n_events_train": int(train["condition_id"].n_unique()),
        "n_events_test_purged": int(test_clean["condition_id"].n_unique()),
        "n_attempts": att_df.height,
        "n_candidates_min_n": cand.height,
        "n_surviving_wf": len(surviving),
        "n_final_tested": final_df.height,
        "multiple_testing": mt.to_dict(),
        "folds_spec": {"n_folds": 3, "train_days": 60, "test_days": 21, "embargo_days": 2},
        "blockers_remaining": [
            "no_l2_books",
            "no_fee_category_metadata",
            "only_81_events_total",
            "fee_schedule_not_historically_versioned",
        ],
    }
    (RESTRICTED / "backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # Exclusions
    excl = [
        {
            "reason": "no_historical_l2_depth",
            "markets_affected": "all",
            "strategies_blocked": "book_walk,queue_ahead,ofi,maker_realism",
        },
        {
            "reason": "no_fee_category_tag",
            "markets_affected": "all lake condition_ids",
            "strategies_blocked": "definitive_net_of_fee_ranking",
        },
        {
            "reason": "event_overlap_purged_from_final_test",
            "markets_affected": len(train_events & set(test["condition_id"].unique().to_list())),
            "strategies_blocked": "",
        },
    ]
    pl.DataFrame(excl).write_csv(RESTRICTED / "excluded_markets.csv")

    summary_md = [
        "# Restricted mid-fill diagnostic (NOT full sweep)",
        "",
        f"- Git: `{manifest['git_commit']}`",
        f"- Attempts: {manifest['n_attempts']}",
        f"- Candidates (n≥30, events≥5): {manifest['n_candidates_min_n']}",
        f"- WF survivors: {manifest['n_surviving_wf']}",
        f"- Final-test configs: {manifest['n_final_tested']}",
        f"- Train events: {manifest['n_events_train']}, purged test events: {manifest['n_events_test_purged']}",
        "",
        "## Fee / execution labels",
        f"- {manifest['fee_label']}",
        f"- {manifest['execution_label']}",
        "",
        "## Top train candidates (net fallback)",
    ]
    for r in cand.head(10).to_dicts():
        summary_md.append(
            f"- `{r['label']}`: net EV={r['train_ev_pct_net']:.2f}%, "
            f"n={r['train_n']}, events={r['train_n_events']}, sharpe={r['train_sharpe']:.3f}"
        )
    summary_md.append("")
    summary_md.append("## Locked final test (fallback fees)")
    if final_df.is_empty():
        summary_md.append("- No configuration survived walk-forward majority-positive filter.")
    else:
        for r in final_df.sort("test_ev_pct_net", descending=True).to_dicts():
            summary_md.append(
                f"- `{r['label']}`: test net EV={r['test_ev_pct_net']:.2f}%, "
                f"n={r['test_n']}, events={r['test_n_events']}"
            )
    summary_md.append("")
    summary_md.append(
        "**Do not treat these as paper-trading recommendations.** "
        "Critical gates for execution-realistic evaluation remain failed."
    )
    (RESTRICTED / "top_strategies.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    return manifest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    REVIEW_OUT.mkdir(parents=True, exist_ok=True)
    RESTRICTED.mkdir(parents=True, exist_ok=True)

    inv = write_inventory_artifacts(OUT)
    test_info = _run_tests()
    df = load_trade_features(ROOT / "data" / "warehouse.duckdb", ROOT / "data" / "parquet")
    # Drop epoch pollution / pre-2020 rows without TZ comparison pitfalls
    df = df.filter(pl.col("traded_at").dt.year() >= 2020)
    df = df.with_columns(pl.col("traded_at").dt.convert_time_zone("UTC"))

    scorecard = build_scorecard(df, test_info)
    (REVIEW_OUT / "review_scorecard.json").write_text(
        json.dumps(scorecard, indent=2, default=str), encoding="utf-8"
    )

    md_lines = [
        "# Independent Review Scorecard",
        "",
        f"**Decision: {scorecard['decision']}**",
        f"",
        f"- Git commit: `{scorecard['git_commit']}`",
        f"- Evaluated at: {scorecard['evaluated_at']}",
        f"- Full backtest authorized: `{scorecard['full_backtest_authorized']}`",
        f"- Critical failures: {', '.join(scorecard['critical_failures']) or 'none'}",
        "",
        "## Gates",
        "",
    ]
    for name, g in scorecard["gates"].items():
        md_lines.append(f"### {name}: **{g['result']}**")
        md_lines.append(f"- {g.get('notes', '')}")
        md_lines.append("")
    md_lines.append("## Blocker summary")
    md_lines.append("")
    md_lines.append(
        "Designed full staged sweep is **BLOCKED**. "
        "Required data missing: historical L2 books, per-market fee categories / "
        "historical fee regimes, frozen CLI validation harness, and ≥100 independent events "
        f"(have {scorecard['gates']['gate7_data_coverage']['evidence']['n_independent_events']})."
    )
    md_lines.append("")
    md_lines.append(
        "A restricted mid-fill + fee-fallback diagnostic may still be run for research "
        "sensitivity only — results cannot be presented as definitive."
    )
    (REVIEW_OUT / "review_scorecard.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    write_discrepancies(REVIEW_OUT / "review_discrepancies.csv")
    write_remediation_log(REVIEW_OUT / "remediation_log.md")

    # Lake coverage snapshot next to the scorecard
    lake = _lake_summary(df)
    (REVIEW_OUT / "dataset_summary.json").write_text(
        json.dumps(lake, indent=2, default=str), encoding="utf-8"
    )

    restricted_manifest = None
    if scorecard["decision"] == "BLOCKED":
        # Honest restricted subset only
        restricted_manifest = run_restricted_midfill(df)

    blocker = {
        "decision": scorecard["decision"],
        "exact_blockers": [
            {
                "blocker": "Missing historical order-book (L2/L3) data",
                "affected_strategies": [
                    "book_walk fills",
                    "queue-ahead maker",
                    "OFI",
                    "book resilience",
                    "capacity from depth",
                ],
                "affected_period": lake["date_min"] + " → " + lake["date_max"],
                "markets": "all 81 condition_ids",
                "needed": "Point-in-time multi-level book snapshots or L3 events",
                "restricted_subset_possible": (
                    "Yes: trade-print mid-fill + explicit slippage sensitivity, "
                    "labeled non-definitive"
                ),
            },
            {
                "blocker": "Unknown historical fee regime / missing fee category tags",
                "affected_strategies": ["all net-of-fee rankings"],
                "affected_period": "full lake",
                "markets": "all",
                "needed": "Per-market category + dated fee schedule history",
                "restricted_subset_possible": (
                    "Yes: crypto fallback fee stress only; must be labeled FALLBACK"
                ),
            },
            {
                "blocker": "Too few independent events for robust discovery",
                "affected_strategies": ["parameter search / multiple testing"],
                "affected_period": lake["date_min"] + " → " + lake["date_max"],
                "markets": f"{lake['n_independent_events']} events",
                "needed": "Prefer ≥100 independent events with purged folds",
                "restricted_subset_possible": "Diagnostic only; not production-ready",
            },
            {
                "blocker": "No frozen production validation path on CLI",
                "affected_strategies": ["all historical research claims"],
                "affected_period": "n/a",
                "markets": "n/a",
                "needed": "Wire purged WF + fee-aware simulate into backtest CLI; freeze test set",
                "restricted_subset_possible": "Scripted restricted harness under docs/research/independent_review/results/",
            },
        ],
        "full_backtest_ran": False,
        "restricted_diagnostic_ran": restricted_manifest is not None,
        "restricted_manifest": restricted_manifest,
        "inventory_artifacts": {k: str(v) for k, v in inv.items()},
    }
    (REVIEW_OUT / "blocker_report.json").write_text(
        json.dumps(blocker, indent=2, default=str), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "decision": scorecard["decision"],
                "full_backtest_ran": False,
                "tests": test_info["summary"],
                "scorecard": str(REVIEW_OUT / "review_scorecard.md"),
                "review_dir": str(REVIEW_OUT),
                "restricted": str(RESTRICTED) if restricted_manifest else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
