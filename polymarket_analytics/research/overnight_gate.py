"""Evidence gate for deciding whether outcome research is actionable."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


def evaluate_outcome_gate(
    features_or_trades: Path | pl.DataFrame,
    *,
    min_events: int = 100,
    min_train_events: int = 50,
    min_test_events: int = 20,
    train_end_exclusive: str | None = "2025-01-01T00:00:00+00:00",
    baseline_feature_rows: int | None = None,
    require_expansion: bool = True,
    used_baseline_fallback: bool = False,
) -> dict[str, Any]:
    """Evaluate expanded-feature evidence. Never falls back to condition_id as event_id."""
    frame = (
        pl.read_parquet(features_or_trades)
        if isinstance(features_or_trades, Path)
        else features_or_trades
    )
    reasons: list[str] = []
    evidence: dict[str, Any] = {
        "n_rows": frame.height,
        "used_baseline_fallback": used_baseline_fallback,
    }

    if used_baseline_fallback:
        reasons.append("requested expanded features were replaced by a baseline fallback")
    if frame.is_empty():
        return {"decision": "BLOCKED", "evidence": evidence, "reasons": ["expanded feature file empty"]}

    if "event_id" not in frame.columns:
        return {
            "decision": "BLOCKED",
            "evidence": evidence,
            "reasons": ["event_id column missing; condition_id fallback is forbidden"],
        }

    event_null_frac = frame["event_id"].null_count() / frame.height
    n_events = int(frame["event_id"].drop_nulls().n_unique())
    evidence.update(
        {
            "event_column": "event_id",
            "n_events": n_events,
            "event_id_populated_frac": 1.0 - event_null_frac,
        }
    )
    if event_null_frac > 0.05:
        reasons.append(f"event_id populated fraction {1.0 - event_null_frac:.3f} < 0.95")
    if n_events < min_events:
        reasons.append(f"independent events {n_events} < min_events {min_events}")

    if "token_won" not in frame.columns:
        reasons.append("token_won missing")
        label_cov = 0.0
    else:
        label_cov = 1.0 - (frame["token_won"].null_count() / frame.height)
    evidence["label_coverage"] = label_cov
    if label_cov < 0.95:
        reasons.append(f"outcome label coverage {label_cov:.3f} < 0.95")

    fee_evidenced = 0
    if "fee_confidence" in frame.columns:
        fee_evidenced = int(
            frame.filter(pl.col("fee_confidence").is_in(["exact", "strongly_evidenced"])).height
        )
    fee_frac = fee_evidenced / frame.height if frame.height else 0.0
    evidence["fee_evidenced_rows"] = fee_evidenced
    evidence["fee_evidenced_frac"] = fee_frac
    if fee_frac < 0.5:
        reasons.append("point-in-time fee evidence insufficient (<50% exact/strongly_evidenced)")

    id_col = "trade_id" if "trade_id" in frame.columns else ("fill_id" if "fill_id" in frame.columns else None)
    if id_col is None:
        reasons.append("trade/fill identifier missing")
    else:
        unique = frame[id_col].n_unique() == frame.height
        evidence["id_unique"] = unique
        if not unique:
            reasons.append(f"{id_col} is not unique")

    years: list[int] = []
    if "traded_at" in frame.columns:
        years = sorted({y for y in frame["traded_at"].drop_nulls().dt.year().to_list() if y is not None})
    evidence["years"] = years
    required_years = {2022, 2023, 2024, 2025}  # 2026 optional if early-year sparse
    missing_years = sorted(y for y in (2022, 2023, 2024) if y not in years)
    # Require at least one post-2023 year in expansion set
    if not any(y >= 2024 for y in years):
        reasons.append("no trades in 2024+ after expansion")
    evidence["missing_core_years"] = missing_years

    if baseline_feature_rows is not None and require_expansion and frame.height <= baseline_feature_rows:
        reasons.append(
            f"expanded rows {frame.height} did not exceed baseline rows {baseline_feature_rows}"
        )

    n_train_events = 0
    n_test_events = 0
    n_test_rows = 0
    if train_end_exclusive and "traded_at" in frame.columns:
        train_end = datetime.fromisoformat(train_end_exclusive.replace("Z", "+00:00"))
        if train_end.tzinfo is None:
            train_end = train_end.replace(tzinfo=timezone.utc)
        train_df = frame.filter(pl.col("traded_at") < pl.lit(train_end))
        test_df = frame.filter(pl.col("traded_at") >= pl.lit(train_end))
        train_events = set(train_df["event_id"].drop_nulls().to_list())
        test_events = set(test_df["event_id"].drop_nulls().to_list()) - train_events
        n_train_events = len(train_events)
        n_test_events = len(test_events)
        n_test_rows = test_df.filter(pl.col("event_id").is_in(list(test_events))).height if test_events else 0
        evidence.update(
            {
                "train_end_exclusive": train_end_exclusive,
                "n_train_events": n_train_events,
                "n_test_events_purged": n_test_events,
                "n_test_rows_purged": n_test_rows,
            }
        )
        if n_train_events < min_train_events:
            reasons.append(f"train events {n_train_events} < {min_train_events}")
        if n_test_events < min_test_events:
            reasons.append(f"purged locked-test events {n_test_events} < {min_test_events}")
        if n_test_rows == 0:
            reasons.append("locked-test rows equal zero")

    fingerprints_ok = "trade_id" in frame.columns and frame.height > 0
    evidence["lineage_ok"] = fingerprints_ok
    if not fingerprints_ok:
        reasons.append("data lineage/fingerprints incomplete")

    if reasons:
        decision = "BLOCKED"
    else:
        # Trade-print execution is not L2-realistic → PASS-BOUNDED is the normal ceiling.
        decision = "PASS-BOUNDED"
        if (
            n_events >= max(min_events, 200)
            and fee_frac >= 0.95
            and label_cov >= 0.99
            and event_null_frac <= 0.01
            and n_test_events >= max(min_test_events, 50)
        ):
            # Still not definitive without L2-joined execution evidence.
            decision = "PASS-BOUNDED"
            evidence["definitive_blocked_reason"] = (
                "historical trade-print execution is not L2-realistic; PASS-DEFINITIVE withheld"
            )

    return {"decision": decision, "evidence": evidence, "reasons": reasons}
