"""Evidence gate for deciding whether outcome research is actionable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def evaluate_outcome_gate(
    features_or_trades: Path | pl.DataFrame, *, min_events: int = 100
) -> dict[str, Any]:
    """Evaluate outcome-label, fee, duplicate, and temporal coverage evidence."""
    frame = pl.read_parquet(features_or_trades) if isinstance(features_or_trades, Path) else features_or_trades
    event_col = "event_id" if "event_id" in frame.columns and frame["event_id"].null_count() < frame.height else "condition_id"
    n_events = int(frame[event_col].drop_nulls().n_unique()) if event_col in frame.columns else 0
    has_outcomes = "token_won" in frame.columns and frame["token_won"].null_count() < frame.height
    fee_evidenced = 0
    if "fee_confidence" in frame.columns:
        fee_evidenced = int(frame.filter(pl.col("fee_confidence").is_in(["exact", "strongly_evidenced"])).height)
    fee_majority = frame.height > 0 and fee_evidenced / frame.height >= 0.5
    duplicate_cleaned = (
        ("fill_id" in frame.columns or "trade_id" in frame.columns)
        and (frame["fill_id"].n_unique() == frame.height if "fill_id" in frame.columns else frame["trade_id"].n_unique() == frame.height)
    )
    years: list[int] = []
    if "traded_at" in frame.columns:
        years = sorted(set(frame["traded_at"].drop_nulls().dt.year().to_list()))
    multiple_years = len(years) >= 2
    evidence = {
        "n_rows": frame.height,
        "event_column": event_col,
        "n_events": n_events,
        "token_won_present": has_outcomes,
        "fee_evidenced_rows": fee_evidenced,
        "fee_evidenced_majority": fee_majority,
        "duplicates_cleaned": duplicate_cleaned,
        "years": years,
        "multiple_years": multiple_years,
    }
    if n_events < 50 or not has_outcomes:
        decision = "BLOCKED"
        reasons = ["fewer than 50 events" if n_events < 50 else "", "token_won missing" if not has_outcomes else ""]
    elif n_events >= min_events and fee_majority and duplicate_cleaned and multiple_years:
        decision, reasons = "PASS-DEFINITIVE", []
    else:
        decision = "PASS-BOUNDED"
        reasons = []
        if not fee_majority:
            reasons.append("fees are mixed or insufficiently evidenced")
        if not duplicate_cleaned:
            reasons.append("duplicate cleaning is not evidenced")
        if not multiple_years:
            reasons.append("date span does not cover multiple years")
        reasons.append("execution evidence is trade-print/mid only unless L2 is joined")
    return {"decision": decision, "evidence": evidence, "reasons": [r for r in reasons if r]}
