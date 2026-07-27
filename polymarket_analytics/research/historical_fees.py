"""Point-in-time historical fee reconstruction for pre-2026 lake periods."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Final, Literal

from polymarket_analytics.research.fees import (
    CATEGORY_TAKER_FEE_RATES,
    DEFAULT_MAKER_MULTIPLIER,
    FeeModel,
    _as_date_str,
    _parabolic_fee,
    round_fee,
)

FEE_MODEL_VERSION: Final[str] = "clob-v1-historical-2026-07-27"

# Inclusive end of zero-fee era (Polymarket fee-free until Jan 6 2026).
_ZERO_FEE_END: Final[str] = "2026-01-05"
_CURRENT_FEE_START: Final[str] = "2026-01-06"

_ZERO_FEE_EVIDENCE: Final[str] = (
    "Multi-year zero-fee policy ended 2026-01-06; lake period 2022-11 to 2023-08 "
    "predates all CLOB V2 category fees."
)

Role = Literal["maker", "taker"]
Confidence = Literal["exact", "strongly_evidenced", "fallback"]


def lookup_fee_regime(
    as_of: datetime | str | None,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Return dated fee regime metadata for ``as_of``.

    1970-01-01 through 2026-01-05 (inclusive): maker=0, taker=0, confidence=exact.
    From 2026-01-06 onward: current CATEGORY_TAKER_FEE_RATES schedule.
    """
    as_of_date = _as_date_str(as_of)
    cat = (category or "crypto").strip().lower()

    if as_of_date <= _ZERO_FEE_END:
        return {
            "fee_regime": "zero_fee_historical",
            "fee_confidence": "exact",
            "maker_bps": 0.0,
            "taker_bps": 0.0,
            "taker_rate": 0.0,
            "maker_multiplier": 0.0,
            "category": cat,
            "formula": "fee = 0 (historical zero-fee era)",
            "evidence": _ZERO_FEE_EVIDENCE,
            "fee_model_version": FEE_MODEL_VERSION,
            "as_of": as_of_date,
            "effective_from": "1970-01-01",
            "effective_to": _ZERO_FEE_END,
        }

    rate = float(CATEGORY_TAKER_FEE_RATES.get(cat, CATEGORY_TAKER_FEE_RATES["crypto"]))
    return {
        "fee_regime": "clob_v2_category",
        "fee_confidence": "strongly_evidenced",
        "maker_bps": 0.0,
        "taker_bps": rate * 10_000.0 * 0.25,  # peak at p=0.5 → rate * 0.25 notional
        "taker_rate": rate,
        "maker_multiplier": DEFAULT_MAKER_MULTIPLIER,
        "category": cat,
        "formula": "fee = C × feeRate × p × (1 − p)",
        "evidence": (
            "Current Polymarket category schedule (post 2026-01-06); "
            "not applicable to 2022-2023 lake backtests."
        ),
        "fee_model_version": FEE_MODEL_VERSION,
        "as_of": as_of_date,
        "effective_from": _CURRENT_FEE_START,
        "effective_to": None,
    }


def compute_historical_fill_fee(
    shares: float,
    price: float,
    as_of: datetime | str | None,
    role: Role = "taker",
    category: str | None = None,
) -> dict[str, Any]:
    """
    Per-fill fee using historical regime lookup.

    For 2022-2023 lake dates: always fee=0 with confidence exact.
    """
    regime = lookup_fee_regime(as_of, category=category)
    fm = FeeModel()
    decimals = fm.rounding_decimals

    if regime["fee_regime"] == "zero_fee_historical" or float(regime["taker_rate"]) <= 0.0:
        fee = 0.0
    elif role == "maker":
        raw = _parabolic_fee(shares, price, float(regime["taker_rate"])) * float(
            regime["maker_multiplier"]
        )
        fee = round_fee(raw, decimals)
    else:
        raw = _parabolic_fee(shares, price, float(regime["taker_rate"]))
        fee = round_fee(raw, decimals)

    return {
        "fee": fee,
        "role": role,
        "category": regime["category"],
        "fee_rate": float(regime["taker_rate"]),
        "maker_multiplier": float(regime["maker_multiplier"]),
        "fee_free": fee == 0.0,
        "fee_regime": regime["fee_regime"],
        "fee_confidence": regime["fee_confidence"],
        "formula": regime["formula"],
        "evidence": regime["evidence"],
        "fee_model_version": FEE_MODEL_VERSION,
        "as_of": regime["as_of"],
    }
