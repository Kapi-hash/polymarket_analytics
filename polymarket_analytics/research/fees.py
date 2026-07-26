"""Point-in-time, market-specific fee model (maker/taker, rounding, versioning)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final, Literal, Mapping

Role = Literal["maker", "taker"]

# Bumped when fee schedule semantics change; recorded on run snapshots.
FEE_MODEL_VERSION: Final[str] = "2026-07-polymarket-v1"

# Category taker feeRate inputs (docs.polymarket.com/trading/fees).
# fee = C × feeRate × p × (1 − p); peaks at p = 0.50.
# Historically valid as of FEE_MODEL_VERSION — do not apply retroactively
# without matching metadata when rates differed.
CATEGORY_TAKER_FEE_RATES: dict[str, float] = {
    "crypto": 0.07,
    "sports": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "general": 0.05,
    "mentions": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
}

# Maker rebate / fee multipliers relative to taker curve (platform-dependent).
# Polymarket makers are typically fee-free on many markets; model as 0.0 default.
DEFAULT_MAKER_MULTIPLIER: Final[float] = 0.0

FEE_FREE_CATEGORIES: frozenset[str] = frozenset({"geopolitics"})


@dataclass(frozen=True)
class FeeScheduleEntry:
    """Point-in-time fee schedule for one category."""

    category: str
    taker_fee_rate: float
    maker_multiplier: float = DEFAULT_MAKER_MULTIPLIER
    effective_from: str = "1970-01-01"  # ISO date; inclusive
    effective_to: str | None = None  # exclusive; None = open-ended
    fee_free: bool = False
    notes: str = ""


def _default_schedule() -> tuple[FeeScheduleEntry, ...]:
    rows: list[FeeScheduleEntry] = []
    for cat, rate in CATEGORY_TAKER_FEE_RATES.items():
        rows.append(
            FeeScheduleEntry(
                category=cat,
                taker_fee_rate=rate,
                maker_multiplier=DEFAULT_MAKER_MULTIPLIER,
                fee_free=(cat in FEE_FREE_CATEGORIES or rate <= 0.0),
                notes=f"Bundled with {FEE_MODEL_VERSION}",
            )
        )
    return tuple(rows)


@dataclass
class FeeModel:
    """Versioned fee model with PIT category lookup."""

    version: str = FEE_MODEL_VERSION
    schedule: tuple[FeeScheduleEntry, ...] = field(default_factory=_default_schedule)
    rounding_decimals: int = 6  # USDC micro-precision style

    def lookup(
        self,
        category: str | None,
        *,
        as_of: datetime | str | None = None,
    ) -> FeeScheduleEntry:
        key = (category or "crypto").strip().lower()
        as_of_date = _as_date_str(as_of)
        candidates = [e for e in self.schedule if e.category == key]
        if not candidates:
            # Fallback: crypto schedule (conservative for unknown)
            candidates = [e for e in self.schedule if e.category == "crypto"]
        valid = [
            e
            for e in candidates
            if e.effective_from <= as_of_date
            and (e.effective_to is None or as_of_date < e.effective_to)
        ]
        if not valid:
            # If history lacks rates for as_of, use earliest entry and flag
            valid = sorted(candidates, key=lambda e: e.effective_from)[:1]
        return valid[-1] if valid else FeeScheduleEntry(
            category=key,
            taker_fee_rate=CATEGORY_TAKER_FEE_RATES.get("crypto", 0.07),
            notes="fallback-missing-history",
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "fee_model_version": self.version,
            "rounding_decimals": self.rounding_decimals,
            "n_schedule_entries": len(self.schedule),
        }


def _as_date_str(as_of: datetime | str | None) -> str:
    if as_of is None:
        return datetime.now(timezone.utc).date().isoformat()
    if isinstance(as_of, datetime):
        return as_of.astimezone(timezone.utc).date().isoformat()
    text = str(as_of).strip()
    return text[:10]


def round_fee(amount: float, decimals: int = 6) -> float:
    """Banker's-avoiding half-up rounding to USDC micro units."""
    q = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(amount)).quantize(q, rounding=ROUND_HALF_UP))


def _parabolic_fee(shares: float, price: float, fee_rate: float) -> float:
    c = max(float(shares), 0.0)
    p = min(max(float(price), 0.0), 1.0)
    r = max(float(fee_rate), 0.0)
    return c * r * p * (1.0 - p)


def taker_fee(
    shares: float,
    price: float,
    *,
    fee_rate: float,
    rounding_decimals: int = 6,
) -> float:
    """Dynamic taker fee: C × feeRate × p × (1 − p), rounded per fill."""
    return round_fee(_parabolic_fee(shares, price, fee_rate), rounding_decimals)


def maker_fee(
    shares: float,
    price: float,
    *,
    fee_rate: float,
    maker_multiplier: float = DEFAULT_MAKER_MULTIPLIER,
    rounding_decimals: int = 6,
) -> float:
    """Maker fee = multiplier × taker curve (default 0 = fee-free maker)."""
    raw = _parabolic_fee(shares, price, fee_rate) * float(maker_multiplier)
    return round_fee(raw, rounding_decimals)


def compute_fill_fee(
    shares: float,
    price: float,
    *,
    role: Role = "taker",
    category: str | None = "crypto",
    fee_rate: float | None = None,
    as_of: datetime | str | None = None,
    model: FeeModel | None = None,
) -> dict[str, Any]:
    """
    Per-fill fee with role, category PIT schedule, and model version.

    Returns dict suitable for journaling / run metadata.
    """
    fm = model or FeeModel()
    entry = fm.lookup(category, as_of=as_of)
    rate = float(fee_rate) if fee_rate is not None else float(entry.taker_fee_rate)
    if entry.fee_free or rate <= 0.0:
        fee = 0.0
    elif role == "maker":
        fee = maker_fee(
            shares,
            price,
            fee_rate=rate,
            maker_multiplier=entry.maker_multiplier,
            rounding_decimals=fm.rounding_decimals,
        )
    else:
        fee = taker_fee(shares, price, fee_rate=rate, rounding_decimals=fm.rounding_decimals)

    return {
        "fee": fee,
        "role": role,
        "category": entry.category,
        "fee_rate": rate,
        "maker_multiplier": entry.maker_multiplier,
        "fee_free": entry.fee_free or rate <= 0.0,
        "fee_model_version": fm.version,
        "as_of": _as_date_str(as_of),
        "schedule_notes": entry.notes,
    }


def relative_fee_rate(price: float, *, fee_rate: float) -> float:
    """Fee as fraction of notional ≈ feeRate · (1 − p)."""
    p = min(max(float(price), 1e-12), 1.0)
    return max(float(fee_rate), 0.0) * (1.0 - p)


def fee_model_from_mapping(raw: Mapping[str, Any] | None) -> FeeModel:
    """Rebuild FeeModel from run metadata (forward-compatible)."""
    if not raw:
        return FeeModel()
    version = str(raw.get("fee_model_version") or FEE_MODEL_VERSION)
    decimals = int(raw.get("rounding_decimals") or 6)
    return FeeModel(version=version, rounding_decimals=decimals)


def schedule_as_dicts(model: FeeModel | None = None) -> list[dict[str, Any]]:
    fm = model or FeeModel()
    return [asdict(e) for e in fm.schedule]
