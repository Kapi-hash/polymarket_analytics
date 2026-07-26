"""Hierarchical research configuration loader (global → family → regime → feature).

IMPORTANT: Code-defined defaults remain authoritative for runtime unless a caller
explicitly merges this YAML. Recommended priors live under `recommended_priors`
and must not be confused with existing sweep grids in `backtest.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "research_priors.yaml"


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_research_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {"_meta": {"loaded": False, "path": str(path)}}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        # Minimal fallback: JSON subset only
        data = json.loads(text) if text.strip().startswith("{") else {"_raw": text}
    data["_meta"] = {"loaded": True, "path": str(path), "yaml_available": yaml is not None}
    return data


def resolve_hierarchy(
    cfg: Mapping[str, Any],
    *,
    family: str | None = None,
    category: str | None = None,
    liquidity_regime: str | None = None,
    ttr_regime: str | None = None,
) -> dict[str, Any]:
    """Merge global → family → category → liquidity/TTR regime layers."""
    merged: dict[str, Any] = dict(cfg.get("global") or {})
    if family and isinstance(cfg.get("families"), Mapping):
        fam = cfg["families"].get(family) or {}
        if isinstance(fam, Mapping):
            merged = _deep_merge(merged, fam)
    if category and isinstance(cfg.get("categories"), Mapping):
        cat = cfg["categories"].get(category) or {}
        if isinstance(cat, Mapping):
            merged = _deep_merge(merged, cat)
    regimes = cfg.get("regimes") if isinstance(cfg.get("regimes"), Mapping) else {}
    if liquidity_regime and isinstance(regimes.get("liquidity"), Mapping):
        liq = regimes["liquidity"].get(liquidity_regime) or {}
        if isinstance(liq, Mapping):
            merged = _deep_merge(merged, liq)
    if ttr_regime and isinstance(regimes.get("ttr"), Mapping):
        ttr = regimes["ttr"].get(ttr_regime) or {}
        if isinstance(ttr, Mapping):
            merged = _deep_merge(merged, ttr)
    return merged


def existing_code_defaults_snapshot() -> dict[str, Any]:
    """Snapshot of authoritative code defaults (not YAML priors)."""
    from polymarket_analytics.backtest import (
        DEFAULT_MOMENTUM_SIGNS,
        DEFAULT_SPIKE_THRESHOLDS,
        DEFAULT_TTR_BOUNDS,
        DEFAULT_WHALE_THRESHOLDS,
    )
    from polymarket_analytics.paper_trader import PaperConfig
    from polymarket_analytics.schema import (
        DECAY_TTR_FLOOR,
        WHALE_RATIO_DIVERGENCE_THRESHOLD,
    )
    from polymarket_analytics.swing_trader import SwingConfig

    pc = PaperConfig()
    sc = SwingConfig()
    return {
        "source": "code_defined",
        "schema": {
            "WHALE_RATIO_DIVERGENCE_THRESHOLD": WHALE_RATIO_DIVERGENCE_THRESHOLD,
            "DECAY_TTR_FLOOR": DECAY_TTR_FLOOR,
        },
        "edge_grid_existing": {
            "DEFAULT_SPIKE_THRESHOLDS": list(DEFAULT_SPIKE_THRESHOLDS),
            "DEFAULT_TTR_BOUNDS": list(DEFAULT_TTR_BOUNDS),
            "DEFAULT_WHALE_THRESHOLDS": list(DEFAULT_WHALE_THRESHOLDS),
            "DEFAULT_MOMENTUM_SIGNS": list(DEFAULT_MOMENTUM_SIGNS),
            "momentum_6h_default_grid": ["any"],
        },
        "PaperConfig": {k: getattr(pc, k) for k in pc.__dataclass_fields__},
        "SwingConfig": {k: getattr(sc, k) for k in sc.__dataclass_fields__},
    }
