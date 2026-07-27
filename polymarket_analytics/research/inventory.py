"""Parameter inventory detection helpers (acceptance items 21–24)."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

ParamStatus = Literal[
    "Active",
    "Declared but unused",
    "Partially wired",
    "Conflicting",
    "Unreachable",
    "Hardcoded",
]

ParamCategory = Literal[
    "signal/feature",
    "entry",
    "exit",
    "risk",
    "execution",
    "fee",
    "microstructure",
    "calibration",
    "lifecycle",
    "exogenous",
    "data/ingest",
    "backtest/validation",
    "CLI/config",
    "other",
]


@dataclass
class ParameterInventoryItem:
    name: str
    category: ParamCategory
    declaration_location: str
    current_default: Any
    existing_sweep_range: str | list[Any]
    recommended_sweep_range: str | list[Any]
    usage_sites: list[str] = field(default_factory=list)
    status: ParamStatus = "Active"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Known conflicting conceptual parameters (code-audited).
KNOWN_CONFLICTS: tuple[dict[str, Any], ...] = (
    {
        "concept": "stall_hours",
        "locations": [
            "swing_trader.SwingConfig.stall_hours=36.0",
            "cli.swing-trade --stall-hours default=36.0",
        ],
        "notes": "Resolved 2026-07-26: CLI default aligned to SwingConfig/profiles (was 48.0)",
        "status": "Resolved",
    },
    {
        "concept": "cooldown_sec (swing)",
        "locations": [
            "swing_trader.SwingConfig.cooldown_sec=300.0",
            "profiles._build_* cooldown_sec=120.0",
        ],
        "notes": "Incubator profiles tighten cooldown vs bare SwingConfig default",
    },
    {
        "concept": "book_imbalance_min",
        "locations": [
            "SwingConfig.book_imbalance_min=3.0",
            "SwingConfig.confluence_book_min=2.5",
            "profiles confluence uses 2.5",
        ],
        "notes": "Standalone book strategy vs confluence threshold diverge",
    },
    {
        "concept": "min_ev paper vs swing CLI",
        "locations": [
            "cli.paper-trade --min-ev default=0.10 (→10pp)",
            "cli.swing-trade --min-ev default=0.08 (→8pp)",
            "PaperConfig.min_oos_ev_pct=10.0",
            "SwingConfig.min_ev_pct=8.0",
        ],
        "notes": "Intentional family difference but easy to confuse",
    },
    {
        "concept": "whale threshold",
        "locations": [
            "schema.WHALE_RATIO_DIVERGENCE_THRESHOLD=3.0",
            "backtest.DEFAULT_WHALE_THRESHOLDS=(None,2.0,3.0)",
            "profiles whale min_whale_ratio=3.0",
        ],
        "notes": "Aligned at 3.0 for divergence; grid also sweeps 2.0",
    },
)


def detect_cli_defaults(cli_source: str) -> list[dict[str, Any]]:
    """Parse argparse add_argument defaults from cli.py source text."""
    pattern = re.compile(
        r'add_argument\(\s*[\'"](--[\w-]+)[\'"].*?default\s*=\s*([^,\)]+)',
        re.DOTALL,
    )
    found: list[dict[str, Any]] = []
    for m in pattern.finditer(cli_source):
        name = m.group(1)
        raw = m.group(2).strip()
        found.append({"flag": name, "default_expr": raw})
    return found


def detect_hardcoded_thresholds(source: str, *, path: str = "") -> list[dict[str, Any]]:
    """
    Heuristic: numeric comparisons in if/while with magic literals.

    Flags comparisons like `x > 3.0` / `ret >= 0.20` outside obvious 0/1/-1.
    """
    ignore = {0, 1, -1, 0.0, 1.0, -1.0, 2, 10, 100}
    hits: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return hits

    class Visitor(ast.NodeVisitor):
        def visit_Compare(self, node: ast.Compare) -> None:
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
                    if comp.value in ignore:
                        continue
                    hits.append(
                        {
                            "path": path,
                            "lineno": getattr(node, "lineno", None),
                            "literal": comp.value,
                            "snippet": ast.unparse(node) if hasattr(ast, "unparse") else "",
                        }
                    )
            self.generic_visit(node)

    Visitor().visit(tree)
    return hits


def detect_conflicting_defaults(
    items: Sequence[ParameterInventoryItem],
) -> list[dict[str, Any]]:
    """Group items by name; flag when defaults disagree."""
    by_name: dict[str, list[ParameterInventoryItem]] = {}
    for it in items:
        by_name.setdefault(it.name, []).append(it)
    conflicts: list[dict[str, Any]] = []
    for name, group in by_name.items():
        defaults = {json.dumps(g.current_default, sort_keys=True, default=str) for g in group}
        if len(defaults) > 1 and len(group) > 1:
            conflicts.append(
                {
                    "name": name,
                    "defaults": [g.current_default for g in group],
                    "locations": [g.declaration_location for g in group],
                }
            )
    # Merge known conceptual conflicts
    for kc in KNOWN_CONFLICTS:
        conflicts.append(dict(kc))
    return conflicts


def detect_declared_but_unused(
    declared_names: Iterable[str],
    used_names: Iterable[str],
) -> list[str]:
    declared = set(declared_names)
    used = set(used_names)
    return sorted(declared - used)


def configuration_precedence() -> list[str]:
    """Documented effective precedence for this repo."""
    return [
        "1. Explicit CLI flags (argparse) when command invoked",
        "2. Profile builder overrides (profiles.py) for multi-profile incubator",
        "3. Dataclass field defaults (PaperConfig / SwingConfig / StrategyParams)",
        "4. Module-level constants (schema.WHALE_*, backtest.DEFAULT_* grids)",
        "5. Hardcoded literals inside functions (lowest; flagged Hardcoded)",
        "Note: hierarchical YAML (config/research_priors.yaml) is advisory recommended "
        "priors and does NOT override code defaults unless explicitly loaded.",
    ]


def summarize_counts(
    items: Sequence[ParameterInventoryItem],
) -> dict[str, Any]:
    by_cat: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for it in items:
        by_cat[it.category] = by_cat.get(it.category, 0) + 1
        by_status[it.status] = by_status.get(it.status, 0) + 1
    return {
        "total": len(items),
        "by_category": by_cat,
        "by_status": by_status,
    }


def build_baseline_inventory() -> list[ParameterInventoryItem]:
    """Hand-audited inventory of effective runtime parameters (Phase 1 gate)."""
    items: list[ParameterInventoryItem] = []

    def add(**kwargs: Any) -> None:
        items.append(ParameterInventoryItem(**kwargs))

    # --- schema / features ---
    add(
        name="WHALE_RATIO_DIVERGENCE_THRESHOLD",
        category="signal/feature",
        declaration_location="schema.py:WHALE_RATIO_DIVERGENCE_THRESHOLD",
        current_default=3.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[2.0, 2.5, 3.0, 4.0, 5.0],
        usage_sites=["features.compute_composite_features"],
        status="Active",
        notes="Used for price_volume_divergence boolean",
    )
    add(
        name="DECAY_TTR_FLOOR",
        category="signal/feature",
        declaration_location="schema.py:DECAY_TTR_FLOOR",
        current_default=0.1,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[0.05, 0.1, 0.25, 0.5],
        usage_sites=["features.compute_composite_features"],
        status="Active",
    )
    add(
        name="FEATURE_WINDOWS",
        category="signal/feature",
        declaration_location="schema.py:FEATURE_WINDOWS",
        current_default=["1h", "6h", "24h"],
        existing_sweep_range="Not defined",
        recommended_sweep_range=["30m", "1h", "6h", "24h", "72h"],
        usage_sites=["features.compute_rolling_features"],
        status="Active",
    )
    add(
        name="PRICE_BUCKET_BREAKS",
        category="signal/feature",
        declaration_location="schema.py:PRICE_BUCKET_BREAKS",
        current_default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        existing_sweep_range="Not defined (labels used as grid axis)",
        recommended_sweep_range="Keep; optionally finer near 0.45-0.55",
        usage_sites=["features.assign_price_bucket", "backtest.iter_grid_params"],
        status="Active",
    )
    add(
        name="_EPS_HOURS",
        category="signal/feature",
        declaration_location="features.py:_EPS_HOURS",
        current_default=1e-6,
        existing_sweep_range="Not defined",
        recommended_sweep_range="Not defined — numerical floor",
        usage_sites=["features._rolling_window_frame"],
        status="Hardcoded",
        notes="Module private constant",
    )

    # --- backtest grid ---
    add(
        name="DEFAULT_SPIKE_THRESHOLDS",
        category="entry",
        declaration_location="backtest.py:DEFAULT_SPIKE_THRESHOLDS",
        current_default=[None, 1.5, 2.0, 3.0],
        existing_sweep_range=[None, 1.5, 2.0, 3.0],
        recommended_sweep_range=[None, 1.25, 1.5, 2.0, 3.0, 4.0],
        usage_sites=["backtest.iter_grid_params", "backtest.find_edges"],
        status="Active",
    )
    add(
        name="DEFAULT_TTR_BOUNDS",
        category="lifecycle",
        declaration_location="backtest.py:DEFAULT_TTR_BOUNDS",
        current_default=[None, 24.0, 48.0, 72.0],
        existing_sweep_range=[None, 24.0, 48.0, 72.0],
        recommended_sweep_range=[None, 6.0, 12.0, 24.0, 48.0, 72.0, 168.0],
        usage_sites=["backtest.iter_grid_params"],
        status="Active",
    )
    add(
        name="DEFAULT_WHALE_THRESHOLDS",
        category="entry",
        declaration_location="backtest.py:DEFAULT_WHALE_THRESHOLDS",
        current_default=[None, 2.0, 3.0],
        existing_sweep_range=[None, 2.0, 3.0],
        recommended_sweep_range=[None, 1.5, 2.0, 3.0, 4.0, 5.0],
        usage_sites=["backtest.iter_grid_params"],
        status="Active",
    )
    add(
        name="DEFAULT_MOMENTUM_SIGNS",
        category="entry",
        declaration_location="backtest.py:DEFAULT_MOMENTUM_SIGNS",
        current_default=["any", "pos", "neg"],
        existing_sweep_range=["any", "pos", "neg"],
        recommended_sweep_range=["any", "pos", "neg"],
        usage_sites=["backtest.iter_grid_params"],
        status="Active",
    )
    add(
        name="find_edges.min_samples",
        category="backtest/validation",
        declaration_location="backtest.py:find_edges",
        current_default=5,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[20, 50, 100],
        usage_sites=["backtest.find_edges", "cli.find-edges --min-samples"],
        status="Active",
        notes="CLI default 5 is aggressive for inference; recommend >=20 OOS",
    )
    add(
        name="run_oos_edge_validation.min_samples",
        category="backtest/validation",
        declaration_location="backtest.py:run_oos_edge_validation",
        current_default=20,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[20, 50, 100],
        usage_sites=["backtest.run_oos_edge_validation"],
        status="Active",
    )

    # --- fees ---
    add(
        name="CATEGORY_FEE_RATES",
        category="fee",
        declaration_location="paper_trader.py:CATEGORY_FEE_RATES",
        current_default={"crypto": 0.07, "sports": 0.05, "geopolitics": 0.0},
        existing_sweep_range="Not defined",
        recommended_sweep_range="Use FeeModel schedule / PIT versions; do not sweep rates as strategy knobs",
        usage_sites=["paper_trader.resolve_fee_rate", "research.fees"],
        status="Active",
        notes="Duplicated into research.fees.CATEGORY_TAKER_FEE_RATES with FEE_MODEL_VERSION",
    )
    add(
        name="PaperConfig.use_dynamic_fees",
        category="fee",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=True,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[True],
        usage_sites=["PaperTrader.compute_taker_fee"],
        status="Active",
    )
    add(
        name="PaperConfig.fee_category",
        category="fee",
        declaration_location="paper_trader.py:PaperConfig",
        current_default="crypto",
        existing_sweep_range="Not defined",
        recommended_sweep_range="Market-specific PIT category; not a free sweep axis",
        usage_sites=["PaperTrader", "cli --fee-category"],
        status="Active",
    )
    add(
        name="PaperConfig.fee_bps",
        category="fee",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=0.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range="Legacy only when use_dynamic_fees=False",
        usage_sites=["PaperTrader.compute_taker_fee"],
        status="Partially wired",
        notes="Active only with --flat-fees",
    )
    add(
        name="PaperConfig.spread_slippage_bps",
        category="execution",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=50.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[10, 25, 50, 100, 200],
        usage_sites=["PaperTrader.fill modeling"],
        status="Active",
    )

    # --- paper risk ---
    add(
        name="PaperConfig.bankroll",
        category="risk",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=10_000.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range="Not a strategy parameter",
        usage_sites=["PaperTrader", "cli --bankroll"],
        status="Active",
    )
    add(
        name="PaperConfig.kelly_fraction",
        category="risk",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=0.25,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[0.1, 0.25, 0.5],
        usage_sites=["kelly_fraction", "cli --kelly-fraction"],
        status="Active",
    )
    add(
        name="PaperConfig.max_position_pct",
        category="risk",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=0.05,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[0.02, 0.05, 0.10],
        usage_sites=["PaperTrader sizing"],
        status="Active",
    )
    add(
        name="PaperConfig.min_oos_ev_pct",
        category="entry",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=10.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[5.0, 8.0, 10.0, 15.0],
        usage_sites=["PaperTrader signal gate", "cli --min-ev"],
        status="Active",
    )
    add(
        name="PaperConfig.max_open_positions",
        category="risk",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=25,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[10, 15, 25, 40],
        usage_sites=["PaperTrader"],
        status="Active",
    )
    add(
        name="PaperConfig.cooldown_sec",
        category="lifecycle",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=60.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[30, 60, 120, 300],
        usage_sites=["PaperTrader"],
        status="Active",
    )
    add(
        name="PaperConfig.take_profit_pct",
        category="exit",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=None,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[None, 0.10, 0.20, 0.30],
        usage_sites=["PaperTrader early exit"],
        status="Active",
    )
    add(
        name="PaperConfig.stop_loss_pct",
        category="exit",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=None,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[None, 0.05, 0.10, 0.15],
        usage_sites=["PaperTrader early exit"],
        status="Active",
    )
    add(
        name="PaperConfig.resolve_poll_sec",
        category="lifecycle",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=60.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[30, 60, 120],
        usage_sites=["PaperTrader resolution poll"],
        status="Active",
    )

    # --- swing ---
    for name, default, cat, rec, status, notes in [
        ("SwingConfig.bankroll", 10_000.0, "risk", "N/A", "Active", ""),
        ("SwingConfig.position_pct", 0.05, "risk", [0.02, 0.05, 0.10], "Active", ""),
        ("SwingConfig.min_liquidity_usd", 50_000.0, "entry", [10_000, 25_000, 50_000, 100_000], "Active", ""),
        ("SwingConfig.min_ev_pct", 8.0, "entry", [5.0, 8.0, 10.0, 12.0], "Active", ""),
        ("SwingConfig.rsi_period", 14, "signal/feature", [7, 14, 21], "Active", ""),
        ("SwingConfig.rsi_oversold", 25.0, "entry", [20, 25, 30, 35], "Active", "profiles RSI use 25; confluence 30"),
        ("SwingConfig.bb_period", 20, "signal/feature", [10, 20, 30], "Active", ""),
        ("SwingConfig.bb_std", 2.5, "signal/feature", [2.0, 2.5, 3.0], "Active", ""),
        ("SwingConfig.hurst_min", 0.55, "entry", [0.50, 0.55, 0.60, 0.65], "Active", ""),
        ("SwingConfig.ema_fast", 5, "signal/feature", [3, 5, 8], "Active", ""),
        ("SwingConfig.ema_slow", 20, "signal/feature", [15, 20, 30], "Active", ""),
        ("SwingConfig.momentum_min_move", 0.10, "entry", [0.05, 0.08, 0.10, 0.15], "Active", "profiles momentum use 0.08"),
        ("SwingConfig.whale_volume_ratio", 2.0, "entry", [1.5, 2.0, 3.0], "Active", ""),
        ("SwingConfig.book_imbalance_min", 3.0, "microstructure", [2.0, 2.5, 3.0, 4.0], "Conflicting", "vs confluence_book_min=2.5"),
        ("SwingConfig.book_price_lo", 0.20, "entry", [0.10, 0.20, 0.30], "Active", ""),
        ("SwingConfig.book_price_hi", 0.80, "entry", [0.70, 0.80, 0.90], "Active", ""),
        ("SwingConfig.take_profit_pct", 0.20, "exit", [0.10, 0.15, 0.20, 0.30], "Active", ""),
        ("SwingConfig.take_profit_atr_mult", 2.0, "exit", [1.5, 2.0, 3.0], "Active", ""),
        ("SwingConfig.stop_loss_pct", 0.10, "exit", [0.05, 0.10, 0.15], "Active", ""),
        ("SwingConfig.stop_loss_atr_mult", 1.0, "exit", [0.5, 1.0, 1.5], "Active", ""),
        ("SwingConfig.atr_period", 14, "signal/feature", [7, 14, 21], "Active", ""),
        ("SwingConfig.atr_stop_mult", 2.0, "exit", [1.5, 2.0, 3.0], "Active", ""),
        ("SwingConfig.stall_hours", 36.0, "lifecycle", [24, 36, 48, 72], "Active", "CLI default aligned to 36.0"),
        ("SwingConfig.use_bb_take_profit", True, "exit", [True, False], "Active", ""),
        ("SwingConfig.max_open_positions", 15, "risk", [10, 15, 25], "Active", ""),
        ("SwingConfig.cooldown_sec", 300.0, "lifecycle", [60, 120, 300], "Conflicting", "profiles use 120"),
        ("SwingConfig.history_len", 200, "signal/feature", [100, 200, 500], "Active", ""),
        ("SwingConfig.require_confluence", False, "entry", [True, False], "Active", ""),
        ("SwingConfig.min_confluence", 2, "entry", [2, 3], "Active", ""),
        ("SwingConfig.confluence_rsi", 30.0, "entry", [25, 30, 35], "Active", ""),
        ("SwingConfig.confluence_volume_usd", 25_000.0, "entry", [10_000, 25_000, 50_000], "Active", ""),
        ("SwingConfig.confluence_book_min", 2.5, "microstructure", [2.0, 2.5, 3.0], "Conflicting", ""),
    ]:
        add(
            name=name,
            category=cat,  # type: ignore[arg-type]
            declaration_location="swing_trader.py:SwingConfig",
            current_default=default,
            existing_sweep_range="Not defined",
            recommended_sweep_range=rec,
            usage_sites=["SwingTrader", "profiles.py"],
            status=status,  # type: ignore[arg-type]
            notes=notes,
        )

    # --- CLI-only / ingest ---
    add(
        name="DEFAULT_CHUNK_ROWS",
        category="data/ingest",
        declaration_location="ingest.py:DEFAULT_CHUNK_ROWS",
        current_default=500_000,
        existing_sweep_range="Not defined",
        recommended_sweep_range="Not a research parameter",
        usage_sites=["ingest.run_ingest", "cli --chunk-rows"],
        status="Active",
    )
    add(
        name="cli.paper-trade.duration",
        category="CLI/config",
        declaration_location="cli.py:paper-trade --duration",
        current_default=120.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range="Ops only (GHA uses 18000)",
        usage_sites=["cli._cmd_paper_trade"],
        status="Active",
    )
    add(
        name="cli.swing-trade.stall_hours",
        category="lifecycle",
        declaration_location="cli.py:swing-trade --stall-hours",
        current_default=36.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[24, 36, 48, 72],
        usage_sites=["cli._cmd_swing_trade"],
        status="Active",
        notes="Aligned with SwingConfig.stall_hours=36.0 (was 48.0)",
    )
    add(
        name="StrategyParams.require_price_volume_divergence",
        category="entry",
        declaration_location="backtest.py:StrategyParams",
        current_default=False,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[False, True],
        usage_sites=["apply_strategy_filter"],
        status="Partially wired",
        notes="Field exists but iter_grid_params never sets True — not in default grid",
    )
    add(
        name="momentum_6h grid",
        category="entry",
        declaration_location="backtest.py:iter_grid_params",
        current_default=["any"],
        existing_sweep_range=["any"],
        recommended_sweep_range=["any", "pos", "neg"],
        usage_sites=["find_edges"],
        status="Partially wired",
        notes="Axis declared but default grid freezes momentum_6h at any",
    )
    add(
        name="logit_half_tick",
        category="signal/feature",
        declaration_location="research/logit.py:DEFAULT_HALF_TICK",
        current_default=0.005,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[0.001, 0.005, 0.01],
        usage_sites=["research.logit", "feature_registry.logit_price"],
        status="Active",
        notes="New foundation",
    )
    add(
        name="FEE_MODEL_VERSION",
        category="fee",
        declaration_location="research/fees.py:FEE_MODEL_VERSION",
        current_default="2026-07-polymarket-v1",
        existing_sweep_range="Not defined",
        recommended_sweep_range="Version pin only",
        usage_sites=["compute_fill_fee", "run metadata"],
        status="Active",
        notes="New foundation",
    )
    add(
        name="ExecutionConfig.latency_ms",
        category="execution",
        declaration_location="research/execution.py:ExecutionConfig",
        current_default=50.0,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[0, 50, 100, 250, 500],
        usage_sites=["simulate_aggressive_fill", "PaperConfig.latency_ms"],
        status="Partially wired",
        notes=(
            "Present on PaperConfig and FillResult.meta, but does not delay quote "
            "selection or change fill price (inert until delayed-book model exists)."
        ),
    )
    add(
        name="PaperConfig.use_book_walk",
        category="execution",
        declaration_location="paper_trader.py:PaperConfig",
        current_default=False,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[False, True],
        usage_sites=["PaperTrader.apply_fill_price"],
        status="Partially wired",
        notes="Opt-in L1 ask cross; CLI does not expose the flag; no historical L2",
    )
    add(
        name="PaperTrader.matches.min_volume_spike",
        category="entry",
        declaration_location="paper_trader.py:PaperTrader.matches",
        current_default=None,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[None, 1.5, 2.0, 3.0],
        usage_sites=["PaperTrader.matches"],
        status="Declared but unused",
        notes="If set, matches() always returns False — LiveFeatures lacks volume_spike",
    )
    add(
        name="PaperTrader.matches.max_time_to_resolution_hours",
        category="entry",
        declaration_location="paper_trader.py:PaperTrader.matches",
        current_default=None,
        existing_sweep_range="Not defined",
        recommended_sweep_range=[None, 24, 48, 72],
        usage_sites=["PaperTrader.matches"],
        status="Declared but unused",
        notes="If set, matches() always returns False — LiveFeatures lacks TTR",
    )
    add(
        name="exogenous.*",
        category="exogenous",
        declaration_location="research/exogenous.py",
        current_default=None,
        existing_sweep_range="Not defined",
        recommended_sweep_range="Blocked until PIT providers exist",
        usage_sites=[],
        status="Declared but unused",
        notes="Provider stubs only",
    )

    return items


def write_inventory_artifacts(out_dir: Path) -> dict[str, Path]:
    """Write markdown/json/csv inventory + analysis reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build_baseline_inventory()
    conflicts = detect_conflicting_defaults(items)
    unused = [it for it in items if it.status == "Declared but unused"]
    hardcoded = [it for it in items if it.status == "Hardcoded"]
    partial = [it for it in items if it.status == "Partially wired"]
    counts = summarize_counts(items)

    json_path = out_dir / "parameter_inventory.json"
    json_path.write_text(
        json.dumps([it.to_dict() for it in items], indent=2, default=str),
        encoding="utf-8",
    )

    # CSV
    csv_path = out_dir / "parameter_inventory.csv"
    headers = [
        "name",
        "category",
        "declaration_location",
        "current_default",
        "existing_sweep_range",
        "recommended_sweep_range",
        "status",
        "notes",
    ]
    lines = [",".join(headers)]
    for it in items:
        row = [
            it.name,
            it.category,
            it.declaration_location,
            json.dumps(it.current_default, default=str).replace(",", ";"),
            json.dumps(it.existing_sweep_range, default=str).replace(",", ";"),
            json.dumps(it.recommended_sweep_range, default=str).replace(",", ";"),
            it.status,
            it.notes.replace(",", ";"),
        ]
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    conflict_path = out_dir / "conflict_report.md"
    conflict_path.write_text(
        "# Configuration conflict report\n\n"
        + "\n".join(
            f"- **{c.get('concept', c.get('name'))}**: {c}\n" for c in conflicts
        ),
        encoding="utf-8",
    )

    unused_path = out_dir / "unused_parameter_report.md"
    unused_path.write_text(
        "# Declared-but-unused / partially wired\n\n## Unused\n"
        + "\n".join(f"- `{u.name}`: {u.notes}" for u in unused)
        + "\n\n## Partially wired\n"
        + "\n".join(f"- `{p.name}`: {p.notes}" for p in partial),
        encoding="utf-8",
    )

    hard_path = out_dir / "hardcoded_threshold_report.md"
    # Also scan package sources
    root = Path(__file__).resolve().parents[2]
    pkg = root / "polymarket_analytics"
    scan_hits: list[dict[str, Any]] = []
    for py in pkg.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        scan_hits.extend(detect_hardcoded_thresholds(py.read_text(encoding="utf-8"), path=str(py.relative_to(root))))
    hard_path.write_text(
        "# Hardcoded threshold report\n\n## Inventory-flagged\n"
        + "\n".join(f"- `{h.name}` @ {h.declaration_location} = {h.current_default}" for h in hardcoded)
        + "\n\n## AST heuristic hits (sample)\n"
        + "\n".join(
            f"- `{h['path']}:{h['lineno']}` literal={h['literal']} `{h['snippet'][:80]}`"
            for h in scan_hits[:80]
        ),
        encoding="utf-8",
    )

    missing_tests_path = out_dir / "missing_test_report.md"
    missing_tests_path.write_text(
        """# Missing-test report

## Covered (pre-expansion)
- Phase 1 ingest, Phase 2 features, Phase 3 backtest/edges, OOS split
- Phase 4 paper feed, Phase 5 fees/resolution/TP-SL
- Profiles + swing trader unit tests

## Gaps addressed by new tests
- Logit clamp / logit edge
- Fee model version + maker vs taker + rounding + fee-free
- Complete-set residual when YES/NO present
- OFI helper (synthetic levels)
- PIT calibrator fit-on-train-only
- Event-grouped purge / embargo folds
- TP/SL same-tick ordering
- Inventory detectors (CLI defaults, conflicts, unused, hardcoded)

## Still thin / blocked
- Full L2 OFI on historical lake (no depth data)
- Exogenous providers (interfaces only)
- Cross-market lead-lag
- End-to-end PaperTrader latency/book-walk path (execution helpers unit-tested only)
""",
        encoding="utf-8",
    )

    precedence_path = out_dir / "configuration_precedence_report.md"
    precedence_path.write_text(
        "# Configuration precedence\n\n"
        + "\n".join(f"- {line}" for line in configuration_precedence()),
        encoding="utf-8",
    )

    counts_path = out_dir / "inventory_summary_counts.json"
    counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    # Full markdown inventory sections A–D
    md_path = out_dir / "parameter_inventory.md"
    sections = {
        "A": [i for i in items if i.category in {"signal/feature", "entry", "exit", "microstructure"}],
        "B": [i for i in items if i.category in {"fee", "execution", "risk", "lifecycle"}],
        "C": [i for i in items if i.category in {"backtest/validation", "calibration", "CLI/config", "data/ingest"}],
        "D": [i for i in items if i.category in {"exogenous", "other"}],
    }
    md: list[str] = [
        "# Parameter inventory (Phase 1)\n",
        "Code-defined defaults and existing sweep ranges are taken from the repo.",
        "Recommended ranges are labeled separately and are **not** claimed as existing.\n",
        f"Total parameters: **{counts['total']}**\n",
    ]
    titles = {
        "A": "A — Signals, entries, exits, microstructure",
        "B": "B — Fees, execution, risk, lifecycle",
        "C": "C — Validation, calibration, CLI, ingest",
        "D": "D — Exogenous / other",
    }
    for key in ("A", "B", "C", "D"):
        md.append(f"## {titles[key]}\n")
        for it in sections[key]:
            md.append(f"### `{it.name}`")
            md.append(f"- Category: {it.category}")
            md.append(f"- Declaration: `{it.declaration_location}`")
            md.append(f"- Current default: `{it.current_default}`")
            md.append(f"- Existing sweep range: `{it.existing_sweep_range}`")
            md.append(f"- Recommended sweep range: `{it.recommended_sweep_range}`")
            md.append(f"- Usage: {', '.join(it.usage_sites) or '—'}")
            md.append(f"- Status: **{it.status}**")
            if it.notes:
                md.append(f"- Notes: {it.notes}")
            md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")

    return {
        "markdown": md_path,
        "json": json_path,
        "csv": csv_path,
        "conflicts": conflict_path,
        "unused": unused_path,
        "hardcoded": hard_path,
        "missing_tests": missing_tests_path,
        "precedence": precedence_path,
        "counts": counts_path,
    }
