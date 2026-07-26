"""Multi-profile forward-testing incubator — isolated virtual bankrolls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from polymarket_analytics.backtest import StrategyParams
from polymarket_analytics.live_feed import LiveFeatures
from polymarket_analytics.paper_trader import PaperConfig, PaperTrader
from polymarket_analytics.swing_trader import SwingConfig, SwingTrader, TokenBar

ProfileKind = Literal["paper", "swing"]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

JOURNAL_WHALE = DATA / "paper_trades_whale.json"
JOURNAL_RSI = DATA / "paper_trades_rsi.json"
JOURNAL_MOMENTUM = DATA / "paper_trades_momentum.json"
JOURNAL_SWING = DATA / "paper_trades_swing.json"

DEFAULT_BANKROLL = 10_000.0


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    kind: ProfileKind
    journal: Path
    description: str


PROFILE_SPECS: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        name="profile_whale_midbucket",
        kind="paper",
        journal=JOURNAL_WHALE,
        description="Whale Ratio > 3 on 0.40–0.50, min EV +10pp",
    ),
    ProfileSpec(
        name="profile_rsi_mean_reversion",
        kind="swing",
        journal=JOURNAL_RSI,
        description="RSI < 25 on >$50k liq → 20-EMA target",
    ),
    ProfileSpec(
        name="profile_momentum_drift",
        kind="swing",
        journal=JOURNAL_MOMENTUM,
        description="Hurst > 0.55 + 5/20 EMA cross on heavy volume",
    ),
    ProfileSpec(
        name="profile_swing_confluence",
        kind="swing",
        journal=JOURNAL_SWING,
        description="2+ of book imbalance / EMA cross / RSI oversold",
    ),
)


@dataclass
class ProfileRuntime:
    spec: ProfileSpec
    paper: PaperTrader | None = None
    swing: SwingTrader | None = None
    peak_equity: float = DEFAULT_BANKROLL
    max_drawdown_pct: float = 0.0

    def equity(self) -> float:
        if self.paper is not None:
            return self.paper.equity
        assert self.swing is not None
        return self.swing.equity

    def update_drawdown(self) -> None:
        eq = self.equity()
        if eq > self.peak_equity:
            self.peak_equity = eq
        if self.peak_equity > 0:
            dd = 100.0 * (self.peak_equity - eq) / self.peak_equity
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

    def on_tick(self, feat: LiveFeatures, bar: TokenBar) -> None:
        if self.paper is not None:
            self.paper.on_features(feat)
        if self.swing is not None:
            strategies = None
            if self.spec.name == "profile_rsi_mean_reversion":
                strategies = ("mean_reversion",)
            elif self.spec.name == "profile_momentum_drift":
                strategies = ("momentum",)
            elif self.spec.name == "profile_swing_confluence":
                strategies = ("confluence",)
            self.swing.on_bar(feat.token_id, feat.condition_id, bar, strategies=strategies)
        self.update_drawdown()

    def persist(self) -> None:
        engine = self.paper if self.paper is not None else self.swing
        assert engine is not None
        snap = engine.snapshot()
        snap["profile"] = self.spec.name
        snap["max_drawdown_pct"] = self.max_drawdown_pct
        snap["peak_equity"] = self.peak_equity
        path = self.spec.journal
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    def metrics(self) -> dict[str, Any]:
        if self.paper is not None:
            t = self.paper
            closed = [p for p in t.positions if p.status != "open"]
            wins = sum(1 for p in closed if p.win is True or (p.realized_pnl or 0) > 0)
            fees = sum(
                float(p.fees) + float(getattr(p, "exit_fees", 0) or 0) for p in t.positions
            )
            realized, unrealized, fills = t.realized_pnl, t.unrealized_pnl, t.fills
            bankroll, cash = t.config.bankroll, t.cash
        else:
            assert self.swing is not None
            t = self.swing
            closed = [p for p in t.positions if p.status != "open"]
            wins = sum(1 for p in closed if (p.realized_pnl or 0) > 0)
            fees = 0.0
            realized, unrealized, fills = t.realized_pnl, t.unrealized_pnl, t.fills
            bankroll, cash = t.config.bankroll, t.cash
        n_closed = len(closed)
        total_pnl = realized + unrealized
        roi = 100.0 * total_pnl / bankroll if bankroll else 0.0
        win_rate = 100.0 * wins / n_closed if n_closed else 0.0
        return {
            "profile": self.spec.name,
            "description": self.spec.description,
            "kind": self.spec.kind,
            "journal": str(self.spec.journal),
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": total_pnl,
            "roi_pct": roi,
            "win_rate_pct": win_rate,
            "fills": fills,
            "n_closed": n_closed,
            "fees_paid": fees,
            "max_drawdown_pct": self.max_drawdown_pct,
            "equity": self.equity(),
            "bankroll": bankroll,
            "cash": cash,
        }


def _build_whale_trader(journal: Path, bankroll: float, min_ev_pct: float) -> PaperTrader:
    cfg = PaperConfig(
        bankroll=bankroll,
        min_oos_ev_pct=min_ev_pct,
        require_persists=False,
        use_dynamic_fees=True,
        fee_category="crypto",
        cooldown_sec=60.0,
    )
    strategies = [
        (
            StrategyParams(
                price_bucket="0.40-0.50",
                min_whale_ratio=3.0,
                momentum_1h="any",
                side="BUY",
            ),
            max(min_ev_pct, 10.0),
            0.55,
        )
    ]
    return PaperTrader(config=cfg, strategies=strategies, journal_path=journal)


def _build_rsi_trader(journal: Path, bankroll: float, min_ev_pct: float) -> SwingTrader:
    return SwingTrader(
        config=SwingConfig(
            bankroll=bankroll,
            min_ev_pct=min_ev_pct,
            min_liquidity_usd=50_000.0,
            rsi_oversold=25.0,
            stall_hours=36.0,
            take_profit_pct=0.20,
            atr_stop_mult=2.0,
            require_confluence=False,
            cooldown_sec=120.0,
        ),
        journal_path=journal,
    )


def _build_momentum_trader(journal: Path, bankroll: float, min_ev_pct: float) -> SwingTrader:
    return SwingTrader(
        config=SwingConfig(
            bankroll=bankroll,
            min_ev_pct=min_ev_pct,
            hurst_min=0.55,
            ema_fast=5,
            ema_slow=20,
            whale_volume_ratio=2.0,
            momentum_min_move=0.08,
            stall_hours=36.0,
            take_profit_pct=0.20,
            require_confluence=False,
            cooldown_sec=120.0,
        ),
        journal_path=journal,
    )


def _build_confluence_trader(journal: Path, bankroll: float, min_ev_pct: float) -> SwingTrader:
    return SwingTrader(
        config=SwingConfig(
            bankroll=bankroll,
            min_ev_pct=min_ev_pct,
            require_confluence=True,
            min_confluence=2,
            book_imbalance_min=2.5,
            confluence_book_min=2.5,
            confluence_rsi=30.0,
            confluence_volume_usd=25_000.0,
            rsi_oversold=30.0,
            min_liquidity_usd=25_000.0,
            stall_hours=36.0,
            take_profit_pct=0.20,
            take_profit_atr_mult=2.0,
            stop_loss_atr_mult=1.0,
            stop_loss_pct=0.10,
            cooldown_sec=120.0,
        ),
        journal_path=journal,
    )


@dataclass
class MultiProfileIncubator:
    """Routes each live tick through all isolated profile sub-accounts."""

    profiles: list[ProfileRuntime] = field(default_factory=list)
    bankroll: float = DEFAULT_BANKROLL

    @classmethod
    def create(
        cls,
        *,
        bankroll: float = DEFAULT_BANKROLL,
        min_ev_pct: float = 10.0,
        data_dir: Path | None = None,
    ) -> "MultiProfileIncubator":
        data = data_dir or DATA
        data.mkdir(parents=True, exist_ok=True)
        runtimes: list[ProfileRuntime] = []
        for base in PROFILE_SPECS:
            journal = data / Path(base.journal).name
            spec = ProfileSpec(
                name=base.name,
                kind=base.kind,
                journal=journal,
                description=base.description,
            )
            rt = ProfileRuntime(spec=spec, peak_equity=bankroll)
            if spec.name == "profile_whale_midbucket":
                rt.paper = _build_whale_trader(journal, bankroll, min_ev_pct)
            elif spec.name == "profile_rsi_mean_reversion":
                rt.swing = _build_rsi_trader(journal, bankroll, max(min_ev_pct * 0.5, 1.0))
            elif spec.name == "profile_momentum_drift":
                rt.swing = _build_momentum_trader(journal, bankroll, max(min_ev_pct * 0.5, 1.0))
            elif spec.name == "profile_swing_confluence":
                rt.swing = _build_confluence_trader(journal, bankroll, max(min_ev_pct * 0.5, 1.0))
            runtimes.append(rt)
        inc = cls(profiles=runtimes, bankroll=bankroll)
        inc.persist_all()
        return inc

    def on_features(self, feat: LiveFeatures) -> None:
        bar = TokenBar(
            ts=feat.ts,
            mid=feat.price,
            volume=feat.size,
            whale_ratio=feat.whale_ratio,
            liquidity_usd=50_000.0,
        )
        if feat.best_bid is not None and feat.best_ask is not None:
            mid = feat.price
            bar.bid_depth = 1.0 / max(mid - feat.best_bid, 1e-4)
            bar.ask_depth = 1.0 / max(feat.best_ask - mid, 1e-4)
        for rt in self.profiles:
            rt.on_tick(feat, bar)
        self.persist_all()

    def on_bar(self, token_id: str, condition_id: str, bar: TokenBar) -> None:
        feat = LiveFeatures(
            token_id=token_id,
            condition_id=condition_id,
            price=bar.mid,
            size=bar.volume,
            ts=bar.ts,
            momentum_1h=None,
            whale_ratio=bar.whale_ratio,
            n_trades_1h=1,
            best_bid=None,
            best_ask=None,
        )
        for rt in self.profiles:
            rt.on_tick(feat, bar)
        self.persist_all()

    def persist_all(self) -> None:
        for rt in self.profiles:
            rt.persist()

    def comparison_rows(self) -> list[dict[str, Any]]:
        return [rt.metrics() for rt in self.profiles]


def format_profile_leaderboard(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = (
        "profile",
        "PnL($)",
        "ROI%",
        "Win%",
        "fills",
        "fees($)",
        "maxDD%",
        "equity",
    )
    lines = [
        "=" * 96,
        " MULTI-PROFILE FORWARD TEST LEADERBOARD",
        "=" * 96,
        " | ".join(f"{h:>12}" for h in headers),
        " | ".join("-" * 12 for _ in headers),
    ]
    ranked = sorted(rows, key=lambda r: float(r.get("total_pnl") or 0), reverse=True)
    for r in ranked:
        name = str(r.get("profile", ""))[:12]
        lines.append(
            " | ".join(
                [
                    f"{name:>12}",
                    f"{float(r.get('total_pnl') or 0):>+12.2f}",
                    f"{float(r.get('roi_pct') or 0):>+12.2f}",
                    f"{float(r.get('win_rate_pct') or 0):>12.1f}",
                    f"{int(r.get('fills') or 0):>12d}",
                    f"{float(r.get('fees_paid') or 0):>12.4f}",
                    f"{float(r.get('max_drawdown_pct') or 0):>12.2f}",
                    f"{float(r.get('equity') or 0):>12.2f}",
                ]
            )
        )
    lines.append("=" * 96)
    return "\n".join(lines)


def load_profile_journals(data_dir: Path | None = None) -> list[dict[str, Any]]:
    data = data_dir or DATA
    rows: list[dict[str, Any]] = []
    for spec in PROFILE_SPECS:
        path = data / Path(spec.journal).name
        if not path.exists():
            rows.append(
                {
                    "profile": spec.name,
                    "description": spec.description,
                    "total_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "roi_pct": 0.0,
                    "win_rate_pct": 0.0,
                    "fills": 0,
                    "fees_paid": 0.0,
                    "max_drawdown_pct": 0.0,
                    "equity": DEFAULT_BANKROLL,
                    "bankroll": DEFAULT_BANKROLL,
                }
            )
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        bankroll = float(raw.get("bankroll_initial") or DEFAULT_BANKROLL)
        realized = float(raw.get("realized_pnl") or 0.0)
        unrealized = float(raw.get("unrealized_pnl") or 0.0)
        total = realized + unrealized
        closed = list(raw.get("closed_positions") or [])
        wins = sum(
            1
            for p in closed
            if p.get("win") is True or float(p.get("realized_pnl") or 0) > 0
        )
        fees = sum(
            float(p.get("fees") or 0) + float(p.get("exit_fees") or 0)
            for p in list(raw.get("open_positions") or []) + closed
        )
        rows.append(
            {
                "profile": raw.get("profile") or spec.name,
                "description": spec.description,
                "total_pnl": total,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "roi_pct": 100.0 * total / bankroll if bankroll else 0.0,
                "win_rate_pct": 100.0 * wins / len(closed) if closed else 0.0,
                "fills": int(raw.get("fills") or 0),
                "fees_paid": fees,
                "max_drawdown_pct": float(raw.get("max_drawdown_pct") or 0.0),
                "equity": float(raw.get("equity") or bankroll),
                "bankroll": bankroll,
            }
        )
    return rows
