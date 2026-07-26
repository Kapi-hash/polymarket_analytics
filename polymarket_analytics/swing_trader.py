"""Swing trading: multi-hour/day probability swings before resolution."""

from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Literal, Mapping, Sequence

StrategyName = Literal["mean_reversion", "momentum", "book_imbalance"]

DEFAULT_SWING_JOURNAL = Path("data/swing_trades.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sma(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ema(values: Sequence[float], period: int) -> float | None:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1.0)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1.0 - k)
    return ema


def compute_rsi(closes: Sequence[float], period: int = 14) -> float | None:
    """Wilder-style RSI on close prices."""
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss <= 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger(
    closes: Sequence[float],
    period: int = 20,
    num_std: float = 2.5,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (mid/SMA, upper, lower, std)."""
    if len(closes) < period:
        return None, None, None, None
    window = list(closes[-period:])
    mid = sum(window) / period
    var = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(var)
    return mid, mid + num_std * std, mid - num_std * std, std


def compute_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Average True Range (simplified: |high-low| / close-to-close)."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None
    trs: list[float] = []
    for i in range(n - period, n):
        prev_c = closes[i - 1]
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_c),
            abs(lows[i] - prev_c),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


def compute_hurst(closes: Sequence[float], min_lags: int = 8) -> float | None:
    """
    Rough Hurst exponent via rescaled-range (R/S) on log returns.

    H > 0.5 → trending; H < 0.5 → mean-reverting; H ≈ 0.5 → random.
    """
    if len(closes) < min_lags + 2:
        return None
    rets = []
    for i in range(1, len(closes)):
        a, b = closes[i - 1], closes[i]
        if a <= 0 or b <= 0:
            continue
        rets.append(math.log(b / a))
    if len(rets) < min_lags:
        return None

    max_k = min(len(rets) // 2, 32)
    if max_k < 4:
        return None
    ks: list[float] = []
    log_rs: list[float] = []
    for k in range(4, max_k + 1):
        chunks = [rets[i : i + k] for i in range(0, len(rets) - k + 1, k)]
        if not chunks:
            continue
        rs_vals: list[float] = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = sum(chunk) / len(chunk)
            y = 0.0
            path = []
            for r in chunk:
                y += r - mean
                path.append(y)
            r_range = max(path) - min(path)
            std = statistics.pstdev(chunk)
            if std <= 1e-12:
                continue
            rs_vals.append(r_range / std)
        if not rs_vals:
            continue
        rs = sum(rs_vals) / len(rs_vals)
        if rs <= 0:
            continue
        ks.append(math.log(k))
        log_rs.append(math.log(rs))
    if len(ks) < 3:
        return None
    # OLS slope
    n = len(ks)
    mx = sum(ks) / n
    my = sum(log_rs) / n
    num = sum((x - mx) * (y - my) for x, y in zip(ks, log_rs))
    den = sum((x - mx) ** 2 for x in ks)
    if den <= 1e-12:
        return None
    return num / den


@dataclass
class SwingConfig:
    bankroll: float = 10_000.0
    position_pct: float = 0.05
    min_liquidity_usd: float = 50_000.0
    min_ev_pct: float = 8.0  # filter / sizing hint
    # Strategy A
    rsi_period: int = 14
    rsi_oversold: float = 25.0
    bb_period: int = 20
    bb_std: float = 2.5
    # Strategy B
    hurst_min: float = 0.55
    ema_fast: int = 5
    ema_slow: int = 20
    momentum_min_move: float = 0.10  # e.g. 0.30 → 0.40+
    whale_volume_ratio: float = 2.0
    # Strategy C
    book_imbalance_min: float = 3.0
    book_price_lo: float = 0.20
    book_price_hi: float = 0.80
    # Exits
    take_profit_pct: float = 0.20  # +20% price move default
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    stall_hours: float = 48.0
    use_bb_take_profit: bool = True
    max_open_positions: int = 15
    cooldown_sec: float = 300.0
    history_len: int = 200


@dataclass
class TokenBar:
    """One mid-price sample (+ optional book / volume context)."""

    ts: float
    mid: float
    high: float | None = None
    low: float | None = None
    volume: float = 0.0
    whale_ratio: float | None = None
    bid_depth: float | None = None
    ask_depth: float | None = None
    liquidity_usd: float | None = None


@dataclass
class SwingIndicators:
    rsi: float | None = None
    bb_mid: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    bb_std: float | None = None
    ema_fast: float | None = None
    ema_slow: float | None = None
    hurst: float | None = None
    atr: float | None = None


@dataclass
class SwingSignal:
    strategy: StrategyName
    token_id: str
    condition_id: str
    ts: float
    entry_price: float
    target_price: float
    stop_price: float
    reason: str
    indicators: dict[str, Any] = field(default_factory=dict)


@dataclass
class SwingPosition:
    position_id: str
    strategy: StrategyName
    token_id: str
    condition_id: str
    entry_ts: float
    entry_price: float
    shares: float
    notional: float
    target_price: float
    stop_price: float
    trailing_stop: float
    mark_price: float
    status: str = "open"  # open | take_profit | stop_loss | time_decay | resolved
    exit_ts: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    realized_pnl: float | None = None
    reason: str = ""

    @property
    def unrealized_pnl(self) -> float:
        if self.status != "open":
            return 0.0
        return (self.mark_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.mark_price - self.entry_price) / self.entry_price

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["unrealized_pnl"] = self.unrealized_pnl
        d["return_pct"] = self.return_pct
        return d


class TokenSeries:
    """Rolling mid-price / book history for indicator computation."""

    def __init__(self, maxlen: int = 200) -> None:
        self.bars: Deque[TokenBar] = deque(maxlen=maxlen)

    def add(self, bar: TokenBar) -> None:
        if self.bars and bar.high is None:
            prev = self.bars[-1].mid
            bar.high = max(prev, bar.mid)
            bar.low = min(prev, bar.mid)
        elif bar.high is None:
            bar.high = bar.mid
            bar.low = bar.mid
        self.bars.append(bar)

    @property
    def closes(self) -> list[float]:
        return [b.mid for b in self.bars]

    def indicators(self, cfg: SwingConfig) -> SwingIndicators:
        closes = self.closes
        highs = [b.high if b.high is not None else b.mid for b in self.bars]
        lows = [b.low if b.low is not None else b.mid for b in self.bars]
        mid, up, lo, std = compute_bollinger(closes, cfg.bb_period, cfg.bb_std)
        return SwingIndicators(
            rsi=compute_rsi(closes, cfg.rsi_period),
            bb_mid=mid,
            bb_upper=up,
            bb_lower=lo,
            bb_std=std,
            ema_fast=_ema(closes, cfg.ema_fast),
            ema_slow=_ema(closes, cfg.ema_slow),
            hurst=compute_hurst(closes),
            atr=compute_atr(highs, lows, closes, cfg.atr_period),
        )


def evaluate_mean_reversion(
    token_id: str,
    condition_id: str,
    bar: TokenBar,
    ind: SwingIndicators,
    series: TokenSeries,
    cfg: SwingConfig,
) -> SwingSignal | None:
    """Strategy A: RSI / Bollinger oversold bounce toward EMA/BB mid."""
    liq = bar.liquidity_usd if bar.liquidity_usd is not None else cfg.min_liquidity_usd
    if liq < cfg.min_liquidity_usd:
        return None
    oversold_rsi = ind.rsi is not None and ind.rsi < cfg.rsi_oversold
    below_band = (
        ind.bb_lower is not None and bar.mid < ind.bb_lower and ind.bb_lower > 0
    )
    if not (oversold_rsi or below_band):
        return None
    target = ind.bb_mid if ind.bb_mid is not None else bar.mid * (1.0 + cfg.take_profit_pct)
    # Prefer 20-EMA when available as mid-line exit
    ema20 = _ema(series.closes, cfg.bb_period)
    if ema20 is not None:
        target = max(target, ema20)
    atr = ind.atr if ind.atr is not None else bar.mid * 0.03
    stop = max(bar.mid - cfg.atr_stop_mult * atr, 0.01)
    if target <= bar.mid:
        target = min(bar.mid * (1.0 + cfg.take_profit_pct), 0.99)
    return SwingSignal(
        strategy="mean_reversion",
        token_id=token_id,
        condition_id=condition_id,
        ts=bar.ts,
        entry_price=bar.mid,
        target_price=min(target, 0.99),
        stop_price=stop,
        reason=(
            f"oversold rsi={ind.rsi:.1f}" if oversold_rsi and ind.rsi is not None
            else "price below lower Bollinger"
        ),
        indicators={
            "rsi": ind.rsi,
            "bb_mid": ind.bb_mid,
            "bb_lower": ind.bb_lower,
            "liquidity_usd": liq,
        },
    )


def evaluate_momentum(
    token_id: str,
    condition_id: str,
    bar: TokenBar,
    ind: SwingIndicators,
    series: TokenSeries,
    cfg: SwingConfig,
) -> SwingSignal | None:
    """Strategy B: Hurst trend + EMA cross + structural re-rate on whale volume."""
    if ind.hurst is None or ind.hurst < cfg.hurst_min:
        return None
    if ind.ema_fast is None or ind.ema_slow is None:
        return None
    if ind.ema_fast <= ind.ema_slow:
        return None
    # Prior EMA cross: fast was below slow recently
    if len(series.closes) < cfg.ema_slow + 2:
        return None
    prev_fast = _ema(series.closes[:-1], cfg.ema_fast)
    prev_slow = _ema(series.closes[:-1], cfg.ema_slow)
    crossed = (
        prev_fast is not None
        and prev_slow is not None
        and prev_fast <= prev_slow
        and ind.ema_fast > ind.ema_slow
    )
    # Structural move over lookback window
    lookback = series.closes[-(cfg.ema_slow) :]
    move = lookback[-1] - lookback[0] if lookback else 0.0
    whale_ok = (
        bar.whale_ratio is not None and bar.whale_ratio >= cfg.whale_volume_ratio
    ) or bar.volume > 0
    if not (crossed or (move >= cfg.momentum_min_move and whale_ok)):
        return None
    if move < cfg.momentum_min_move * 0.5 and not crossed:
        return None

    atr = ind.atr if ind.atr is not None else bar.mid * 0.03
    target = min(bar.mid * (1.0 + cfg.take_profit_pct), 0.99)
    if cfg.use_bb_take_profit and ind.bb_upper is not None:
        target = max(target, min(ind.bb_upper, 0.99))
    stop = max(bar.mid - cfg.atr_stop_mult * atr, 0.01)
    return SwingSignal(
        strategy="momentum",
        token_id=token_id,
        condition_id=condition_id,
        ts=bar.ts,
        entry_price=bar.mid,
        target_price=target,
        stop_price=stop,
        reason=(
            f"hurst={ind.hurst:.2f} ema_cross={crossed} move={move:+.3f} "
            f"whale={bar.whale_ratio}"
        ),
        indicators={
            "hurst": ind.hurst,
            "ema_fast": ind.ema_fast,
            "ema_slow": ind.ema_slow,
            "move": move,
            "whale_ratio": bar.whale_ratio,
        },
    )


def evaluate_book_imbalance(
    token_id: str,
    condition_id: str,
    bar: TokenBar,
    ind: SwingIndicators,
    cfg: SwingConfig,
) -> SwingSignal | None:
    """Strategy C: bid depth >> ask depth in mid-range prices."""
    if bar.bid_depth is None or bar.ask_depth is None:
        return None
    if bar.ask_depth <= 0:
        return None
    if not (cfg.book_price_lo <= bar.mid <= cfg.book_price_hi):
        return None
    imbalance = bar.bid_depth / bar.ask_depth
    if imbalance < cfg.book_imbalance_min:
        return None
    atr = ind.atr if ind.atr is not None else bar.mid * 0.03
    target = min(bar.mid * (1.0 + cfg.take_profit_pct), 0.99)
    if cfg.use_bb_take_profit and ind.bb_upper is not None:
        target = max(target, min(ind.bb_upper, 0.99))
    stop = max(bar.mid - cfg.atr_stop_mult * atr, 0.01)
    return SwingSignal(
        strategy="book_imbalance",
        token_id=token_id,
        condition_id=condition_id,
        ts=bar.ts,
        entry_price=bar.mid,
        target_price=target,
        stop_price=stop,
        reason=f"bid/ask depth imbalance={imbalance:.2f}x",
        indicators={
            "bid_depth": bar.bid_depth,
            "ask_depth": bar.ask_depth,
            "imbalance": imbalance,
        },
    )


@dataclass
class SwingTrader:
    """Swing engine: indicators → signals → managed positions."""

    config: SwingConfig = field(default_factory=SwingConfig)
    journal_path: Path | None = None
    cash: float = 0.0
    realized_pnl: float = 0.0
    ticks_seen: int = 0
    signals_fired: int = 0
    fills: int = 0
    positions: list[SwingPosition] = field(default_factory=list)
    signals: list[SwingSignal] = field(default_factory=list)
    _series: dict[str, TokenSeries] = field(default_factory=dict)
    _conditions: dict[str, str] = field(default_factory=dict)
    _last_entry_ts: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash <= 0:
            self.cash = float(self.config.bankroll)

    @property
    def open_positions(self) -> list[SwingPosition]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def equity(self) -> float:
        return self.cash + sum(p.mark_price * p.shares for p in self.open_positions)

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.open_positions)

    def _series_for(self, token_id: str) -> TokenSeries:
        if token_id not in self._series:
            self._series[token_id] = TokenSeries(maxlen=self.config.history_len)
        return self._series[token_id]

    def on_bar(
        self,
        token_id: str,
        condition_id: str,
        bar: TokenBar,
        *,
        strategies: Sequence[StrategyName] | None = None,
    ) -> SwingPosition | None:
        """Ingest one mid sample; manage exits; maybe open a swing."""
        self.ticks_seen += 1
        self._conditions[token_id] = condition_id
        series = self._series_for(token_id)
        series.add(bar)
        ind = series.indicators(self.config)

        # Update marks + exits first
        for pos in list(self.open_positions):
            if pos.token_id == token_id:
                pos.mark_price = bar.mid
                self._update_trailing_stop(pos, ind)
        self._check_exits(now=bar.ts)

        if len(self.open_positions) >= self.config.max_open_positions:
            return None
        if any(p.token_id == token_id for p in self.open_positions):
            return None
        last = self._last_entry_ts.get(token_id, 0.0)
        if bar.ts - last < self.config.cooldown_sec:
            return None

        enabled = set(strategies) if strategies else {
            "mean_reversion",
            "momentum",
            "book_imbalance",
        }
        candidates: list[SwingSignal] = []
        if "mean_reversion" in enabled:
            s = evaluate_mean_reversion(
                token_id, condition_id, bar, ind, series, self.config
            )
            if s:
                candidates.append(s)
        if "momentum" in enabled:
            s = evaluate_momentum(
                token_id, condition_id, bar, ind, series, self.config
            )
            if s:
                candidates.append(s)
        if "book_imbalance" in enabled:
            s = evaluate_book_imbalance(
                token_id, condition_id, bar, ind, self.config
            )
            if s:
                candidates.append(s)

        if not candidates:
            return None

        # Prefer largest expected move to target
        sig = max(candidates, key=lambda x: x.target_price - x.entry_price)
        edge_pp = 100.0 * (sig.target_price - sig.entry_price)
        if edge_pp < self.config.min_ev_pct:
            return None

        self.signals_fired += 1
        self.signals.append(sig)
        return self._open_from_signal(sig)

    def _open_from_signal(self, sig: SwingSignal) -> SwingPosition | None:
        budget = self.equity * self.config.position_pct
        if budget < 1.0 or budget > self.cash or sig.entry_price <= 0:
            return None
        shares = budget / sig.entry_price
        cost = shares * sig.entry_price
        self.cash -= cost
        pos = SwingPosition(
            position_id=str(uuid.uuid4())[:8],
            strategy=sig.strategy,
            token_id=sig.token_id,
            condition_id=sig.condition_id,
            entry_ts=sig.ts,
            entry_price=sig.entry_price,
            shares=shares,
            notional=cost,
            target_price=sig.target_price,
            stop_price=sig.stop_price,
            trailing_stop=sig.stop_price,
            mark_price=sig.entry_price,
            reason=sig.reason,
        )
        self.positions.append(pos)
        self._last_entry_ts[sig.token_id] = sig.ts
        self.fills += 1
        self.persist()
        return pos

    def _update_trailing_stop(self, pos: SwingPosition, ind: SwingIndicators) -> None:
        atr = ind.atr if ind.atr is not None else pos.entry_price * 0.03
        trail = pos.mark_price - self.config.atr_stop_mult * atr
        pos.trailing_stop = max(pos.trailing_stop, trail, pos.stop_price)

    def _check_exits(self, *, now: float) -> None:
        stall_sec = self.config.stall_hours * 3600.0
        for pos in list(self.open_positions):
            # Take profit
            if pos.mark_price >= pos.target_price:
                self.close_position(
                    pos.position_id,
                    exit_price=pos.mark_price,
                    exit_ts=now,
                    reason="take_profit",
                )
                continue
            # Stop / trailing stop
            stop = max(pos.stop_price, pos.trailing_stop)
            if pos.mark_price <= stop:
                self.close_position(
                    pos.position_id,
                    exit_price=pos.mark_price,
                    exit_ts=now,
                    reason="stop_loss",
                )
                continue
            # Time decay / stall
            if now - pos.entry_ts >= stall_sec:
                self.close_position(
                    pos.position_id,
                    exit_price=pos.mark_price,
                    exit_ts=now,
                    reason="time_decay",
                )

    def close_position(
        self,
        position_id: str,
        *,
        exit_price: float,
        exit_ts: float | None = None,
        reason: str = "manual",
    ) -> SwingPosition | None:
        for pos in self.positions:
            if pos.position_id != position_id or pos.status != "open":
                continue
            px = min(max(float(exit_price), 0.0), 1.0)
            proceeds = px * pos.shares
            pnl = proceeds - pos.notional
            self.cash += proceeds
            self.realized_pnl += pnl
            pos.status = reason if reason in {
                "take_profit",
                "stop_loss",
                "time_decay",
                "resolved",
            } else "resolved"
            pos.exit_reason = reason
            pos.exit_ts = exit_ts if exit_ts is not None else time.time()
            pos.exit_price = px
            pos.realized_pnl = pnl
            pos.mark_price = px
            self.persist()
            return pos
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "updated_at": utc_now_iso(),
            "bankroll_initial": self.config.bankroll,
            "cash": self.cash,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "ticks_seen": self.ticks_seen,
            "signals_fired": self.signals_fired,
            "fills": self.fills,
            "min_ev_pct": self.config.min_ev_pct,
            "config": asdict(self.config),
            "open_positions": [p.to_dict() for p in self.open_positions],
            "closed_positions": [
                p.to_dict() for p in self.positions if p.status != "open"
            ],
            "recent_signals": [asdict(s) for s in self.signals[-20:]],
        }

    def persist(self) -> None:
        if self.journal_path is None:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.write_text(
            json.dumps(self.snapshot(), indent=2, default=str),
            encoding="utf-8",
        )


def inject_swing_demo_bars(
    *,
    n: int = 80,
    token_id: str = "swing_demo",
    condition_id: str = "swing_cond",
) -> list[tuple[str, str, TokenBar]]:
    """
    Synthetic path: grind down into oversold, then bounce (mean-reversion demo),
    with a later momentum leg and book imbalance spike.
    """
    now = time.time()
    out: list[tuple[str, str, TokenBar]] = []
    # Phase 1: decline into oversold (~40 bars from 0.55 → 0.28)
    price = 0.55
    for i in range(40):
        price = max(0.22, price - 0.007)
        out.append(
            (
                token_id,
                condition_id,
                TokenBar(
                    ts=now + i * 60.0,
                    mid=price,
                    volume=1_000.0,
                    whale_ratio=1.0,
                    bid_depth=10_000.0,
                    ask_depth=12_000.0,
                    liquidity_usd=75_000.0,
                ),
            )
        )
    # Phase 2: bounce (mean reversion toward mid)
    for i in range(20):
        price = min(0.50, price + 0.008)
        out.append(
            (
                token_id,
                condition_id,
                TokenBar(
                    ts=now + (40 + i) * 60.0,
                    mid=price,
                    volume=2_000.0,
                    whale_ratio=2.5,
                    bid_depth=30_000.0,
                    ask_depth=8_000.0,
                    liquidity_usd=80_000.0,
                ),
            )
        )
    # Phase 3: momentum drift 0.35→0.50 with whale + imbalance
    price = 0.35
    for i in range(max(n - 60, 10)):
        price = min(0.55, price + 0.012)
        out.append(
            (
                token_id,
                condition_id,
                TokenBar(
                    ts=now + (60 + i) * 60.0,
                    mid=price,
                    volume=5_000.0,
                    whale_ratio=3.5,
                    bid_depth=50_000.0,
                    ask_depth=10_000.0,
                    liquidity_usd=100_000.0,
                ),
            )
        )
    return out


def format_swing_dashboard(
    trader: SwingTrader,
    *,
    feed_status: str = "idle",
    extra_lines: Sequence[str] | None = None,
) -> str:
    lines = [
        "=" * 72,
        " POLYMARKET SWING TRADER  (simulated — no live orders)",
        "=" * 72,
        (
            f" feed={feed_status}  ticks={trader.ticks_seen}  "
            f"signals={trader.signals_fired}  fills={trader.fills}"
        ),
        (
            f" cash=${trader.cash:,.2f}  equity=${trader.equity:,.2f}  "
            f"realized=${trader.realized_pnl:,.2f}  "
            f"unrealized=${trader.unrealized_pnl:,.2f}"
        ),
        f" min_ev>={trader.config.min_ev_pct:.1f}pp  "
        f"TP={trader.config.take_profit_pct:.0%}  "
        f"stall={trader.config.stall_hours:g}h",
        "-" * 72,
        f" open swings ({len(trader.open_positions)}):",
    ]
    if not trader.open_positions:
        lines.append("   (none)")
    else:
        for p in trader.open_positions[-8:]:
            lines.append(
                f"   [{p.position_id}] {p.strategy} {p.token_id[:10]}…  "
                f"entry={p.entry_price:.3f}  mark={p.mark_price:.3f}  "
                f"tp={p.target_price:.3f}  trail={p.trailing_stop:.3f}  "
                f"uPnL=${p.unrealized_pnl:+.2f}"
            )
    if trader.signals:
        s = trader.signals[-1]
        lines.append(
            f" last signal: {s.strategy} @ {s.entry_price:.3f} → {s.target_price:.3f} "
            f"({s.reason[:40]})"
        )
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("=" * 72)
    return "\n".join(lines)


def format_swing_summary(journal: Mapping[str, Any] | Path | str) -> str:
    if isinstance(journal, (str, Path)):
        path = Path(journal)
        if not path.exists():
            return f"No swing journal found at {path}"
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = dict(journal)
        path = None
    opens = list(raw.get("open_positions") or [])
    closed = list(raw.get("closed_positions") or [])
    lines = [
        "=" * 72,
        " SWING TRADE SUMMARY",
        f" journal: {path}" if path else " journal: (memory)",
        f" updated: {raw.get('updated_at', 'n/a')}",
        "=" * 72,
        (
            f" cash=${float(raw.get('cash') or 0):,.2f}  "
            f"equity=${float(raw.get('equity') or 0):,.2f}  "
            f"realized=${float(raw.get('realized_pnl') or 0):,.2f}  "
            f"unrealized=${float(raw.get('unrealized_pnl') or 0):,.2f}"
        ),
        f" fills={raw.get('fills', 0)}  signals={raw.get('signals_fired', 0)}",
        "-" * 72,
        f" open ({len(opens)}):",
    ]
    if not opens:
        lines.append("   (none)")
    else:
        for p in opens:
            lines.append(
                f"   [{p.get('position_id')}] {p.get('strategy')}  "
                f"entry={float(p.get('entry_price') or 0):.3f}  "
                f"mark={float(p.get('mark_price') or 0):.3f}  "
                f"trail={float(p.get('trailing_stop') or 0):.3f}  "
                f"uPnL=${float(p.get('unrealized_pnl') or 0):+.2f}"
            )
    lines.append(f" closed ({len(closed)}):")
    if not closed:
        lines.append("   (none)")
    else:
        for p in closed[-12:]:
            lines.append(
                f"   [{p.get('position_id')}] {p.get('strategy')}  "
                f"{p.get('exit_reason')}  "
                f"PnL=${float(p.get('realized_pnl') or 0):+.2f}"
            )
    lines.append("=" * 72)
    return "\n".join(lines)
