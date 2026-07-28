"""CLI entrypoints: ingest, features, edges, backtest, paper-trade, swing-trade, status."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from polymarket_analytics.backtest import (
    StrategyParams,
    format_oos_comparison_table,
    run_backtest,
    run_find_edges,
)
from polymarket_analytics.features import run_compute_features
from polymarket_analytics.ingest import DEFAULT_CHUNK_ROWS, run_ingest
from polymarket_analytics.store import bootstrap_warehouse, warehouse_status

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_PARQUET = ROOT / "data" / "parquet"
DEFAULT_WAREHOUSE = ROOT / "data" / "warehouse.duckdb"
DEFAULT_OOS_REPORT = ROOT / "data" / "oos_edge_report.json"
DEFAULT_PAPER_JOURNAL = ROOT / "data" / "paper_trades.json"
DEFAULT_SWING_JOURNAL = ROOT / "data" / "swing_trades.json"
FIXTURES = ROOT / "fixtures"


def _copy_fixtures(raw_dir: Path) -> None:
    for kind, name in (
        ("trades", "sample_trades.json"),
        ("markets", "sample_markets.json"),
    ):
        src = FIXTURES / name
        if not src.exists():
            raise FileNotFoundError(f"Missing fixture: {src}")
        dst_dir = raw_dir / kind
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / name)


def _cmd_ingest(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    warehouse = Path(args.warehouse)

    if args.with_fixtures:
        _copy_fixtures(raw_dir)

    stats = run_ingest(
        raw_dir,
        out_dir,
        bootstrap_warehouse=True,
        warehouse_path=warehouse,
        chunk_rows=args.chunk_rows,
    )
    print(
        json.dumps(
            {"ok": True, "ingested": stats, "warehouse": str(warehouse)},
            indent=2,
        )
    )
    return 0


def _cmd_compute_features(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    warehouse = Path(args.warehouse)
    result = run_compute_features(
        out_dir,
        warehouse_path=warehouse,
        bootstrap_warehouse=True,
    )
    print(
        json.dumps(
            {"ok": True, "features": result, "warehouse": str(warehouse)},
            indent=2,
        )
    )
    return 0


def _cmd_find_edges(args: argparse.Namespace) -> int:
    warehouse = Path(args.warehouse)
    parquet_dir = Path(args.out_dir)
    buckets = args.bucket if args.bucket else None
    result = run_find_edges(
        warehouse,
        parquet_dir,
        min_samples=args.min_samples,
        min_ev_pct=args.min_ev_pct,
        price_buckets=buckets,
        min_whale_ratio=args.min_whale_ratio,
        max_ttr_hours=args.max_ttr_hours,
        top_k=args.top,
        start=args.start,
        end=args.end,
        split_date=args.split_date,
        oos_report_path=Path(args.oos_report) if args.split_date else None,
    )

    if isinstance(result, dict):
        # OOS mode
        print(
            format_oos_comparison_table(result.get("comparisons", [])),
            flush=True,
        )
        print(flush=True)
        payload = {
            "ok": True,
            "mode": "oos",
            "split_date": result.get("split_date"),
            "start": result.get("start") or args.start,
            "end": result.get("end") or args.end,
            "train_rows": result.get("train_rows"),
            "test_rows": result.get("test_rows"),
            "n_is_edges": result.get("n_is_edges"),
            "report_path": result.get("report_path"),
            "comparisons": result.get("comparisons"),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    payload = {
        "ok": True,
        "mode": "full",
        "min_samples": args.min_samples,
        "min_whale_ratio": args.min_whale_ratio,
        "max_ttr_hours": args.max_ttr_hours,
        "start": args.start,
        "end": args.end,
        "n_edges": result.height,
        "edges": result.to_dicts(),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    warehouse = Path(args.warehouse)
    parquet_dir = Path(args.out_dir)
    params = StrategyParams(
        price_bucket=args.bucket,
        min_volume_spike=args.min_volume_spike,
        min_whale_ratio=args.min_whale_ratio,
        momentum_1h=args.momentum_1h,
        momentum_6h=args.momentum_6h,
        max_time_to_resolution_hours=args.max_ttr_hours,
        side=args.side,
    )
    result = run_backtest(warehouse, params, parquet_dir)
    print(
        json.dumps(
            {"ok": True, "backtest": result.to_dict(include_equity=args.equity)},
            indent=2,
            default=str,
        )
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    warehouse = Path(args.warehouse)
    parquet_dir = Path(args.out_dir)
    if not warehouse.exists():
        bootstrap_warehouse(parquet_dir, warehouse)
    status = warehouse_status(warehouse, parquet_dir=parquet_dir)
    print(json.dumps(status, indent=2, default=str))
    return 0


def _cmd_paper_trade(args: argparse.Namespace) -> int:
    """Live (or demo) paper trading dashboard — never submits real orders."""
    from polymarket_analytics.live_feed import (
        LiveFeatureEngine,
        MarketWebSocketFeed,
        fetch_active_token_ids,
        inject_demo_ticks,
    )
    from polymarket_analytics.paper_trader import (
        PaperConfig,
        PaperTrader,
        format_dashboard,
        format_journal_summary,
    )

    if getattr(args, "summary", False):
        print(format_journal_summary(Path(args.journal)))
        return 0

    if getattr(args, "compare", False):
        from polymarket_analytics.profiles import (
            format_profile_leaderboard,
            load_profile_journals,
        )

        rows = load_profile_journals(Path(args.journal).parent if args.journal else None)
        print(format_profile_leaderboard(rows))
        print(json.dumps({"ok": True, "mode": "compare", "profiles": rows}, indent=2, default=str))
        return 0

    if getattr(args, "multi_profile", False):
        return _cmd_multi_profile_paper(args)

    # --min-ev 0.10 means 10 percentage points (matches OOS report units)
    min_ev_pp = float(args.min_ev)
    if min_ev_pp <= 1.0:
        min_ev_pp *= 100.0

    cfg = PaperConfig(
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position_pct,
        fee_bps=args.fee_bps,
        fee_category=args.fee_category,
        fee_rate=args.fee_rate,
        use_dynamic_fees=not args.flat_fees,
        spread_slippage_bps=args.spread_slippage_bps,
        min_oos_ev_pct=min_ev_pp,
        require_persists=not args.allow_decayed,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        resolve_poll_sec=args.resolve_poll_sec,
    )
    trader = PaperTrader.from_oos_report(
        Path(args.oos_report),
        config=cfg,
        journal_path=Path(args.journal),
    )
    trader.persist()

    engine = LiveFeatureEngine()
    last_feat = None
    feed_status = "demo" if args.demo else "connecting"

    def _render(status: str) -> None:
        # Clear-ish refresh without depending on curses
        print("\033[2J\033[H", end="", flush=True)
        print(
            format_dashboard(
                trader,
                feed_status=status,
                last_feat=last_feat,
                extra_lines=[
                    f" journal={args.journal}",
                    f" min_oos_ev>={min_ev_pp:.1f}pp  duration={args.duration}s",
                    " Ctrl+C to stop (paper only)",
                ],
            ),
            flush=True,
        )

    if args.demo:
        feats = inject_demo_ticks(engine, n=args.demo_ticks)
        for feat in feats:
            last_feat = feat
            trader.on_features(feat)
            _render("demo")
            time.sleep(args.demo_sleep)
        trader.persist()
        print(json.dumps({"ok": True, "mode": "demo", **trader.snapshot()}, indent=2, default=str))
        return 0

    async def _run_live() -> int:
        nonlocal last_feat, feed_status
        token_ids = list(args.token_id) if args.token_id else []
        if not token_ids:
            try:
                token_ids = fetch_active_token_ids(limit=args.n_markets)
            except Exception as exc:
                print(f"Failed to fetch active markets: {exc}", file=sys.stderr)
                print("Hint: pass --token-id … or use --demo", file=sys.stderr)
                return 1
        if not token_ids:
            print("No token ids to subscribe; use --demo or --token-id", file=sys.stderr)
            return 1

        feed = MarketWebSocketFeed(
            token_ids,
            feature_engine=engine,
            on_resolution=lambda ev: trader.on_market_resolved(
                condition_id=ev.condition_id,
                winning_asset_id=ev.winning_asset_id,
                exit_ts=ev.ts,
            ),
        )
        feed_status = f"live:{len(token_ids)} tokens"
        _render(feed_status)
        end = time.time() + float(args.duration)

        try:
            async for feat in feed.stream():
                last_feat = feat
                trader.on_features(feat)
                err = f" err={feed.last_error}" if feed.last_error else ""
                status = f"live:{len(token_ids)} tokens ticks={trader.ticks_seen}{err}"
                if trader.ticks_seen % max(1, args.refresh_every) == 0:
                    _render(status)
                if time.time() >= end:
                    feed.stop()
                    break
            # If the generator ends before duration (shouldn't with reconnect), wait out clock
            while time.time() < end and not feed._stop.is_set():  # noqa: SLF001
                feed_status = f"reconnecting… last_err={feed.last_error}"
                _render(feed_status)
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            feed.stop()
        except Exception as exc:
            feed_status = f"error:{exc}"
            _render(feed_status)
            trader.persist()
            print(json.dumps({"ok": False, "error": str(exc), **trader.snapshot()}, indent=2, default=str))
            return 1

        feed.stop()
        trader.persist()
        _render("stopped")
        print(json.dumps({"ok": True, "mode": "live", **trader.snapshot()}, indent=2, default=str))
        return 0

    return int(asyncio.run(_run_live()))


def _cmd_multi_profile_paper(args: argparse.Namespace) -> int:
    """Forward-test all incubator profiles on demo bars or live ticks."""
    from polymarket_analytics.live_feed import (
        LiveFeatureEngine,
        MarketWebSocketFeed,
        TopOfBook,
        fetch_active_token_ids,
        inject_demo_ticks,
    )
    from polymarket_analytics.profiles import (
        MultiProfileIncubator,
        format_profile_leaderboard,
    )
    from polymarket_analytics.swing_trader import TokenBar, inject_swing_demo_bars

    min_ev_pp = float(args.min_ev)
    if min_ev_pp <= 1.0:
        min_ev_pp *= 100.0

    data_dir = Path(args.journal).parent if args.journal else ROOT / "data"
    inc = MultiProfileIncubator.create(
        bankroll=args.bankroll,
        min_ev_pct=min_ev_pp,
        data_dir=data_dir,
    )

    if args.demo:
        # Shared synthetic path covering whale mid-bucket + swing setups
        engine = LiveFeatureEngine()
        for feat in inject_demo_ticks(engine, n=max(args.demo_ticks, 40), base_price=0.45):
            # Attach synthetic book for confluence / imbalance
            feat = type(feat)(
                **{
                    **feat.__dict__,
                    "best_bid": feat.price - 0.01,
                    "best_ask": feat.price + 0.005,
                }
            )
            inc.on_features(feat)
            time.sleep(args.demo_sleep)
        for token_id, condition_id, bar in inject_swing_demo_bars(n=max(args.demo_ticks, 60)):
            inc.on_bar(token_id, condition_id, bar)
            time.sleep(args.demo_sleep)
        inc.persist_all()
        rows = inc.comparison_rows()
        print(format_profile_leaderboard(rows))
        print(json.dumps({"ok": True, "mode": "multi-demo", "profiles": rows}, indent=2, default=str))
        return 0

    async def _run() -> int:
        token_ids = list(args.token_id) if args.token_id else []
        if not token_ids:
            try:
                token_ids = fetch_active_token_ids(limit=args.n_markets)
            except Exception as exc:
                print(f"Failed to fetch markets: {exc}", file=sys.stderr)
                return 1
        engine = LiveFeatureEngine()
        books: dict[str, TopOfBook] = {}

        def _on_feat(feat) -> None:
            book = books.get(feat.token_id)
            if book is not None:
                feat = type(feat)(
                    **{
                        **feat.__dict__,
                        "best_bid": book.best_bid,
                        "best_ask": book.best_ask,
                    }
                )
            inc.on_features(feat)

        feed = MarketWebSocketFeed(token_ids, feature_engine=engine, on_features=_on_feat)
        _orig = feed.handle_message

        def _wrapped(raw: str | bytes):
            text = raw.decode() if isinstance(raw, bytes) else str(raw)
            if text.strip() and text.strip().upper() not in {"PING", "PONG"}:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = None
                if payload is not None:
                    from polymarket_analytics.live_feed import parse_market_event

                    for item in parse_market_event(payload):
                        if isinstance(item, TopOfBook):
                            books[item.token_id] = item
            return _orig(raw)

        feed.handle_message = _wrapped  # type: ignore[method-assign]
        end = time.time() + float(args.duration)
        try:
            async for _ in feed.stream():
                if time.time() >= end:
                    feed.stop()
                    break
        except KeyboardInterrupt:
            feed.stop()
        feed.stop()
        inc.persist_all()
        rows = inc.comparison_rows()
        print(format_profile_leaderboard(rows))
        print(json.dumps({"ok": True, "mode": "multi-live", "profiles": rows}, indent=2, default=str))
        return 0

    return int(asyncio.run(_run()))


def _cmd_swing_trade(args: argparse.Namespace) -> int:
    """Swing trading dashboard / demo — simulated fills only."""
    from polymarket_analytics.live_feed import (
        LiveFeatureEngine,
        MarketWebSocketFeed,
        TopOfBook,
        fetch_active_token_ids,
    )
    from polymarket_analytics.swing_trader import (
        SwingConfig,
        SwingTrader,
        TokenBar,
        format_swing_dashboard,
        format_swing_summary,
        inject_swing_demo_bars,
    )

    if getattr(args, "summary", False):
        print(format_swing_summary(Path(args.journal)))
        return 0

    min_ev_pp = float(args.min_ev)
    if min_ev_pp <= 1.0:
        min_ev_pp *= 100.0

    cfg = SwingConfig(
        bankroll=args.bankroll,
        position_pct=args.position_pct,
        min_ev_pct=min_ev_pp,
        take_profit_pct=args.take_profit_pct,
        stall_hours=args.stall_hours,
        atr_stop_mult=args.atr_stop_mult,
        min_liquidity_usd=args.min_liquidity,
    )
    strategies = tuple(args.strategy) if args.strategy else None
    trader = SwingTrader(config=cfg, journal_path=Path(args.journal))
    trader.persist()

    def _render(status: str) -> None:
        print("\033[2J\033[H", end="", flush=True)
        print(
            format_swing_dashboard(
                trader,
                feed_status=status,
                extra_lines=[
                    f" journal={args.journal}",
                    f" min_ev>={min_ev_pp:.1f}pp  strategies={strategies or 'all'}",
                    " Ctrl+C to stop (paper only)",
                ],
            ),
            flush=True,
        )

    if args.demo:
        bars = inject_swing_demo_bars(n=args.demo_ticks)
        for token_id, condition_id, bar in bars:
            trader.on_bar(token_id, condition_id, bar, strategies=strategies)
            _render("demo")
            time.sleep(args.demo_sleep)
        trader.persist()
        print(
            json.dumps(
                {"ok": True, "mode": "demo", **trader.snapshot()},
                indent=2,
                default=str,
            )
        )
        return 0

    async def _run_live() -> int:
        token_ids = list(args.token_id) if args.token_id else []
        if not token_ids:
            try:
                token_ids = fetch_active_token_ids(limit=args.n_markets)
            except Exception as exc:
                print(f"Failed to fetch markets: {exc}", file=sys.stderr)
                return 1
        if not token_ids:
            print("No tokens; use --demo or --token-id", file=sys.stderr)
            return 1

        engine = LiveFeatureEngine()
        books: dict[str, TopOfBook] = {}

        def _on_feat(feat) -> None:
            book = books.get(feat.token_id)
            bid_depth = ask_depth = None
            if book and book.best_bid is not None and book.best_ask is not None:
                mid = feat.price
                bid_depth = max(mid - book.best_bid, 1e-4) ** -1
                ask_depth = max(book.best_ask - mid, 1e-4) ** -1
            bar = TokenBar(
                ts=feat.ts,
                mid=feat.price,
                volume=feat.size,
                whale_ratio=feat.whale_ratio,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                liquidity_usd=cfg.min_liquidity_usd,
            )
            trader.on_bar(
                feat.token_id,
                feat.condition_id,
                bar,
                strategies=strategies,
            )

        feed = MarketWebSocketFeed(token_ids, feature_engine=engine, on_features=_on_feat)
        _orig = feed.handle_message

        def _wrapped(raw: str | bytes):
            text = raw.decode() if isinstance(raw, bytes) else str(raw)
            if text.strip() and text.strip().upper() not in {"PONG", "PING"}:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = None
                if payload is not None:
                    from polymarket_analytics.live_feed import parse_market_event

                    for item in parse_market_event(payload):
                        if isinstance(item, TopOfBook):
                            books[item.token_id] = item
            return _orig(raw)

        feed.handle_message = _wrapped  # type: ignore[method-assign]
        _render(f"live:{len(token_ids)} tokens")
        end = time.time() + float(args.duration)
        try:
            async for _feat in feed.stream():
                if trader.ticks_seen % max(1, args.refresh_every) == 0:
                    _render(f"live ticks={trader.ticks_seen}")
                if time.time() >= end:
                    feed.stop()
                    break
        except KeyboardInterrupt:
            feed.stop()
        except Exception as exc:
            trader.persist()
            print(json.dumps({"ok": False, "error": str(exc), **trader.snapshot()}, indent=2))
            return 1
        feed.stop()
        trader.persist()
        _render("stopped")
        print(json.dumps({"ok": True, "mode": "live", **trader.snapshot()}, indent=2, default=str))
        return 0

    return int(asyncio.run(_run_live()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polymarket_analytics",
        description="Polymarket local analytics: ingest, features, edges, paper trading",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser(
        "ingest", help="Ingest raw trade/market files into Parquet + DuckDB"
    )
    ingest_p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ingest_p.add_argument("--out-dir", type=Path, default=DEFAULT_PARQUET)
    ingest_p.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    ingest_p.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help="Rows per streaming chunk (default: %(default)s)",
    )
    ingest_p.add_argument(
        "--with-fixtures",
        action="store_true",
        help="Copy sample fixtures into raw-dir before ingest",
    )
    ingest_p.set_defaults(func=_cmd_ingest)

    feat_p = sub.add_parser(
        "compute-features",
        help="Compute rolling/bucket/TTR features into Parquet + v_trade_features",
    )
    feat_p.add_argument("--out-dir", type=Path, default=DEFAULT_PARQUET)
    feat_p.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    feat_p.set_defaults(func=_cmd_compute_features)

    edges_p = sub.add_parser(
        "find-edges",
        help="Grid-search v_trade_features for historical win-rate edges",
    )
    edges_p.add_argument("--out-dir", type=Path, default=DEFAULT_PARQUET)
    edges_p.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    edges_p.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum N per slice (default: %(default)s)",
    )
    edges_p.add_argument(
        "--min-ev-pct",
        type=float,
        default=None,
        help="Optional minimum EV%% filter",
    )
    edges_p.add_argument(
        "--bucket",
        action="append",
        default=None,
        help="Restrict to price bucket(s); repeatable",
    )
    edges_p.add_argument(
        "--min-whale-ratio",
        type=float,
        default=None,
        help="Require whale_ratio > threshold (fixes that grid axis)",
    )
    edges_p.add_argument(
        "--max-ttr-hours",
        type=float,
        default=None,
        help="Require time_to_resolution_hours < bound (fixes that grid axis)",
    )
    edges_p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Include trades with traded_at >= YYYY-MM-DD",
    )
    edges_p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Include trades with traded_at < YYYY-MM-DD",
    )
    edges_p.add_argument(
        "--split-date",
        type=str,
        default=None,
        help="Train on trades before this date; evaluate top setups OOS after it",
    )
    edges_p.add_argument(
        "--oos-report",
        type=Path,
        default=DEFAULT_OOS_REPORT,
        help=f"Path for OOS JSON report (default: {DEFAULT_OOS_REPORT})",
    )
    edges_p.add_argument(
        "--top",
        type=int,
        default=50,
        help="Max rows to print / validate OOS (default: %(default)s)",
    )
    edges_p.set_defaults(func=_cmd_find_edges)

    bt_p = sub.add_parser(
        "backtest",
        help="Simulate a long-token strategy on v_trade_features",
    )
    bt_p.add_argument("--out-dir", type=Path, default=DEFAULT_PARQUET)
    bt_p.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    bt_p.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="price_bucket filter, e.g. 0.90-0.95",
    )
    bt_p.add_argument(
        "--min-volume-spike",
        type=float,
        default=None,
        help="Require volume_spike_1h_24h > threshold",
    )
    bt_p.add_argument(
        "--min-whale-ratio",
        type=float,
        default=None,
        help="Require whale_ratio > threshold",
    )
    bt_p.add_argument(
        "--momentum-1h",
        choices=("any", "pos", "neg"),
        default="any",
    )
    bt_p.add_argument(
        "--momentum-6h",
        choices=("any", "pos", "neg"),
        default="any",
    )
    bt_p.add_argument(
        "--max-ttr-hours",
        type=float,
        default=None,
        help="Require time_to_resolution_hours < bound",
    )
    bt_p.add_argument(
        "--side",
        type=str,
        default="BUY",
        help="Fill side filter (default: BUY); pass empty string to disable",
    )
    bt_p.add_argument(
        "--equity",
        action="store_true",
        help="Include equity curve in JSON output",
    )
    bt_p.set_defaults(func=_cmd_backtest)

    status_p = sub.add_parser("status", help="Print warehouse row counts and date range")
    status_p.add_argument("--out-dir", type=Path, default=DEFAULT_PARQUET)
    status_p.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    status_p.set_defaults(func=_cmd_status)

    paper_p = sub.add_parser(
        "paper-trade",
        help="Live WebSocket paper trading dashboard (simulated fills only)",
    )
    paper_p.add_argument(
        "--min-ev",
        type=float,
        default=0.10,
        help="Min OOS EV to trade: 0.10 → 10pp, or pass 10 for pp directly (default: %(default)s)",
    )
    paper_p.add_argument(
        "--oos-report",
        type=Path,
        default=DEFAULT_OOS_REPORT,
        help=f"OOS edge report (default: {DEFAULT_OOS_REPORT})",
    )
    paper_p.add_argument(
        "--journal",
        type=Path,
        default=DEFAULT_PAPER_JOURNAL,
        help=f"Paper trade journal JSON (default: {DEFAULT_PAPER_JOURNAL})",
    )
    paper_p.add_argument(
        "--summary",
        action="store_true",
        help="Print journal summary (open positions, fees, PnL) and exit",
    )
    paper_p.add_argument(
        "--compare",
        action="store_true",
        help="Print multi-profile forward-test leaderboard from journals",
    )
    paper_p.add_argument(
        "--multi-profile",
        action="store_true",
        help="Route ticks through all incubator profiles (isolated $10k bankrolls)",
    )
    paper_p.add_argument("--bankroll", type=float, default=10_000.0)
    paper_p.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier (default: %(default)s)",
    )
    paper_p.add_argument(
        "--max-position-pct",
        type=float,
        default=0.05,
        help="Cap fraction of equity per fill (default: %(default)s)",
    )
    paper_p.add_argument("--fee-bps", type=float, default=0.0,
                        help="Legacy flat fee in bps (only with --flat-fees)")
    paper_p.add_argument(
        "--fee-category",
        type=str,
        default="crypto",
        help="Polymarket fee category for dynamic curve (default: crypto → 0.07)",
    )
    paper_p.add_argument(
        "--fee-rate",
        type=float,
        default=None,
        help="Override dynamic feeRate constant (else category default)",
    )
    paper_p.add_argument(
        "--flat-fees",
        action="store_true",
        help="Use legacy flat --fee-bps instead of dynamic C·feeRate·p·(1-p)",
    )
    paper_p.add_argument(
        "--take-profit-pct",
        type=float,
        default=None,
        help="Optional early exit when mark return ≥ this fraction of entry",
    )
    paper_p.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Optional early exit when mark return ≤ -this fraction of entry",
    )
    paper_p.add_argument(
        "--resolve-poll-sec",
        type=float,
        default=60.0,
        help="Seconds between Gamma resolution polls (default: %(default)s)",
    )
    paper_p.add_argument(
        "--spread-slippage-bps",
        type=float,
        default=50.0,
        help="Adverse spread slippage in bps (default: %(default)s)",
    )
    paper_p.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Seconds to run live feed (default: %(default)s)",
    )
    paper_p.add_argument(
        "--n-markets",
        type=int,
        default=20,
        help="Active Gamma markets to subscribe when --token-id omitted",
    )
    paper_p.add_argument(
        "--token-id",
        action="append",
        default=None,
        help="CLOB asset/token id to subscribe; repeatable",
    )
    paper_p.add_argument(
        "--refresh-every",
        type=int,
        default=1,
        help="Redraw dashboard every N trade ticks (default: %(default)s)",
    )
    paper_p.add_argument(
        "--allow-decayed",
        action="store_true",
        help="Include OOS setups that did not persist (persists=false)",
    )
    paper_p.add_argument(
        "--demo",
        action="store_true",
        help="Run offline synthetic ticks (no WebSocket)",
    )
    paper_p.add_argument("--demo-ticks", type=int, default=40)
    paper_p.add_argument("--demo-sleep", type=float, default=0.05)
    paper_p.set_defaults(func=_cmd_paper_trade)

    swing_p = sub.add_parser(
        "swing-trade",
        help="Swing-trade probability moves (mean-reversion / momentum / book imbalance)",
    )
    swing_p.add_argument(
        "--min-ev",
        type=float,
        default=0.08,
        help="Min expected move to target in EV fraction (0.08→8pp) or pp",
    )
    swing_p.add_argument(
        "--journal",
        type=Path,
        default=DEFAULT_SWING_JOURNAL,
        help=f"Swing journal JSON (default: {DEFAULT_SWING_JOURNAL})",
    )
    swing_p.add_argument(
        "--summary",
        action="store_true",
        help="Print swing journal summary and exit",
    )
    swing_p.add_argument("--bankroll", type=float, default=10_000.0)
    swing_p.add_argument("--position-pct", type=float, default=0.05)
    swing_p.add_argument("--take-profit-pct", type=float, default=0.20)
    swing_p.add_argument(
        "--stall-hours",
        type=float,
        default=36.0,
        help="Time-exit hours (aligned with SwingConfig.stall_hours default)",
    )
    swing_p.add_argument("--atr-stop-mult", type=float, default=2.0)
    swing_p.add_argument("--min-liquidity", type=float, default=50_000.0)
    swing_p.add_argument(
        "--strategy",
        action="append",
        choices=("mean_reversion", "momentum", "book_imbalance"),
        default=None,
        help="Restrict strategies; repeatable (default: all)",
    )
    swing_p.add_argument("--demo", action="store_true", help="Offline synthetic bars")
    swing_p.add_argument("--demo-ticks", type=int, default=80)
    swing_p.add_argument("--demo-sleep", type=float, default=0.02)
    swing_p.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Live feed seconds (default: %(default)s)",
    )
    swing_p.add_argument("--n-markets", type=int, default=20)
    swing_p.add_argument("--token-id", action="append", default=None)
    swing_p.add_argument("--refresh-every", type=int, default=1)
    swing_p.set_defaults(func=_cmd_swing_trade)

    # --- Outcome research / L2 collection ---
    audit_p = sub.add_parser("audit-data", help="Audit duplicate trades and lake coverage")
    audit_p.add_argument("--data-root", type=Path, default=ROOT / "data")
    audit_p.add_argument("--source", choices=("lake", "hf_sample", "merged"), default="hf_sample")
    audit_p.set_defaults(func=_cmd_audit_data)

    lake_p = sub.add_parser(
        "build-outcome-lake",
        help="Build canonical deduped trades + features for outcome research",
    )
    lake_p.add_argument("--data-root", type=Path, default=ROOT / "data")
    lake_p.add_argument("--source", choices=("lake", "hf_sample", "merged"), default="hf_sample")
    lake_p.set_defaults(func=_cmd_build_outcome_lake)

    sweep_p = sub.add_parser(
        "sweep-outcomes",
        help="Purged walk-forward outcome strategy sweep (no microstructure claims)",
    )
    sweep_p.add_argument("--data-root", type=Path, default=ROOT / "data")
    sweep_p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "research")
    sweep_p.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Optional features parquet (default: data/curated/trade_features_canonical.parquet)",
    )
    sweep_p.add_argument("--train-end", type=str, default="2023-06-01T00:00:00+00:00")
    sweep_p.add_argument("--seed", type=int, default=42)
    sweep_p.set_defaults(func=_cmd_sweep_outcomes)

    books_p = sub.add_parser(
        "collect-books",
        help="Prospective L2 book collector smoke test / continuous capture",
    )
    books_p.add_argument("--duration", type=float, default=30.0, help="Seconds to collect")
    books_p.add_argument("--n-tokens", type=int, default=3)
    books_p.add_argument("--token-id", action="append", default=None)
    books_p.add_argument("--data-root", type=Path, default=ROOT / "data")
    books_p.set_defaults(func=_cmd_collect_books)

    backfill_p = sub.add_parser("backfill-year", help="Resumable year-sharded Gamma/Data API backfill")
    backfill_p.add_argument("--year", type=int, required=True)
    backfill_p.add_argument("--out-dir", type=Path, required=True)
    backfill_p.add_argument("--market-limit", type=int, default=None)
    backfill_p.add_argument("--max-markets", type=int, default=None)
    backfill_p.add_argument("--trade-limit-per-market", type=int, default=500)
    backfill_p.add_argument("--tiny", action="store_true", help="Collect at most 5 markets and 50 trades each")
    backfill_p.set_defaults(func=_cmd_backfill_year)

    merge_p = sub.add_parser("merge-expanded", help="Merge annual backfills into canonical outcome lake")
    merge_p.add_argument("--data-root", type=Path, default=ROOT / "data")
    merge_p.add_argument("--year-dir", type=Path, action="append", required=True)
    merge_p.add_argument("--existing-canonical", type=Path, default=None)
    merge_p.add_argument("--out-dir", type=Path, required=True)
    merge_p.set_defaults(func=_cmd_merge_expanded)

    gate_p = sub.add_parser("outcome-gate", help="Evaluate outcome research evidence gate")
    gate_p.add_argument("--features", type=Path, required=True)
    gate_p.add_argument("--min-events", type=int, default=100)
    gate_p.add_argument("--train-end", type=str, default="2025-01-01T00:00:00+00:00")
    gate_p.add_argument("--baseline-rows", type=int, default=None)
    gate_p.add_argument("--used-baseline-fallback", action="store_true")
    gate_p.set_defaults(func=_cmd_outcome_gate)

    report_p = sub.add_parser("overnight-report", help="Write morning report from research manifests")
    report_p.add_argument("--root", type=Path, default=ROOT)
    report_p.add_argument("--out", type=Path, required=True)
    report_p.add_argument("--manifest", type=Path, action="append", default=[],
                          help="JSON manifest; section is derived from its filename")
    report_p.set_defaults(func=_cmd_overnight_report)

    return parser


def _cmd_audit_data(args: argparse.Namespace) -> int:
    from polymarket_analytics.research.canonical_lake import load_source_trades
    from polymarket_analytics.research.duplicates import (
        audit_duplicate_trades,
        write_duplicate_audit_parquet,
        write_duplicate_detail_parquet,
    )

    data_root = Path(args.data_root)
    raw = load_source_trades(data_root, source=args.source)
    audit = audit_duplicate_trades(raw)
    quality = data_root / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    write_duplicate_audit_parquet(audit, quality / "duplicate_trade_audit.parquet")
    write_duplicate_detail_parquet(raw, quality / "duplicate_trade_detail.parquet")
    print(json.dumps({"ok": True, "source": args.source, "audit": audit}, indent=2, default=str))
    return 0


def _cmd_build_outcome_lake(args: argparse.Namespace) -> int:
    from polymarket_analytics.research.canonical_lake import build_canonical_lake

    stats = build_canonical_lake(Path(args.data_root), source=args.source)
    print(json.dumps({"ok": True, **stats}, indent=2, default=str))
    return 0


def _cmd_sweep_outcomes(args: argparse.Namespace) -> int:
    import polars as pl

    from polymarket_analytics.research.outcome_sweep import run_outcome_sweep

    feats_path = (
        Path(args.features)
        if getattr(args, "features", None)
        else Path(args.data_root) / "curated" / "trade_features_canonical.parquet"
    )
    if not feats_path.exists():
        print(json.dumps({"ok": False, "error": f"missing {feats_path}; run build-outcome-lake first"}))
        return 1
    features = pl.read_parquet(feats_path)
    result = run_outcome_sweep(
        features,
        train_end_exclusive=args.train_end,
        out_dir=Path(args.out_dir),
        seed=args.seed,
    )
    ok = result.get("status") == "ok"
    print(json.dumps({"ok": ok, **result}, indent=2, default=str))
    return 0 if ok else 1


def _cmd_collect_books(args: argparse.Namespace) -> int:
    from polymarket_analytics.collectors.book_collector import discover_active_token_ids, run_collection

    token_ids = [str(t) for t in (args.token_id or [])]
    if not token_ids:
        try:
            token_ids = discover_active_token_ids(args.n_tokens)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"token discovery failed: {exc}"}))
            return 1
    if not token_ids:
        print(json.dumps({"ok": False, "error": "no token ids"}))
        return 1

    report = run_collection(
        token_ids=token_ids[: args.n_tokens],
        duration_sec=args.duration,
        max_tokens=args.n_tokens,
        data_root=Path(args.data_root),
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 1


def _cmd_backfill_year(args: argparse.Namespace) -> int:
    from polymarket_analytics.research.overnight_backfill import backfill_year

    max_markets = 5 if args.tiny else args.max_markets
    trade_limit = 50 if args.tiny else args.trade_limit_per_market
    try:
        result = backfill_year(
            args.year,
            args.out_dir,
            market_limit=args.market_limit,
            max_markets=max_markets,
            trade_limit_per_market=trade_limit,
        )
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"ok": True, **result}, indent=2, default=str))
    return 0 if result.get("status") in {"ok", "bounded"} else 1


def _cmd_merge_expanded(args: argparse.Namespace) -> int:
    from polymarket_analytics.research.overnight_merge import merge_expanded_lake

    try:
        result = merge_expanded_lake(args.data_root, args.year_dir, args.existing_canonical, args.out_dir)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps({"ok": True, **result}, indent=2, default=str))
    return 0


def _cmd_outcome_gate(args: argparse.Namespace) -> int:
    from polymarket_analytics.research.overnight_gate import evaluate_outcome_gate

    kwargs = {
        "min_events": args.min_events,
        "train_end_exclusive": getattr(args, "train_end", "2025-01-01T00:00:00+00:00"),
        "baseline_feature_rows": getattr(args, "baseline_rows", None),
        "used_baseline_fallback": bool(getattr(args, "used_baseline_fallback", False)),
    }
    result = evaluate_outcome_gate(args.features, **kwargs)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["decision"] != "BLOCKED" else 1


def _cmd_overnight_report(args: argparse.Namespace) -> int:
    from polymarket_analytics.research.overnight_report import write_overnight_report

    pieces: dict[str, object] = {}
    for path in args.manifest:
        try:
            pieces[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            pieces[path.stem] = {"status": "unavailable", "error": str(exc), "path": str(path)}
    report = write_overnight_report(args.root, args.out, pieces)
    print(json.dumps({"ok": True, "report": str(report), "json": str(report.with_suffix(".json"))}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Strip accidental "# comment" tokens if a pasted line includes them
    if argv is None:
        argv = sys.argv[1:]
    cleaned: list[str] = []
    for a in argv:
        if a.startswith("#"):
            break
        cleaned.append(a)
    parser = build_parser()
    args = parser.parse_args(cleaned)
    # Allow --side '' to mean no side filter
    if getattr(args, "side", None) == "":
        args.side = None
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
