"""CLI entrypoints: ingest, compute-features, find-edges, backtest, paper-trade, status."""

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
    )

    # --min-ev 0.10 means 10 percentage points (matches OOS report units)
    min_ev_pp = float(args.min_ev)
    if min_ev_pp <= 1.0:
        min_ev_pp *= 100.0

    cfg = PaperConfig(
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position_pct,
        fee_bps=args.fee_bps,
        spread_slippage_bps=args.spread_slippage_bps,
        min_oos_ev_pct=min_ev_pp,
        require_persists=not args.allow_decayed,
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

        feed = MarketWebSocketFeed(token_ids, feature_engine=engine)
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
    paper_p.add_argument("--fee-bps", type=float, default=0.0)
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

    return parser


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
