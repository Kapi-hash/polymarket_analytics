# Configuration conflict report

- **stall_hours**: {'concept': 'stall_hours', 'locations': ['swing_trader.SwingConfig.stall_hours=36.0', 'cli.swing-trade --stall-hours default=48.0'], 'notes': 'CLI default overrides dataclass when swing-trade invoked; profile builders use 36.0'}

- **cooldown_sec (swing)**: {'concept': 'cooldown_sec (swing)', 'locations': ['swing_trader.SwingConfig.cooldown_sec=300.0', 'profiles._build_* cooldown_sec=120.0'], 'notes': 'Incubator profiles tighten cooldown vs bare SwingConfig default'}

- **book_imbalance_min**: {'concept': 'book_imbalance_min', 'locations': ['SwingConfig.book_imbalance_min=3.0', 'SwingConfig.confluence_book_min=2.5', 'profiles confluence uses 2.5'], 'notes': 'Standalone book strategy vs confluence threshold diverge'}

- **min_ev paper vs swing CLI**: {'concept': 'min_ev paper vs swing CLI', 'locations': ['cli.paper-trade --min-ev default=0.10 (→10pp)', 'cli.swing-trade --min-ev default=0.08 (→8pp)', 'PaperConfig.min_oos_ev_pct=10.0', 'SwingConfig.min_ev_pct=8.0'], 'notes': 'Intentional family difference but easy to confuse'}

- **whale threshold**: {'concept': 'whale threshold', 'locations': ['schema.WHALE_RATIO_DIVERGENCE_THRESHOLD=3.0', 'backtest.DEFAULT_WHALE_THRESHOLDS=(None,2.0,3.0)', 'profiles whale min_whale_ratio=3.0'], 'notes': 'Aligned at 3.0 for divergence; grid also sweeps 2.0'}
