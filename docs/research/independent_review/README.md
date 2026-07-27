# Independent Review (2026-07-26)

Decision: **BLOCKED** — full execution-realistic sweep not authorized.
Restricted mid-fill diagnostic only (non-definitive).

## Layout

```
independent_review/
├── README.md                 ← this file
├── review_scorecard.md       ← gate pass/fail + decision
├── review_scorecard.json
├── review_discrepancies.csv  ← inventory/code discrepancies found
├── remediation_log.md        ← fixes applied during review
├── blocker_report.json       ← exact blockers + restricted-subset notes
├── dataset_summary.json      ← lake coverage snapshot
└── results/
    └── restricted_midfill/   ← honest mid-fill + fee-fallback diagnostic
        ├── backtest_manifest.json
        ├── baselines.parquet
        ├── sweep_attempts.parquet
        ├── sweep_results.parquet
        ├── walk_forward_results.parquet
        ├── final_test_results.parquet
        ├── excluded_markets.csv
        ├── top_strategies.md
        └── feature_attribution.md
```

## Related (parent folder)

Parameter inventory and prior expansion reports live one level up:

- `../parameter_inventory.md` / `.json` / `.csv`
- `../conflict_report.md`, `../hardcoded_threshold_report.md`, …
- `../full_sweep_design.md`

## Re-run

```bash
python3 scripts/run_independent_review.py
```
