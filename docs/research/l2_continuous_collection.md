# Continuous L2 microstructure collection

Prospective order-book capture for **future** swing / microstructure research. Does **not** unlock historical L2 backtests or authorize broad `outcome_sweep` runs.

## Schedules

| Workflow | Cron (UTC) | Purpose |
|----------|------------|---------|
| `l2-collect.yml` | `0 */6 * * *` | ~5h session capture (`18000s` default) |
| `l2-compact-daily.yml` | `30 1 * * *` | Daily bundle + coverage after collection window |
| `l2-watchdog.yml` | `15 * * * *` | Detect missing/failed/stale runs; redispatch (max 2/12h) |
| `l2-research-gate.yml` | `0 12 * * 1` | Weekly readiness check (non-definitive) |

Overnight outcome research remains in `overnight-polymarket-research.yml` and uses legacy `data/books/runs/` via `collect-books` **without** `--full-session`.

## Session layout

Full continuous sessions live under `data/books/sessions/<session_id>/`:

```
sessions/<session_id>/
  raw/session_<id>.jsonl.gz
  normalized/utc_date=YYYY-MM-DD/session_<id>_*.parquet
  snapshots/snapshot_*.json
  metadata/tokens.json, markets.json
  health.json
  gap_report.json
  session_manifest.json
  fingerprints.json
  logs/collector.log
```

Legacy overnight smoke runs: `data/books/runs/<session_id>/` with `collector_health.json`.

## Readiness gate

`l2-readiness --coverage-json PATH` checks aggregate daily coverage (sessions, emitting tokens, normalized rows). Passing readiness permits **staged diagnostics only** — explicitly **not** authorization for broad outcome sweeps.

**Important:** A single ~5 hour sample is **non-definitive** for microstructure edge claims (OFI, queue position, maker realism, capacity-from-depth).

## CLI

```bash
# Legacy overnight-compatible capture
python3 -m polymarket_analytics collect-books --duration 45 --n-tokens 5

# Full continuous session (100 tokens, diversified universe)
python3 -m polymarket_analytics collect-books \
  --full-session --universe diversified \
  --duration 18000 --n-tokens 100 --seed 42

python3 -m polymarket_analytics l2-compact-day \
  --utc-date 2026-07-28 \
  --sessions-dir data/books/sessions \
  --out-dir data/books/daily

python3 -m polymarket_analytics l2-readiness --coverage-json data/books/daily/coverage_2026-07-28.json
python3 -m polymarket_analytics l2-diagnostics --session-dir data/books/sessions/<session_id>
python3 -m polymarket_analytics l2-watchdog-check --runs-json artifacts/l2/runs.json
```

## Manual dispatch (GitHub Actions)

- **Collect:** Actions → *L2 Collect* → Run workflow  
  Inputs: `duration_seconds` (default `18000`), `n_tokens` (`100`), `seed`, `smoke`
- **Compact:** Actions → *L2 Compact Daily* → `utc_date` optional (`YYYY-MM-DD`)
- **Watchdog:** Actions → *L2 Watchdog* → Run workflow
- **Research gate:** Actions → *L2 Research Gate* → optional `coverage_json` path

## Artifacts

| Artifact | Retention | Contents |
|----------|-----------|----------|
| `l2-session-raw-<session_id>` | 7d | gzip raw JSONL |
| `l2-session-norm-<session_id>` | 30d | normalized parquet |
| `l2-session-manifest-<session_id>` | 90d | manifest, health, fingerprints, metadata, index fragment |
| `l2-daily-<date>` | 60d | daily tar.gz + coverage JSON |
| `l2-artifact-index` | 90d | merged index of daily/session artifacts |

Data is **never** committed to git.


## Baseline sample (run 30337111979)

Non-definitive reference session from the repaired overnight pipeline (diagnostics only):

| Field | Value |
| --- | --- |
| Session ID | `71c9b3be66f8` |
| Requested / actual duration | 18000s / ~18000.5s |
| Tokens selected / emitting | 100 / 100 |
| Raw / normalized rows | 194,774 / 195,295 |
| Gaps / reconnects / malformed | 0 / 2 / 0 |
| Artifact SHA256 | `bd1431edffa9eaede4387dcbca502eabd8622132168d0c653dd2e740ce6edc35` |

This ~5h sample is **not** research-authorizing. Continuous collection must accumulate ≥7 calendar days before definitive microstructure sweeps.
