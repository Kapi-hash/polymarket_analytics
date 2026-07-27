# Restricted mid-fill diagnostic (NOT full sweep)

- Git: `c6a043f02a66c560986c33a6e78910ae68e0f114`
- Attempts: 378
- Candidates (n≥30, events≥5): 174
- WF survivors: 5
- Final-test configs: 5
- Train events: 44, purged test events: 37

## Fee / execution labels
- FALLBACK — lake lacks fee_category; crypto schedule applied for sensitivity only
- trade-print mid fill + 50bps slip; no L2 walk

## Top train candidates (net fallback)
- `bucket=0.30-0.40 AND spike>2 AND whale>2 AND mom1h=pos AND side=BUY`: net EV=27.04%, n=157, events=11, sharpe=6.815
- `bucket=0.30-0.40 AND spike>1.5 AND whale>2 AND mom1h=pos AND side=BUY`: net EV=25.78%, n=165, events=11, sharpe=6.576
- `bucket=0.30-0.40 AND spike>2 AND whale>3 AND mom1h=pos AND side=BUY`: net EV=25.05%, n=109, events=11, sharpe=5.187
- `bucket=0.30-0.40 AND spike>1.5 AND whale>3 AND mom1h=pos AND side=BUY`: net EV=24.47%, n=115, events=11, sharpe=5.153
- `bucket=0.40-0.50 AND whale>3 AND side=BUY`: net EV=23.13%, n=293, events=20, sharpe=8.681
- `bucket=0.40-0.50 AND spike>1.5 AND whale>3 AND side=BUY`: net EV=22.33%, n=268, events=18, sharpe=8.008
- `bucket=0.30-0.40 AND whale>3 AND mom1h=pos AND side=BUY`: net EV=22.08%, n=127, events=13, sharpe=4.780
- `bucket=0.40-0.50 AND spike>2 AND whale>3 AND side=BUY`: net EV=21.79%, n=256, events=18, sharpe=7.620
- `bucket=0.70-0.80 AND spike>2 AND whale>2 AND ttr<72h AND side=BUY`: net EV=21.45%, n=31, events=5, sharpe=6.612
- `bucket=0.70-0.80 AND spike>1.5 AND whale>2 AND ttr<72h AND side=BUY`: net EV=21.45%, n=31, events=5, sharpe=6.612

## Locked final test (fallback fees)
- `bucket=0.30-0.40 AND spike>1.5 AND whale>2 AND mom1h=pos AND side=BUY`: test net EV=15.71%, n=21, events=7
- `bucket=0.30-0.40 AND spike>2 AND whale>3 AND mom1h=pos AND side=BUY`: test net EV=13.89%, n=14, events=5
- `bucket=0.30-0.40 AND spike>1.5 AND whale>3 AND mom1h=pos AND side=BUY`: test net EV=13.89%, n=14, events=5
- `bucket=0.30-0.40 AND spike>2 AND whale>2 AND mom1h=pos AND side=BUY`: test net EV=13.49%, n=20, events=7
- `bucket=0.40-0.50 AND whale>3 AND side=BUY`: test net EV=7.37%, n=49, events=8

**Do not treat these as paper-trading recommendations.** Critical gates for execution-realistic evaluation remain failed.

## Robustness label

All five final-test configs are **FRAGILE**, not robust candidates:
- Walk-forward fold 2 is largely negative for every top config (catastrophic single regime).
- Independent-event counts on final test are 5–8 (far below preferred ≥100).
- Fees are FALLBACK crypto; execution is mid-print, not book-walk.
- Placeholder multiple-testing FDR on train sharpes is scaffolding only (DSR/PBO not decision-grade).

**Robust candidates qualifying for paper trading: 0**
