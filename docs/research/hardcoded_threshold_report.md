# Hardcoded threshold report

## Inventory-flagged
- `_EPS_HOURS` @ features.py:_EPS_HOURS = 1e-06

## AST heuristic hits (sample)
- `polymarket_analytics/ingest.py:80` literal=100000000000000.0 `as_num.abs() > 100000000000000.0`
- `polymarket_analytics/ingest.py:82` literal=100000000000.0 `as_num.abs() > 100000000000.0`
- `polymarket_analytics/ingest.py:279` literal=0.99 `max(floats) < 0.99`
- `polymarket_analytics/swing_trader.py:63` literal=1e-12 `avg_loss <= 1e-12`
- `polymarket_analytics/swing_trader.py:124` literal=4 `max_k < 4`
- `polymarket_analytics/swing_trader.py:144` literal=1e-12 `std <= 1e-12`
- `polymarket_analytics/swing_trader.py:154` literal=3 `len(ks) < 3`
- `polymarket_analytics/swing_trader.py:162` literal=1e-12 `den <= 1e-12`
- `polymarket_analytics/paper_trader.py:272` literal=3.0 `feat.whale_ratio <= 3.0`
- `polymarket_analytics/paper_trader.py:644` literal=0.99 `float(px) >= 0.99`
- `polymarket_analytics/live_feed.py:42` literal=1000000000000.0 `ts > 1000000000000.0`
- `polymarket_analytics/live_feed.py:44` literal=10000000000.0 `ts > 10000000000.0`
- `polymarket_analytics/live_feed.py:50` literal=1000000000000.0 `ts > 1000000000000.0`
- `polymarket_analytics/live_feed.py:52` literal=10000000000.0 `ts > 10000000000.0`