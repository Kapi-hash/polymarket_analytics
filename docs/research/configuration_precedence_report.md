# Configuration precedence

- 1. Explicit CLI flags (argparse) when command invoked
- 2. Profile builder overrides (profiles.py) for multi-profile incubator
- 3. Dataclass field defaults (PaperConfig / SwingConfig / StrategyParams)
- 4. Module-level constants (schema.WHALE_*, backtest.DEFAULT_* grids)
- 5. Hardcoded literals inside functions (lowest; flagged Hardcoded)
- Note: hierarchical YAML (config/research_priors.yaml) is advisory recommended priors and does NOT override code defaults unless explicitly loaded.