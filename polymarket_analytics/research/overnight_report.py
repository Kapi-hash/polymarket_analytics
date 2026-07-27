"""Morning report writer for overnight research artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_overnight_report(root: Path, out_md: Path, pieces: dict[str, Any]) -> Path:
    """Write a readable markdown report and a lossless JSON twin."""
    root, out_md = Path(root), Path(out_md)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "pieces": pieces,
    }
    lines = [
        "# Overnight Polymarket Research Report",
        "",
        f"Generated: {payload['generated_at']}",
        f"Root: `{root}`",
        "",
    ]
    for name, value in pieces.items():
        lines.extend([f"## {name.replace('_', ' ').title()}", "", "```json",
                      json.dumps(value, indent=2, default=str), "```", ""])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_md
