#!/usr/bin/env python3
"""Write docs/research/source_checkpoint.{md,json} for the frozen research state."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True)
    patch = hashlib.sha256(diff.encode()).hexdigest()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tests = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    req = hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest()
    cp = {
        "base_commit": head,
        "patch_sha256": patch,
        "dirty": bool(status.strip()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "requirements_sha256": req,
        "test_command": "python3 -m pytest tests/ -q",
        "test_returncode": proc.returncode,
        "test_result": tests[0],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "preserved_untracked": ["data/swing_trades.json"],
        "notes": (
            "Honesty remediations after c6a043f; independent review pack; "
            "outcome-unblock work continues on this dirty tree until research commit."
        ),
    }
    out = ROOT / "docs" / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_checkpoint.json").write_text(json.dumps(cp, indent=2), encoding="utf-8")
    (out / "source_checkpoint.md").write_text(
        "\n".join(
            [
                "# Source checkpoint",
                "",
                f"- Base commit: `{head}`",
                f"- Patch SHA-256: `{patch}`",
                f"- Dirty: `{bool(status.strip())}`",
                f"- Python: `{cp['python']}`",
                f"- Requirements SHA-256: `{req}`",
                f"- Tests: `{tests[0]}` (rc={proc.returncode})",
                f"- Evaluated at: {cp['evaluated_at']}",
                "",
                "## Included remediations (post-c6a043f)",
                "",
                "- Feature registry status honesty (partial vs stub)",
                "- Inventory latency / book-walk / dead paper axes",
                "- Fee-aware `simulate_strategy` opt-in",
                "- Independent review pack under `docs/research/independent_review/`",
                "",
                "Untracked user file preserved (not committed): `data/swing_trades.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, **{k: cp[k] for k in ("base_commit", "patch_sha256", "test_result")}}, indent=2))
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
