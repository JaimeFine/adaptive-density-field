"""Normalise LaTeX typography and remove markdown conversion leftovers.

Fixes the mechanical class of proofreading defects found by
``latex_artifact_scan.py``: Unicode punctuation that leaked in from a markdown
source, and pandoc's ``\\tightlist`` spacing macro.

Dry-run by default; pass --apply to write.

Usage:
    python tools/latex_normalize.py <paper.tex> [--apply]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Unicode -> LaTeX-safe. Only characters actually present are listed first.
CHAR_MAP = {
    "\u2010": "-",        # hyphen
    "\u2011": "-",        # non-breaking hyphen
    "\u2012": "-",        # figure dash
    "\u2013": "--",       # en dash
    "\u2014": "---",      # em dash
    "\u2018": "`",        # left single quote
    "\u2019": "'",        # right single quote
    "\u201c": "``",       # left double quote
    "\u201d": "''",       # right double quote
    "\u00a0": " ",        # non-breaking space
    "\u2009": " ",        # thin space
    "\u2026": "...",      # ellipsis
    "\u2212": "-",        # minus sign
}

TIGHTLIST = "\\tightlist"


def normalize_line(line: str, counts: Counter[str]) -> str:
    """Apply character substitutions and drop bare \\tightlist lines."""
    # A line consisting only of \tightlist is pandoc spacing scaffolding.
    if line.strip() == TIGHTLIST:
        counts["tightlist_removed"] += 1
        return ""

    for ch, repl in CHAR_MAP.items():
        if ch in line:
            counts[ch] += line.count(ch)
            line = line.replace(ch, repl)
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    path = Path(args.paper)
    text = path.read_text(encoding="utf-8")
    had_trailing_newline = text.endswith("\n")
    lines = text.split("\n")

    counts: Counter[str] = Counter()
    changed: list[tuple[int, str, str]] = []
    out: list[str] = []

    for n, line in enumerate(lines, 1):
        new = normalize_line(line, counts)
        if new != line:
            changed.append((n, line, new))
        out.append(new)

    result = "\n".join(out)
    if had_trailing_newline and not result.endswith("\n"):
        result += "\n"

    print(f"File: {path}")
    print(f"Lines changed: {len(changed)}")
    print("Substitutions:")
    for ch, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        label = ch if len(ch) > 1 else f"U+{ord(ch):04X}"
        print(f"  {label}: {n}")

    if changed:
        print("\nFirst 20 changed lines:")
        for n, before, after in changed[:20]:
            print(f"  L{n}")
            print(f"    - {before.strip()[:100]}")
            print(f"    + {after.strip()[:100]}")

    if args.apply:
        path.write_text(result, encoding="utf-8")
        print(f"\nAPPLIED to {path}")
    else:
        print("\nDry run. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
