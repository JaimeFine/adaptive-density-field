"""Detect markdown/pandoc conversion artifacts left inside LaTeX.

The manuscript's appendix was evidently converted from markdown, which leaves
behind constructs that compile but are wrong: pandoc macros, unconverted
bullet lists, Unicode typographic characters, and stray line breaks. These
are invisible to a spell-checker and to LaTeX itself.

Usage:
    python tools/latex_artifact_scan.py <paper.tex> [--report <out.txt>]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Unicode punctuation that commonly leaks from markdown/WYSIWYG sources.
UNICODE_SUSPECTS = {
    "\u2010": "hyphen U+2010",
    "\u2011": "non-breaking hyphen U+2011",
    "\u2012": "figure dash U+2012",
    "\u2013": "en dash U+2013",
    "\u2014": "em dash U+2014",
    "\u2018": "left single quote U+2018",
    "\u2019": "right single quote U+2019",
    "\u201c": "left double quote U+201C",
    "\u201d": "right double quote U+201D",
    "\u2026": "ellipsis U+2026",
    "\u00a0": "non-breaking space U+00A0",
    "\u2009": "thin space U+2009",
    "\u00d7": "multiplication sign U+00D7",
    "\u2212": "minus sign U+2212",
}

# Structural leftovers from pandoc or from pasted markdown.
STRUCTURAL = [
    (r"\\tightlist", "pandoc \\tightlist macro (use, not definition)", r"\\(providecommand|newcommand|renewcommand|def)"),
    (r"(?m)^\s*[-*+]\s+\S", "unconverted markdown bullet at line start"),
    (r"(?m)^\s*\d+\.\s+[A-Za-z]", "unconverted markdown ordered-list item"),
    (r"(?m)^\s*#{1,6}\s+\S", "unconverted markdown heading"),
    (r"(?m)^\s*```", "unconverted markdown code fence"),
    (r"\*\*[A-Za-z][^*]*\*\*", "markdown bold left as **...**"),
]


def strip_comments(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        idx = None
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                idx = i
                break
        out.append(line if idx is None else line[:idx])
    return out


def scan(path: Path) -> tuple[list[str], dict[str, int]]:
    raw = path.read_text(encoding="utf-8")
    lines = strip_comments(raw)
    findings: list[str] = []
    counts: dict[str, int] = {}

    # Unicode suspects, with the first few line numbers for each.
    uni_total = 0
    uni_lines: dict[str, list[int]] = {}
    for n, line in enumerate(lines, 1):
        for ch, label in UNICODE_SUSPECTS.items():
            k = line.count(ch)
            if k:
                uni_total += k
                uni_lines.setdefault(label, []).append(n)
    if uni_lines:
        findings.append("Unicode typographic characters (LaTeX normally renders these poorly):")
        for label, ls in sorted(uni_lines.items()):
            shown = ", ".join(str(x) for x in ls[:12])
            more = "" if len(ls) <= 12 else f", +{len(ls) - 12} more"
            findings.append(f"  {label}: lines {shown}{more}")
    counts["unicode_chars"] = uni_total

    # Structural artifacts.
    struct_total = 0
    for pattern, label, *excl in STRUCTURAL:
        exclude = excl[0] if excl else None
        hits = [
            n
            for n, line in enumerate(lines, 1)
            if re.search(pattern, line) and not (exclude and re.search(exclude, line))
        ]
        if hits:
            struct_total += len(hits)
            shown = ", ".join(str(x) for x in hits[:12])
            more = "" if len(hits) <= 12 else f", +{len(hits) - 12} more"
            findings.append(f"  {label}: lines {shown}{more}")
    counts["structural_artifacts"] = struct_total

    # Stray \\ line breaks at the end of a prose line (outside tables/arrays).
    stray = []
    for n, line in enumerate(lines, 1):
        if re.search(r"\\begin\{(tabular|tabularx|array)\}|\\end\{(tabular|tabularx|array)\}", line):
            continue
        if re.search(r"[A-Za-z]\\\\\s*$", line):
            stray.append(f"  L{n}: {line.strip()[:110]}")
    if stray:
        findings.append("Stray \\\\ line break at end of prose line:")
        findings.extend(stray[:25])
    counts["stray_breaks"] = len(stray)

    # Unbalanced inline math delimiters per line.
    unbalanced = []
    for n, line in enumerate(lines, 1):
        if line.count("$") % 2:
            unbalanced.append(f"  L{n}: {line.strip()[:110]}")
    if unbalanced:
        findings.append("Odd number of $ on a line (possible unbalanced math):")
        findings.extend(unbalanced[:25])
    counts["unbalanced_math"] = len(unbalanced)

    return findings, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper")
    ap.add_argument("--report")
    args = ap.parse_args()

    path = Path(args.paper)
    findings, counts = scan(path)

    header = [
        f"LaTeX artifact scan: {path.name}",
        "",
        "Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
    ]
    body = findings if findings else ["No conversion artifacts found.", ""]
    report = "\n".join(header + body)

    print(report)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
