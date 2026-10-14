"""Machine-detectable proofreading scan for a LaTeX manuscript.

Covers the mechanical half of the "full technical and language proofread"
plan item: spelling candidates, doubled words, notation used outside math
mode, and terminology drift. It does NOT judge prose quality.

Usage:
    python tools/proofread_scan.py <paper.tex> [--report <out.txt>]

Exit code is 0 always; findings are reported, not enforced.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Words that are commonly misspelled; a hit is a candidate, not a verdict.
TYPOS = [
    "accross", "seperate", "recieve", "occured", "adress", "commited",
    "comparision", "dependancy", "existance", "maintainance", "neccessary",
    "occurence", "priviledge", "refered", "sucessful", "untill", "wierd",
    "acheive", "concious", "definately", "independant", "liason", "publically",
    "recomend", "relevent", "similiar", "succesful", "treshold", "useage",
]

# Pairs that should not both appear: pick one form and use it everywhere.
TERMINOLOGY_PAIRS = [
    ("ground truth", "ground-truth"),
    ("real-world", "real world"),
    ("data set", "dataset"),
    ("sub-sample", "subsample"),
    ("hyper parameter", "hyperparameter"),
    ("non-trivial", "nontrivial"),
    ("nprobe", "n-probe"),
    ("nlist", "n-list"),
    ("cut-off", "cutoff"),
    ("trade-off", "tradeoff"),
]

# Symbols that must never appear as bare text outside math mode.
MATH_ONLY = ["sigma", "sigma0", "alpha", "epsilon", "mu", "lambda"]

# Display-math environments whose contents must not be scanned as prose.
MATH_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "displaymath",
    "split", "cases", "aligned", "gathered",
}


def blank_math_envs(lines: list[str]) -> list[str]:
    """Replace the contents of display-math environments with blanks."""
    out: list[str] = []
    env: str | None = None
    for line in lines:
        if env is None:
            m = re.search(r"\\begin\{([A-Za-z*]+)\}", line)
            if m and m.group(1) in MATH_ENVS:
                env = m.group(1)
                out.append(" ")
                continue
            out.append(line)
        else:
            if re.search(rf"\\end\{{{re.escape(env)}\}}", line):
                env = None
            out.append(" ")
    return out


def strip_comments(text: str) -> list[str]:
    """Return lines with LaTeX comments removed, preserving line numbering."""
    out = []
    for line in text.splitlines():
        idx = None
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                idx = i
                break
        out.append(line if idx is None else line[:idx])
    return out


def outside_math(line: str) -> str:
    """Blank out math spans delimited by $...$, \\(...\\) or \\[...\\]."""
    line = re.sub(r"\\\(.*?\\\)", lambda m: " " * len(m.group(0)), line)
    line = re.sub(r"\\\[.*?\\\]", lambda m: " " * len(m.group(0)), line)
    result = []
    in_math = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "$" and (i == 0 or line[i - 1] != "\\"):
            in_math = not in_math
            result.append(" ")
        elif in_math:
            result.append(" ")
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def scan(path: Path) -> tuple[list[str], dict[str, int]]:
    raw = path.read_text(encoding="utf-8")
    lines = blank_display_spans(blank_math_envs(strip_comments(raw)))
    findings: list[str] = []
    counts: dict[str, int] = {}

    # 1. spelling candidates
    hits: Counter[str] = Counter()
    for word in TYPOS:
        for line in lines:
            hits[word] += len(re.findall(rf"\b{word}\b", line, re.I))
    hits = Counter({w: n for w, n in hits.items() if n})
    if hits:
        findings.append("Spelling candidates:")
        for word, n in sorted(hits.items()):
            findings.append(f"  {word} x{n}")
    counts["spelling_candidates"] = sum(hits.values())

    # 2. doubled words
    doubled = []
    for n, line in enumerate(lines, 1):
        # Column specs such as {@{} l XX XX XX @{}} are not prose.
        if re.search(r"\\begin\{(tabular|tabularx|array)\}", line):
            continue
        for m in re.finditer(r"\b([A-Za-z]{2,})\s+\1\b", line, re.I):
            # LaTeX scaffolding such as \end{...} legitimately repeats tokens.
            if m.group(1).lower() in {"begin", "end", "hline", "toprule"}:
                continue
            doubled.append(f"  L{n}: {line.strip()[:120]}")
    if doubled:
        findings.append("Doubled words:")
        findings.extend(doubled)
    counts["doubled_words"] = len(doubled)

    # 3. math-only symbols as bare text
    bare = []
    for n, line in enumerate(lines, 1):
        plain = outside_math(line)
        for sym in MATH_ONLY:
            for m in re.finditer(rf"\b{sym}\b", plain, re.I):
                bare.append(f"  L{n}: '{m.group(0)}' in: ...{plain[max(0, m.start()-40):m.start()+30].strip()}...")
    if bare:
        findings.append("Math-only symbols as bare text:")
        findings.extend(bare[:40])
    counts["bare_math_symbols"] = len(bare)

    # 4. terminology drift
    joined = "\n".join(lines)
    drift = []
    for a, b in TERMINOLOGY_PAIRS:
        na = len(re.findall(rf"\b{re.escape(a)}\b", joined, re.I))
        nb = len(re.findall(rf"\b{re.escape(b)}\b", joined, re.I))
        if na and nb:
            drift.append(f"  '{a}' x{na}  vs  '{b}' x{nb}")
    if drift:
        findings.append("Terminology drift (both forms present):")
        findings.extend(drift)
    counts["terminology_drift"] = len(drift)

    # 5. spacing anomalies
    dbl = [f"  L{n}: {l.strip()[:120]}" for n, l in enumerate(lines, 1) if re.search(r"[a-z]  +[a-z]", l)]
    if dbl:
        findings.append("Possible double spaces inside words:")
        findings.extend(dbl[:20])
    counts["double_spaces"] = len(dbl)

    return findings, counts


def blank_display_spans(lines: list[str]) -> list[str]:
    """Blank out display-math spans delimited by bare \\[ and \\] lines."""
    out: list[str] = []
    open_span = False
    for line in lines:
        stripped = line.strip()
        if not open_span:
            if stripped == "\\[":
                open_span = True
                out.append(" ")
                continue
            out.append(line)
        else:
            if stripped == "\\]":
                open_span = False
            out.append(" ")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper")
    ap.add_argument("--report")
    args = ap.parse_args()

    path = Path(args.paper)
    findings, counts = scan(path)

    header = [
        f"Proofread scan: {path.name}",
        f"Lines: {len(strip_comments(path.read_text(encoding='utf-8')))}",
        "",
        "Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
    ]
    body = findings if findings else ["No machine-detectable issues found.", ""]
    tail = [
        "",
        "NOTE: this covers mechanics only. Prose, argument order, and claim",
        "strength are not checkable by script and still require a human pass.",
    ]
    report = "\n".join(header + body + tail)

    print(report)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
