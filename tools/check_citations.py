"""Report LaTeX citation keys with no matching BibTeX entry, and unused entries.

This is the check behind the reviewer complaint about unresolved "[?]"
references. Run it against the manuscript and its bibliography:

    python tools/check_citations.py <paper.tex> <references.bib>
"""

import re
import sys
from pathlib import Path

BIB_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
CITE_CMD = re.compile(
    r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear|parencite|textcite"
    r"|autocite|smartcite|footcite|nocite)\*?"
    r"\s*(?:\[[^\]]*\]\s*)*"
    r"\{([^}]*)\}"
)


def bib_keys(text: str) -> set:
    return {m.group(1).strip() for m in BIB_ENTRY.finditer(text)}


def cite_keys(text: str) -> set:
    keys = set()
    for match in CITE_CMD.finditer(text):
        for raw in match.group(1).split(","):
            key = raw.strip()
            if key:
                keys.add(key)
    return keys


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python tools/check_citations.py <paper.tex> <refs.bib>")

    tex_path, bib_path = Path(sys.argv[1]), Path(sys.argv[2])
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    bib = bib_path.read_text(encoding="utf-8", errors="replace")

    cited = cite_keys(tex)
    defined = bib_keys(bib)

    missing = sorted(cited - defined)
    unused = sorted(defined - cited)

    print(f"file       : {tex_path}")
    print(f"bib        : {bib_path}")
    print(f"cited keys : {len(cited)}")
    print(f"bib entries: {len(defined)}")

    print(f"\nMISSING from .bib ({len(missing)}) -- these render as [?]:")
    for key in missing:
        print(f"  - {key}")
    if not missing:
        print("  (none)")

    print(f"\nUnused .bib entries ({len(unused)}):")
    for key in unused[:50]:
        print(f"  - {key}")
    if len(unused) > 50:
        print(f"  ... and {len(unused) - 50} more")
    if not unused:
        print("  (none)")


if __name__ == "__main__":
    main()
