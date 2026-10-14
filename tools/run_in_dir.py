"""Run a Python script with its own directory as the working directory.

The ADF scripts address their inputs with paths relative to their own
folder (e.g. ``data/poi_without_ctu.csv``), while the terminal runner may
start in a different directory. This launcher makes execution
cwd-independent.

Usage:
    python tools/run_in_dir.py <path/to/script.py> [script args...]
"""

import os
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python tools/run_in_dir.py <script.py> [args...]")

    script = os.path.abspath(sys.argv[1])
    workdir = os.path.dirname(script)

    os.chdir(workdir)
    sys.argv = [script, *sys.argv[2:]]

    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
