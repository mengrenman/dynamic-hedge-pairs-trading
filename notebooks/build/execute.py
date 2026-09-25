"""Execute a notebook in place with the repo's kernel.

    python notebooks/build/execute.py notebooks/pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb [--kernel python3] [--timeout 3600]

Runs with the notebook's own directory as the working directory (for the notebooks in this
repo that is notebooks/, and they resolve the repo root as Path.cwd().parent), stores the
outputs back into the file, normalizes cell ids, and exits non-zero if any cell raised. Cell
errors do not stop the run, so the failing cell and every later one are visible in the saved
notebook.
"""
import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--kernel", default="python3", help="kernelspec name (default: python3)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    args = ap.parse_args()

    path = args.notebook.resolve()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, kernel_name=args.kernel, timeout=args.timeout, allow_errors=True,
        resources={"metadata": {"path": str(path.parent)}},
    )
    t0 = time.time()
    client.execute()
    _, nb = nbformat.validator.normalize(nb)
    nbformat.write(nb, path)

    errors = [
        (i, o["ename"], o["evalue"])
        for i, c in enumerate(nb.cells) if c.cell_type == "code"
        for o in c.get("outputs", []) if o.output_type == "error"
    ]
    print(f"{path.name}: executed in {time.time() - t0:.0f}s; {len(errors)} error(s)")
    for i, name, value in errors:
        print(f"  cell {i}: {name}: {value[:200]}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
