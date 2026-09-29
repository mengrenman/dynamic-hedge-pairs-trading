"""Write the README gallery figures from the executed notebooks.

    python notebooks/build/readme_figures.py            # writes figures/<name>.png for every entry
    python notebooks/build/readme_figures.py --check    # exit non-zero if any figure on disk differs

Each figure is the image output of one notebook cell, found by a marker string that appears in that
cell's source (not by cell index, so inserting a cell does not move the figure). The README embeds
these files; regenerating them here after a notebook is re-executed keeps the pictures as current as
the numbers, which nothing else checks.
"""
import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NB = ROOT / "notebooks"
FIGURES = ROOT / "figures"

# name -> (notebook stem, marker in the cell source, which image output of that cell)
GALLERY = {
    "nb11_twenty_years":  ("pairs_trading_11_daily_portfolio",           "Twenty years, $10,000 per pair, point-in-time universe", 0),
    "nb12_first_decade":  ("pairs_trading_12_daily_cross_sectional",     "The edge is in the first decade", 0),
    "nb16_revaluation":   ("pairs_trading_16_kalman_pnl_accounting_day_lake", "book / tradable / revaluation cumulative curves", 0),
    "nb21_cost_cliff":    ("pairs_trading_21_intraday_desk_alphas",      "Net Sharpe against a flat per-side cost", 0),
}


def extract(stem: str, marker: str, which: int) -> bytes:
    cells = json.loads((NB / f"{stem}.ipynb").read_text())["cells"]
    hits = [c for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit(f"{stem}: marker {marker!r} matches {len(hits)} cells, expected exactly one")
    pngs = [o["data"]["image/png"] for o in hits[0].get("outputs", []) if "data" in o and "image/png" in o["data"]]
    if len(pngs) <= which:
        raise SystemExit(f"{stem}: the marked cell has {len(pngs)} image output(s), wanted index {which}")
    return base64.b64decode("".join(pngs[which]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    ap.add_argument("--only", nargs="*", help="a subset of gallery names")
    a = ap.parse_args()
    FIGURES.mkdir(exist_ok=True)
    stale = []
    for name, (stem, marker, which) in GALLERY.items():
        if a.only and name not in a.only:
            continue
        png = extract(stem, marker, which)
        out = FIGURES / f"{name}.png"
        if a.check:
            if not out.exists() or out.read_bytes() != png:
                stale.append(name)
            continue
        out.write_bytes(png)
        print(f"{out.relative_to(ROOT)}: {len(png):,} bytes from {stem} [{marker[:40]}]")
    if a.check:
        print("stale:" if stale else "all gallery figures current", *stale)
        return 1 if stale else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
