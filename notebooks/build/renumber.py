"""Renumber the notebook series in one atomic pass.

    python notebooks/build/renumber.py --dry-run     # show every change, touch nothing
    python notebooks/build/renumber.py --apply       # rename files (git mv) and rewrite references

Renumbering by hand is error-prone because the mapping overlaps -- old 08 becomes 10 while old 10
becomes 12 -- so a sequential find-and-replace corrupts anything it has already rewritten. Every
substitution here happens in a single regex pass driven by one mapping, which makes the operation
atomic. Edit MOVES and NUMS below and re-run; the next renumber is then minutes, not a day.

**This transform is not idempotent.** It maps old numbers to new ones, so running it twice maps the
new numbers again. Run ``--apply`` exactly once per mapping, check ``git status`` before re-running,
and if a file was missed, rewrite only that file rather than re-running the whole pass.

What it rewrites:
  * notebook filenames (via ``git mv`` so history follows)
  * the ``--out`` defaults and docstrings of the builders in this directory
  * every textual cross-reference: ``pairs_trading_NN``, ``nbNN``, ``notebook NN``,
    ``notebooks NN-MM`` and the old un-numbered stems
  * README link labels of the form [`NN`](notebooks/...), which are derived from the *target*
    rather than mapped -- several were already stale before this script existed
  * the same references inside executed .ipynb files, so notebooks 01-03 (hand-written, with no
    builder) stay consistent without being re-executed
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NB = ROOT / "notebooks"
BUILD = NB / "build"

# old stem -> new stem. Order is irrelevant; the pass is atomic.
MOVES: dict[str, str] = {
    # 01-05 keep their numbers and names
    "visualize_cointegrated_pairs_yahoo":   "pairs_trading_06_cointegration_network_yahoo",
    "tv_cointegration_kalman_yahoo":        "pairs_trading_07_tv_cointegration_kalman_yahoo",
    "pairs_trading_06_yahoo_vs_day_lake":   "pairs_trading_08_yahoo_vs_day_lake",
    "pairs_trading_07_daily_lake":          "pairs_trading_09_daily_lake",
    "pairs_trading_08_daily_cointegration": "pairs_trading_10_daily_cointegration",
    "pairs_trading_09_daily_portfolio":     "pairs_trading_11_daily_portfolio",
    "pairs_trading_10_daily_cross_sectional": "pairs_trading_12_daily_cross_sectional",
    "network_pairs_day_lake":               "pairs_trading_13_market_networks_day_lake",
    "alpha_concepts_day_lake":              "pairs_trading_14_alpha_concepts_day_lake",
    "pairs_trading_11_minute_data":         "pairs_trading_15_minute_data",
    "pairs_trading_12_intraday_backtest":   "pairs_trading_16_intraday_backtest",
    "pairs_trading_13_intraday_portfolio":  "pairs_trading_17_intraday_portfolio",
}

# old number -> new number, for prose references. The two notebooks that had no number before are
# absent: references to them are by name and are handled by the stem substitution above.
NUMS: dict[int, int] = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                        6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 15, 12: 16, 13: 17}

# Ranges written without a "notebook"/"nb" prefix cannot be matched safely by pattern -- a bare
# "11-13" is indistinguishable from a date or a count -- so the few that exist are listed here.
EXTRA: dict[str, str] = {
    "**day lake** and 11\u201313 the **minute lake**": "**day lake** and 15\u201317 the **minute lake**",
}

_STEMS = sorted(MOVES, key=len, reverse=True)      # longest first so prefixes cannot shadow
PAT = re.compile(
    r"(?P<stem>" + "|".join(re.escape(s) for s in _STEMS) + r")"
    r"|(?P<rng>\b(?P<rw>[Nn]otebooks?|nb)(?P<rsp>\s*)(?P<r1>\d{2})\s*[–-]\s*(?P<r2>\d{2})\b)"
    r"|(?P<one>\b(?P<ow>[Nn]otebooks?|nb)(?P<osp>\s*)(?P<o1>\d{2})\b)"
    r"|(?P<pt>\bpairs_trading_(?P<p1>\d{2}))"
)


def _map(tok: str) -> str:
    n = NUMS.get(int(tok))
    return f"{n:02d}" if n is not None else tok


def _sub(m: re.Match) -> str:
    if m.group("stem"):
        return MOVES[m.group("stem")]
    if m.group("rng"):
        return f"{m.group('rw')}{m.group('rsp')}{_map(m.group('r1'))}–{_map(m.group('r2'))}"
    if m.group("one"):
        return f"{m.group('ow')}{m.group('osp')}{_map(m.group('o1'))}"
    return f"pairs_trading_{_map(m.group('p1'))}"


def rewrite(text: str) -> str:
    out = PAT.sub(_sub, text)
    for old, new in EXTRA.items():
        out = out.replace(old, new)
    # README short labels: take the number from the link target, not from the old label. Several
    # were already wrong before this script existed, and mapping a wrong label keeps it wrong.
    out = re.sub(r"\[`\d{2}`\]\(notebooks/pairs_trading_(\d{2})_",
                 lambda m: f"[`{m.group(1)}`](notebooks/pairs_trading_{m.group(1)}_", out)
    return out


def patch_notebook(p: Path) -> int:
    """Rewrite references inside an executed notebook, leaving its outputs untouched."""
    nb = json.loads(p.read_text())
    hits = 0
    for c in nb["cells"]:
        src = "".join(c["source"])
        new = rewrite(src)
        if new != src:
            hits += 1
            c["source"] = new.splitlines(keepends=True)
    if hits:
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    text_files = (sorted(BUILD.glob("*.py")) + sorted(BUILD.glob("*.md"))
                  + sorted((ROOT / "pairs").rglob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
                  + [ROOT / "README.md"])
    text_files = [f for f in text_files if f.name != "renumber.py"]
    nb_files = sorted(NB.glob("*.ipynb"))

    print("── file renames ──")
    renames = []
    for old, new in MOVES.items():
        src, dst = NB / f"{old}.ipynb", NB / f"{new}.ipynb"
        if src.exists():
            renames.append((src, dst))
            print(f"  {src.name}\n    -> {dst.name}")
        else:
            print(f"  [skip] {src.name} does not exist")

    for f in text_files + nb_files:
        t = f.read_text()
        for old in EXTRA:
            if old in t:
                print(f"  [literal] {f.relative_to(ROOT)}: {old[:50]}…")

    print("\n── reference rewrites ──")
    total = 0
    for f in text_files:
        t = f.read_text()
        n = sum(1 for _ in PAT.finditer(t))
        lab = len(re.findall(r"\[`\d{2}`\]\(notebooks/pairs_trading_\d{2}_", t))
        if n or lab:
            total += n
            print(f"  {f.relative_to(ROOT)}: {n} reference(s), {lab} README label(s)")
    for f in nb_files:
        t = f.read_text()
        n = sum(1 for _ in PAT.finditer(t))
        if n:
            total += n
            print(f"  {f.relative_to(ROOT)}: {n} reference(s)")
    print(f"\n  {total} references across {len(text_files) + len(nb_files)} files")

    if a.dry_run:
        print("\ndry run: nothing written. Re-run with --apply.")
        return 0

    for f in text_files:
        t = f.read_text()
        new = rewrite(t)
        if new != t:
            f.write_text(new)
    for f in nb_files:
        patch_notebook(f)
    for src, dst in renames:
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=True)

    print(f"\napplied: {len(renames)} renames, references rewritten in "
          f"{len(text_files)} text files and {len(nb_files)} notebooks.")
    print("Next: rebuild the generated notebooks and confirm they match the patched files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
