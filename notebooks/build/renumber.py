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
# 2026-09-25: make the numbers follow the data source -- day lake 08-16, minute lake 17-19.
MOVES: dict[str, str] = {
    "pairs_trading_18_avellaneda_lee_day_lake":        "pairs_trading_15_avellaneda_lee_day_lake",
    "pairs_trading_19_kalman_pnl_accounting_day_lake": "pairs_trading_16_kalman_pnl_accounting_day_lake",
    "pairs_trading_15_minute_data":                    "pairs_trading_17_minute_data",
    "pairs_trading_16_intraday_backtest":              "pairs_trading_18_intraday_backtest",
    "pairs_trading_17_intraday_portfolio":             "pairs_trading_19_intraday_portfolio",
}

# old number -> new number, for prose references (identity for the untouched ones).
NUMS: dict[int, int] = {**{n: n for n in range(1, 15)}, 15: 17, 16: 18, 17: 19, 18: 15, 19: 16}

# Ranges written without a "notebook"/"nb" prefix cannot be matched safely by pattern -- a bare
# "11-13" is indistinguishable from a date or a count -- so the few that exist are listed here.
EXTRA: dict[str, str] = {
    "Notebooks 08\u201314, 18 and 19 read a local": "Notebooks 08\u201316 read a local",
    "**08\u201314**, **18** and **19** read": "**08\u201316** read",
    "and **15\u201317** read the **minute": "and **17\u201319** read the **minute",
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
                  + [ROOT / "README.md", ROOT / "HANDOFF.md", ROOT / "SYNTHESIS.md"])
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
    # two phases: park every source under a temporary name first, so a cyclic mapping
    # (15->17 while 17->19 and 18->15) never tries to move onto a file that still exists
    tmp = [(src, src.with_name("_renumber_tmp_" + src.name)) for src, _ in renames]
    for src, t in tmp:
        subprocess.run(["git", "mv", str(src), str(t)], cwd=ROOT, check=True)
    for (src, dst), (_, t) in zip(renames, tmp):
        subprocess.run(["git", "mv", str(t), str(dst)], cwd=ROOT, check=True)

    print(f"\napplied: {len(renames)} renames, references rewritten in "
          f"{len(text_files)} text files and {len(nb_files)} notebooks.")
    print("Next: rebuild the generated notebooks and confirm they match the patched files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
