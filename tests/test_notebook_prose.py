"""Every number a notebook's prose asserts should be one the repository's code actually printed.

This exists because of a specific failure. Commit 3a1965f re-executed notebooks 02, 04 and 05 on a
changed pair shortlist and updated their outputs but not notebook 04's §4b.5 narrative, which went
on describing a run that no longer existed. It was not caught for three days, and by then three of
that section's six conclusions were not merely stale but *backwards*: it said the tuned
configuration traded more when it traded a third less, that two of three runner-up pairs improved
when none did, and that the winning settings were an interaction rather than individually good
choices when the marginals said the opposite. Prose and outputs drift silently; nothing else in the
suite looks at prose.

The check is deliberately loose, because the alternative is a test nobody can keep green:

* rounding is fine. Prose saying 1.59 is satisfied by a printed 1.586 — quoting fewer digits than
  were printed is normal writing.
* cross-notebook citation is fine. Notebook 12 may quote notebook 10's screen size, so the haystack
  is every notebook's output, not just its own.
* anything inside a code span is skipped: those are settings and thresholds, not results.
* DOIs, dates, section numbers and version strings are skipped by shape.

What survives all that is a number the prose states and no notebook ever computed, which is worth a
human look every time.
"""
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("pairs_trading_*.ipynb"))
PROSE_NUM = re.compile(r"(?<![\w.$/-])(\d+\.\d{2,4})(?![\d/])")
DOI = re.compile(r"10\.\d{4,}/\S+")


def _outputs(nb):
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                yield "".join(out.get("text", ""))
            for key in ("text/plain", "text/html"):
                data = out.get("data", {}).get(key)
                if not data:
                    continue
                text = "".join(data) if isinstance(data, list) else data
                yield re.sub("<[^>]+>", " ", text) if key == "text/html" else text


@pytest.fixture(scope="module")
def printed():
    """Every number any notebook printed, as floats."""
    vals = []
    for path in NOTEBOOKS:
        blob = " ".join(_outputs(json.loads(path.read_text())))
        flat = blob.replace(",", "")
        vals += [float(m) for m in re.findall(r"(?<![\w.])(\d+\.\d+)", flat)]
        vals += [float(m) for m in re.findall(r"(?<![\w.])(\d{4,})(?![\d.])", flat)]
    return vals


def _unmatched(path, printed):
    nb = json.loads(path.read_text())
    bad = []
    for k, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        md = DOI.sub(" ", re.sub(r"`[^`]*`", " ", "".join(cell["source"])))
        for tok in PROSE_NUM.findall(md):
            value, places = float(tok), len(tok.split(".")[1])
            tol = 0.5 * 10 ** -places
            # printed at higher precision and rounded down in the prose, or abbreviated to
            # thousands/millions ("1.74 million tests" against a printed 1,738,998)
            if any(abs(p - value) <= tol for p in printed):
                continue
            if any(abs(p / 10 ** e - value) <= tol for p in printed for e in (3, 6)):
                continue
            bad.append(f"cell {k}: {tok}")
    return bad


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_prose_numbers_were_actually_computed(path, printed):
    bad = _unmatched(path, printed)
    assert not bad, (
        f"{path.name} states {len(bad)} number(s) that no notebook printed: {bad}. "
        "Either the prose is stale against a re-execution, or it quotes a figure computed by hand. "
        "Both are the defect this test exists to catch; if the number is legitimately external, "
        "put it in a code span."
    )


def test_the_check_would_have_caught_the_notebook_04_regression():
    """A guard on the guard: the matcher must not be so loose that it accepts anything."""
    printed_vals = [1.338, 0.508, 406.0, 1831.0]
    nb = {"cells": [{"cell_type": "markdown",
                     "source": ["best is 1.36 against 0.66, ranking 227 of 1,802"]}]}
    tmp = Path(__file__).parent / "_prose_fixture.ipynb"
    tmp.write_text(json.dumps(nb))
    try:
        bad = _unmatched(tmp, printed_vals)
    finally:
        tmp.unlink()
    assert len(bad) == 2, bad          # 1.36 and 0.66; the integers are outside this check's scope
