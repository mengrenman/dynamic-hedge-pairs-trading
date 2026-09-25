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

Two checks sit on top of that loose one, because the loose one has a known blind spot: a stale
figure passes whenever it happens to round to *some* number printed elsewhere in the repo. Notebook
11 §4 carried "0.40 against 0.28" against an actual 0.402/0.315 for months — "0.28" is printed, for
unrelated quantities, by eight other notebooks, and that was enough.

* `test_compared_numbers_come_from_the_same_notebook` — when prose immediately below a code cell
  compares two numbers ("X against Y", "X vs Y"), both halves came off the same table, so both must
  appear in *this* notebook's outputs. One local and one not is drift, and cross-notebook citation
  is still allowed everywhere else. This is narrow on purpose: a blanket locality rule fires on
  legitimate prose arithmetic (nb01 §8.1 derives s.e. = √(252/bars) inline, nb08 §4 quotes a
  difference of two printed Sharpes), which is why it is scoped to comparisons.
* comma-grouped integers are now in scope. The same nb11 sentence also said "1,707 round trips"
  against a printed 1,698, and integers were not looked at at all. Tolerance follows significant
  figures — "44,500" is satisfied by 44,512, "1,707" is not satisfied by 1,698 — and an explicitly
  hedged figure ("roughly 290,000") is prose, not a citation, so it is skipped.
"""
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("pairs_trading_*.ipynb"))

# A comma inside a number is part of it: without the "," in the lookbehind, "\$2,074.58" tokenizes
# as the meaningless "074.58" and then passes by matching some unrelated 74.58.
NUM = r"\d+(?:,\d{3})*\.\d{2,4}"
PROSE_NUM = re.compile(rf"(?<![\w.$/,-])({NUM})(?![\d/])")
GROUPED_INT = re.compile(r"(?<![\w.$/,-])(\d{1,3}(?:,\d{3})+)(?![\d.,])")
DOI = re.compile(r"10\.\d{4,}/\S+")
# "roughly 290,000 minute observations" is an order-of-magnitude aside, not a quoted result.
HEDGE = re.compile(r"\b(?:roughly|about|around|approximately|nearly|some|order of)\s+(?:\w+\s+){0,2}$|~\s*$")
# One comparison, inside one sentence. The gap excludes sentence enders so two numbers from
# different claims are never paired up.
JOIN = r"(?:against|vs\.?|versus|compared\s+(?:with|to)|rather\s+than|not|and)"
COMPARISON = re.compile(
    rf"(?<![\w.$/,-])({NUM})(?![\d/])[^.!?]{{0,40}}?\b{JOIN}\b[^.!?]{{0,40}}?(?<![\w.$/,-])({NUM})(?![\d/])"
)


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


def _numbers_printed_by(path):
    blob = " ".join(_outputs(json.loads(path.read_text())))
    flat = blob.replace(",", "")
    vals = [float(m) for m in re.findall(r"(?<![\w.])(\d+\.\d+)", flat)]
    vals += [float(m) for m in re.findall(r"(?<![\w.])(\d{4,})(?![\d.])", flat)]
    return vals


@pytest.fixture(scope="module")
def printed_by_notebook():
    return {path: _numbers_printed_by(path) for path in NOTEBOOKS}


@pytest.fixture(scope="module")
def printed(printed_by_notebook):
    """Every number any notebook printed, as floats."""
    return [v for vals in printed_by_notebook.values() for v in vals]


def _tolerance(tok):
    """How far a printed value may sit from the prose, given how precisely the prose quoted it.

    The 1e-9 slack is not cosmetic. A prose number that is the exact half-up rounding of a printed
    one — 0.197 from 0.1965, 0.32 from 0.315 — lands precisely on the tolerance, where binary
    floating point puts it a few ulps over and the match fails. Those are correct roundings.
    """
    if "." in tok:
        return 0.5 * 10 ** -len(tok.split(".")[1]) * (1 + 1e-9)
    digits = tok.replace(",", "")                       # significant figures: 44,500 admits 44,512
    return 0.5 * 10 ** (len(digits) - len(digits.rstrip("0"))) * (1 + 1e-9)


def _matches(tok, pool):
    value, tol = float(tok.replace(",", "")), _tolerance(tok)
    if any(abs(p - value) <= tol for p in pool):
        return True
    # printed at higher precision and abbreviated to thousands/millions in the prose
    # ("1.74 million tests" against a printed 1,738,998)
    return any(abs(p / 10 ** e - value) <= tol for p in pool for e in (3, 6))


def _prose_cells(nb):
    """Markdown cells, flagged with whether a code cell with output sits immediately above."""
    after_output = False
    for k, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            after_output = bool(cell.get("outputs"))
        elif cell["cell_type"] == "markdown":
            yield k, "".join(cell["source"]), after_output
            after_output = False


def _strip(raw):
    return DOI.sub(" ", re.sub(r"`[^`]*`", " ", raw))


def _unmatched(path, printed):
    nb = json.loads(path.read_text())
    bad = []
    for k, raw, _ in _prose_cells(nb):
        md = _strip(raw)
        toks = [(m.group(1), m.start(1)) for m in PROSE_NUM.finditer(md)]
        toks += [(m.group(1), m.start(1)) for m in GROUPED_INT.finditer(md)
                 if not HEDGE.search(md[:m.start(1)])]
        for tok, _pos in toks:
            if not _matches(tok, printed):
                bad.append(f"cell {k}: {tok}")
    return bad


def _mismatched_comparisons(path, own):
    nb = json.loads(path.read_text())
    bad = []
    for k, raw, after_output in _prose_cells(nb):
        if not after_output:
            continue
        md = re.sub(r"\s+", " ", _strip(raw))
        for left, right in COMPARISON.findall(md):
            hits = _matches(left, own), _matches(right, own)
            if hits[0] != hits[1]:
                stale = right if hits[0] else left
                bad.append(f"cell {k}: '{left}' against '{right}' — {stale} is not in this notebook")
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


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_compared_numbers_come_from_the_same_notebook(path, printed_by_notebook):
    bad = _mismatched_comparisons(path, printed_by_notebook[path])
    assert not bad, (
        f"{path.name} compares {len(bad)} pair(s) of numbers where one side was printed by this "
        f"notebook and the other was not: {bad}. Two numbers compared directly under a code cell "
        "come off the same output; if one of them does not, it is almost certainly stale from an "
        "earlier run and survived the repo-wide check only by matching an unrelated figure "
        "elsewhere. If the comparison really is against another notebook, quote that figure in a "
        "code span."
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
    # 1.36, 0.66, and the grouped 1,802 against a printed 1,831; bare "227" stays out of scope
    assert len(bad) == 3, bad


def test_grouped_integers_allow_rounding_but_not_drift():
    printed_vals = [44512.0, 1698.0]
    nb = {"cells": [{"cell_type": "markdown", "source": [
        "44,500 edges, roughly 290,000 bars, and 1,707 round trips"]}]}
    tmp = Path(__file__).parent / "_prose_fixture_int.ipynb"
    tmp.write_text(json.dumps(nb))
    try:
        bad = _unmatched(tmp, printed_vals)
    finally:
        tmp.unlink()
    assert bad == ["cell 0: 1,707"], bad     # 44,500 rounds; 290,000 is hedged; 1,707 is drift


def test_a_comparison_is_flagged_when_only_one_side_is_local():
    """The nb11 §4 case: 0.40 came off the table above, 0.28 was left over from an earlier run."""
    nb = {"cells": [
        {"cell_type": "code", "source": ["display(vt)"],
         "outputs": [{"output_type": "stream", "text": ["0.402  0.315"]}]},
        {"cell_type": "markdown", "source": ["the regression beats the filter, 0.40 against 0.28"]},
        {"cell_type": "markdown", "source": ["elsewhere, 0.40 against 0.28 is only a citation"]},
    ]}
    tmp = Path(__file__).parent / "_prose_fixture_cmp.ipynb"
    tmp.write_text(json.dumps(nb))
    try:
        bad = _mismatched_comparisons(tmp, [0.402, 0.315])
    finally:
        tmp.unlink()
    assert len(bad) == 1 and "cell 1" in bad[0] and "0.28" in bad[0], bad
