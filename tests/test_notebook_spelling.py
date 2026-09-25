"""Notebook prose, comments, printed strings and outputs — and the package, tests and docs — use
American spelling.

The repository's standing rule is American English everywhere. On 2026-09-25 every committed
notebook, builder, package module, test and Markdown file was swept (127 spellings across 16
notebooks, another 76 elsewhere, and the last British identifiers renamed); this test keeps it that
way. It looks where prose lives — markdown outside code spans, code comments, string literals that
contain whitespace (print/f-strings, plot labels, column labels), matplotlib color names, and
stream/text outputs — and never at identifiers. Cited titles keep their own spelling (``CITED``).
"""
import io
import json
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("pairs_trading_*.ipynb"))
SOURCE_DIRS = ["pairs", "tests", "analysis", "webapp", "notebooks/build"]
PY_FILES = sorted(p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*.py")
                  if p.name != "test_notebook_spelling.py")           # this file's word list is British on purpose
MD_FILES = sorted(set(ROOT.glob("*.md")) | {p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*.md")})

# stems that take -ize/-ization in American English; no generic -ise rule because advise,
# comprise, premise and improvise are American as they stand
_IZE_STEMS = """optim normal real standard initial minim maxim recogn character emphas stabil penal random
serial winsor neutral util organ priorit capital summar visual general formal parametr parameter discret
regular linear symmetr orthogonal diagonal vector quant token memo synchron categor special material final
equal local global annual monet item custom author critic mobil hypothes theor scrutin sanit familiar
marginal internal external rational trivial ideal legitim popular polar symbol systemat digit central
decentral factor amort securit dollar unreal fertil sensit""".split()
_SUFFIXES = ["ise", "ised", "ises", "ising", "iser", "isers", "isation", "isations", "isable"]
_PAIRS = """analyse analysed analysing analyser catalyse paralyse centre centres centred centring recentre
recentred uncentred epicentre metre metres litre fibre theatre calibre colour colours coloured colouring
colourful colourmap behaviour behaviours behavioural favour favours favoured favouring favourable favourably
unfavourable favourite flavour honour honours honoured labour laboured neighbour neighbours neighbouring
neighbourhood harbour humour rumour vapour endeavour savour rigour vigour armour odour valour candour
splendour tumour defence defences offence offences licence licences pretence practise practised practising
programme programmes catalogue catalogues catalogued analogue analogues grey greys greyed greyish greyscale
whilst amongst learnt spelt dreamt modelling modelled modeller labelled labelling unlabelled mislabelled
relabelled signalling signalled cancelled cancelling travelling travelled traveller levelled levelling
channelled channelling totalled totalling equalled equalling fuelled fuelling marshalled marshalling
panelled pencilled tunnelled tunnelling counselled counselling dialled dialling initialled rivalled
jewellery woollen fulfil fulfils fulfilment enrol enrols enrolment instalment instalments instil distil
skilful wilful skilfully artefact artefacts judgement judgements misjudgement acknowledgement
acknowledgements manoeuvre manoeuvres manoeuvring aluminium sulphur sulphate sceptic sceptics sceptical
sceptically scepticism ageing enquiry enquiries enquire draught draughts maths cosy speciality specialities
focussed focussing focusses benefitted benefitting mould moulded plough tyre tyres encyclopaedia mediaeval
anaemia anaesthesia orthopaedic paediatric haemorrhage foetus oestrogen leukaemia aeroplane""".split()
BRITISH = {s + suf for s in _IZE_STEMS for suf in _SUFFIXES} | set(_PAIRS)
BRITISH -= {"analyses", "analysis", "premise", "premises", "storey"}   # American, or Storey's estimator

# cited titles and journal names keep their own spelling
CITED = ["Economic Modelling"]

WORD = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]+)(?![A-Za-z0-9_])")
GREY_COLOR = re.compile(r"""^[rRbBfFuU]{0,2}(['"])(light|dark|dim|slate)?grey\1$""")
FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)


def _british(text):
    for phrase in CITED:
        text = text.replace(phrase, " ")
    return [m.group(1) for m in WORD.finditer(text) if m.group(1).lower() in BRITISH]


def _prose_hits(cell):
    src = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        prose = re.sub(r"```.*?```|`[^`\n]*`", " ", src, flags=re.S)
        return _british(prose)
    if cell["cell_type"] != "code":
        return []
    shadow = re.sub(r"^(\s*)[%!]", r"\1#", src, flags=re.M)          # magics are not Python
    hits = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(shadow).readline):
            if t.type == tokenize.COMMENT or t.type == FSTRING_MIDDLE:
                hits += _british(t.string)
            elif t.type == tokenize.STRING:
                if GREY_COLOR.match(t.string.strip()):
                    hits.append("grey")
                elif re.search(r"\s", t.string.strip("rRbBfFuU'\"")):   # prose, not a key or column name
                    hits += _british(t.string)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        hits += _british(src)                                            # fall back to everything
    return hits


def _output_hits(cell):
    hits = []
    for out in cell.get("outputs", []):
        text = "".join(out.get("text", "")) if out.get("output_type") == "stream" else ""
        data = out.get("data", {}).get("text/plain", "")
        text += "".join(data) if isinstance(data, list) else data
        hits += _british(text)
    return hits


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem[14:] for p in NOTEBOOKS])
def test_notebook_uses_american_spelling(path):
    nb = json.loads(path.read_text())
    found = []
    for i, cell in enumerate(nb["cells"]):
        for w in _prose_hits(cell):
            found.append(f"cell {i} ({cell['cell_type']}): {w}")
        for w in _output_hits(cell):
            found.append(f"cell {i} (output): {w}")
    assert not found, f"{path.name}: British spellings in prose, strings or outputs:\n  " + "\n  ".join(found)


def _code_hits(src):
    return _prose_hits({"cell_type": "code", "source": src})


@pytest.mark.parametrize("path", PY_FILES, ids=[str(p.relative_to(ROOT)) for p in PY_FILES])
def test_python_prose_uses_american_spelling(path):
    """Comments, docstrings and prose strings; identifiers are never looked at."""
    found = _code_hits(path.read_text())
    assert not found, f"{path.relative_to(ROOT)}: British spellings in comments or strings: {found}"


@pytest.mark.parametrize("path", MD_FILES, ids=[str(p.relative_to(ROOT)) for p in MD_FILES])
def test_markdown_uses_american_spelling(path):
    """Prose outside inline code spans and fenced blocks."""
    found = _prose_hits({"cell_type": "markdown", "source": path.read_text()})
    assert not found, f"{path.relative_to(ROOT)}: British spellings in prose: {found}"
