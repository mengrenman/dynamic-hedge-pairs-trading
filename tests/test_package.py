"""Package-level tests: version metadata, lazy public API, packaged universes."""
import importlib.metadata as md

import pytest

import pairs


def test_version_matches_installed_distribution():
    try:
        expected = md.version("pairs-trading")  # distribution name, not import name
    except md.PackageNotFoundError:
        pytest.skip("pairs-trading is not installed; __version__ falls back to '0+unknown'")
    assert pairs.__version__ == expected
    assert pairs.__version__ != "0+unknown"


def test_public_api_names_resolve():
    optional = {"plot_pair_legs_with_trades"}  # needs matplotlib
    for name in pairs.__all__:
        if name == "__version__":
            continue
        try:
            obj = getattr(pairs, name)
        except ImportError:
            if name in optional:
                continue
            raise
        assert obj is not None, name


def test_summarize_walk_forward_is_exported():
    from pairs.validation import summarize_walk_forward
    from pairs.validation.walk_forward import summarize_walk_forward as direct

    assert "summarize_walk_forward" in pairs.__all__
    assert pairs.summarize_walk_forward is direct
    assert summarize_walk_forward is direct


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        pairs.not_a_public_symbol  # noqa: B018


def test_packaged_universes_load_clean():
    names = pairs.list_universes()
    assert {"spx", "ndx", "spx_ndx_combined"} <= set(names)

    u = pairs.load_universe("spx_ndx_combined")
    assert len(u) > 400
    assert u.is_unique
    assert (u == u.str.upper()).all()
    assert u.is_monotonic_increasing


def test_missing_universe_error_lists_available():
    with pytest.raises(FileNotFoundError, match="spx"):
        pairs.load_universe("does_not_exist")
