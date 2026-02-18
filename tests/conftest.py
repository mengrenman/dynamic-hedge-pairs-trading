# tests/conftest.py
# pairs.stats.stationarity exports a public function named test_spread_stationarity().
# When pytest imports the module, it sees this name and tries to collect it as a test,
# failing with "fixture 'spread' not found". This hook skips that false-positive.
import pytest


def pytest_collect_file(parent, file_path):
    # Don't collect from the production source tree
    if "pairs" in file_path.parts and file_path.suffix == ".py":
        return None
