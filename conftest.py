# conftest.py — pytest configuration
# Prevent pytest from traversing into the pairs package source tree.
# This avoids false-positive collection of functions like `test_spread_stationarity`
# which are production API functions, not test functions.
collect_ignore_glob = ["pairs/**/*.py"]
