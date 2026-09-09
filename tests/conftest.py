"""Shared pytest configuration.

Everything under ``tests/regression_tests/`` trains models for tens of thousands
of optimizer steps and takes the better part of an hour. Those tests are marked
``regression`` automatically, by path, and skipped unless ``--regression`` is
passed:

```bash
pytest tests/                # unit tests only, a few seconds
pytest tests/ --regression   # everything
```

Marking by path means the test modules themselves need no decorators, and any
file added to that directory later is covered without further changes.
"""

import pytest

REGRESSION_DIR = "regression_tests"


def pytest_addoption(parser):
    """Register the ``--regression`` opt-in flag."""
    parser.addoption(
        "--regression",
        action="store_true",
        default=False,
        help="run the slow regression tests in tests/regression_tests/",
    )


def pytest_collection_modifyitems(config, items):
    """Mark regression tests and skip them unless they were explicitly requested."""
    run_regression = config.getoption("--regression")
    skip_regression = pytest.mark.skip(reason="slow; pass --regression to run")
    for item in items:
        if REGRESSION_DIR in item.nodeid:
            item.add_marker(pytest.mark.regression)
            if not run_regression:
                item.add_marker(skip_regression)
