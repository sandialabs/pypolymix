# Tests for the Pypolymix Library

## Run Tests
From the repository root run:

```uv run pytest```

This runs the unit tests only and takes a few seconds. Everything under
`regression_tests/` is skipped by default.

## Run the Regression Tests

The regression tests train models for tens of thousands of optimizer steps and
take the better part of an hour. Opt in with `--regression`, which runs the whole
suite:

```uv run pytest --regression```

To run only the regression tests:

```uv run pytest --regression -m regression```

## Run Tests with Coverage

To run the test suite with a terminal coverage summary

```uv run pytest --cov=pypolymix --cov-report=term-missing```

Add `--regression` for a coverage number covering the training code paths as
well; that is what CI reports.

## Development Notes
- Unit test structure follows the structure in src/pypolymix/
- Regression tests are in regression_tests/
- Regression tests take significantly longer to run than the standard unit tests
- Tests are marked `regression` automatically, by directory, in `conftest.py`.
  Adding a file to `regression_tests/` is all that is required; no decorator is
  needed on the test itself.

## CI/CD
- GitHub Actions is configured to run the tests on every push.
    - Workflow file: .github/workflows/tests.yml
    - The `test` job runs the unit tests on Python 3.10, 3.11 and 3.12.
    - The `regression` job runs the full suite with coverage. It is too slow for
      pull requests, so it runs nightly, on pushes to `main`, and on demand via
      "Run workflow" in the Actions tab.

- To view the coverage summary from GitHub Actions:
    1. Navigate to the actions tab of the GitHub repo
    2. Click on the relevant workflow run
    3. Click on the "Regression and coverage" job
    4. Expand the "Run all tests with coverage" step
</content>
