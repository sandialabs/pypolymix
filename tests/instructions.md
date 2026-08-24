# Tests for the Pypolymix Library

## Run Tests
From the repository root run:

```uv run pytest```

## Run Tests with Coverage

To run the test suite with a terminal coverage summary

```uv run pytest --cov=pypolymix --cov-report=term-missing```

## Development Notes
- Unit test structure follows the structure in src/pypolymix/
- Regression tests are in regression_tests/
- Regression tests take significantly longer to run than the standard unit tests

## CI/CD
- GitHub Actions is configured to run the tests on every push.
    - Workflow file: .github/workflows/tests.yml

- To view the coverage summary from GitHub Actions:
    1. Navigate to the actions tab of the GitHub repo
    2. Click on the relevant workflow run
    3. Click on the Python 3.12 job
    4. Expand the "run tests with coverage" step
