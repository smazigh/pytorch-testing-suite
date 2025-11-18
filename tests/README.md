# PyTorch Testing Framework - Test Suite

Comprehensive testing suite for the PyTorch Testing Framework.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests for utilities
│   ├── test_config_loader.py
│   ├── test_data_generators.py
│   └── test_benchmark.py
├── integration/             # Integration tests for workloads
│   └── test_workloads.py
└── README.md               # This file
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements/test.txt
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=utils --cov=workloads

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -m unit

# Integration tests only
pytest tests/integration/ -m integration

# Smoke tests (quick validation)
pytest -m smoke

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Test config loader
pytest tests/unit/test_config_loader.py

# Test data generators
pytest tests/unit/test_data_generators.py

# Test workloads
pytest tests/integration/test_workloads.py
```

### Run Specific Tests

```bash
# Run a specific test function
pytest tests/unit/test_config_loader.py::TestConfigLoader::test_load_config_from_file

# Run tests matching a pattern
pytest -k "config"

# Run tests matching multiple patterns
pytest -k "config or data"
```

## Test Markers

Tests are organized with markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.gpu` - Tests requiring GPU (skipped if no GPU)
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.smoke` - Quick smoke tests
- `@pytest.mark.distributed` - Distributed training tests

### Using Markers

```bash
# Run only unit tests
pytest -m unit

# Run only smoke tests
pytest -m smoke

# Run integration but not slow
pytest -m "integration and not slow"

# Run GPU tests (if GPU available)
pytest -m gpu --rungpu
```

## Test Configuration

### Pytest Configuration

See `pytest.ini` for default pytest settings:

- Test discovery patterns
- Coverage settings
- Timeout configuration
- Marker definitions

### Test Fixtures

Common fixtures defined in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `sample_config` - Sample configuration dictionary
- `config_file` - Temporary config file
- `small_dataset_config` - Minimal config for quick tests
- `mock_gpu_available` - Mock GPU availability

### Environment Variables

Tests can be configured via environment variables:

```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

# Enable debug logging
export PYTEST_RUNNING=1
```

## Writing Tests

### Unit Test Example

```python
import pytest
from utils.config_loader import ConfigLoader

@pytest.mark.unit
class TestMyFeature:
    """Test my feature."""

    def test_basic_functionality(self, config_file):
        """Test basic functionality."""
        config = ConfigLoader(str(config_file))
        assert config is not None

    def test_edge_case(self):
        """Test edge case."""
        # Test code here
        pass
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
class TestWorkload:
    """Test workload execution."""

    def test_training_run(self, config_file, temp_dir):
        """Test running training."""
        from workloads.single_node.cnn_training import CNNTrainer

        trainer = CNNTrainer(config_path=str(config_file))
        model = trainer.create_model()
        # Test code here
```

## Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=utils --cov=workloads --cov-report=term-missing

# HTML report
pytest --cov=utils --cov=workloads --cov-report=html

# XML report (for CI/CD)
pytest --cov=utils --cov=workloads --cov-report=xml
```

### View HTML Coverage Report

```bash
pytest --cov=utils --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

Tests run automatically via GitHub Actions on:
- Push to main, develop, or claude/** branches
- Pull requests to main or develop

### CI/CD Workflow

The CI pipeline includes:

1. **Code Quality** - Black, isort, flake8 checks
2. **Unit Tests** - Run on Python 3.8, 3.9, 3.10, 3.11
3. **Integration Tests** - Smoke and full integration tests
4. **Import Tests** - Verify all modules can be imported
5. **Deployment Scripts** - Validate scripts and manifests
6. **Documentation** - Check README files exist

See `.github/workflows/tests.yml` for details.

## Troubleshooting

### Tests Fail Due to Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements/base.txt
pip install -r requirements/distributed.txt
pip install -r requirements/reinforcement-learning.txt
pip install -r requirements/test.txt
```

### Tests Timeout

```bash
# Increase timeout
pytest --timeout=600

# Or disable timeout for specific tests
pytest --timeout=0
```

### GPU Tests Fail

GPU tests are automatically skipped if no GPU is available. To run GPU tests:

```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Run GPU tests
pytest -m gpu --rungpu
```

### Slow Tests Take Too Long

```bash
# Skip slow tests
pytest -m "not slow"

# Or run only smoke tests
pytest -m smoke
```

### Import Errors

```bash
# Verify Python path
python -c "import sys; print(sys.path)"

# Run tests from repository root
cd /path/to/pytorch-test-suite
pytest
```

## Best Practices

1. **Run Tests Before Committing**
   ```bash
   pytest -m "unit and not slow"
   ```

2. **Use Fixtures** - Leverage conftest.py fixtures for common setup

3. **Mark Tests Appropriately** - Use markers to categorize tests

4. **Keep Tests Fast** - Mark slow tests with `@pytest.mark.slow`

5. **Test on CPU** - Most tests should work on CPU for CI/CD

6. **Use Small Configs** - Use `small_dataset_config` fixture for quick tests

7. **Mock When Needed** - Mock GPU availability, file I/O, etc.

8. **Test Edge Cases** - Include tests for error conditions

## Test Metrics

Current test coverage targets:
- **Unit Tests**: >80% coverage
- **Integration Tests**: All workloads importable and runnable
- **Overall**: >70% code coverage

## Contributing

When adding new features:

1. Write unit tests for utilities
2. Write integration tests for workloads
3. Add appropriate markers
4. Update this README if adding new test categories
5. Ensure tests pass locally before pushing
6. Check CI/CD passes after pushing

## Quick Reference

```bash
# Common test commands
pytest                              # Run all tests
pytest -m unit                      # Unit tests only
pytest -m smoke                     # Quick smoke tests
pytest -v                           # Verbose output
pytest -n auto                      # Parallel execution
pytest --cov                        # With coverage
pytest -k config                    # Tests matching "config"
pytest --lf                         # Last failed tests
pytest --ff                         # Failed first, then others
pytest -x                           # Stop on first failure
pytest --pdb                        # Drop into debugger on failure
```
