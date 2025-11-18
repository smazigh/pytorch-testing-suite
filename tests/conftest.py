"""
Pytest configuration and fixtures for PyTorch Testing Framework.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        'general': {
            'log_level': 'INFO',
            'output_dir': './results',
            'seed': 42,
            'benchmark_interval': 10
        },
        'gpu': {
            'device': 'cpu',  # Use CPU for tests
            'device_ids': [0],
            'mixed_precision': False,
            'cudnn_benchmark': True
        },
        'training': {
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.001,
            'num_workers': 0  # Avoid multiprocessing in tests
        },
        'workloads': {
            'cnn': {
                'model': 'resnet18',
                'image_size': 32,  # Small for tests
                'num_classes': 10,
                'dataset_size': 100
            },
            'transformer': {
                'vocab_size': 1000,
                'max_seq_length': 128,
                'model_dim': 256,
                'num_layers': 2,
                'num_heads': 4,
                'feedforward_dim': 512,
                'dataset_size': 100
            },
            'gpu_burnin': {
                'duration_minutes': 0.1,  # 6 seconds for tests
                'matrix_size': 512,
                'operations': ['matmul']
            }
        }
    }


@pytest.fixture
def config_file(temp_dir, sample_config):
    """Create a temporary config file."""
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def mock_gpu_available(monkeypatch):
    """Mock GPU availability."""
    def mock_is_available():
        return False

    monkeypatch.setattr('torch.cuda.is_available', mock_is_available)
    monkeypatch.setattr('torch.cuda.device_count', lambda: 0)


@pytest.fixture
def small_dataset_config(sample_config):
    """Config with very small dataset for quick tests."""
    config = sample_config.copy()
    config['training']['epochs'] = 1
    config['training']['batch_size'] = 2
    config['workloads']['cnn']['dataset_size'] = 10
    config['workloads']['transformer']['dataset_size'] = 10
    return config


@pytest.fixture(autouse=True)
def set_deterministic():
    """Set deterministic behavior for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def mock_logger(mocker):
    """Mock logger to avoid file I/O in tests."""
    return mocker.MagicMock()


def pytest_configure(config):
    """Pytest configuration hook."""
    # Set environment variable for testing
    os.environ['PYTEST_RUNNING'] = '1'

    # Use CPU by default for tests
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def pytest_unconfigure(config):
    """Pytest cleanup hook."""
    if 'PYTEST_RUNNING' in os.environ:
        del os.environ['PYTEST_RUNNING']


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Mark GPU tests
    for item in items:
        if 'gpu' in item.keywords:
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="GPU not available"))

        # Mark slow tests
        if 'slow' in item.keywords:
            if not config.getoption('--runslow', default=False):
                item.add_marker(pytest.mark.skip(reason="use --runslow to run"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--rungpu",
        action="store_true",
        default=False,
        help="run GPU tests"
    )
