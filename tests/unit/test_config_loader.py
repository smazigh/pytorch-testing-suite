"""
Unit tests for config_loader module.
"""

import os
import pytest
import yaml
from pathlib import Path

from utils.config_loader import ConfigLoader, load_config


@pytest.mark.unit
class TestConfigLoader:
    """Test ConfigLoader class."""

    def test_load_config_from_file(self, config_file):
        """Test loading configuration from file."""
        config = ConfigLoader(str(config_file))
        assert config.config is not None
        assert 'general' in config.config
        assert 'gpu' in config.config

    def test_load_config_file_not_found(self, temp_dir):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader(str(temp_dir / 'nonexistent.yaml'))

    def test_get_value(self, config_file):
        """Test getting configuration values."""
        config = ConfigLoader(str(config_file))

        # Test simple key
        assert config.get('general.log_level') == 'INFO'

        # Test nested key
        assert config.get('training.batch_size') == 4

        # Test default value
        assert config.get('nonexistent.key', 'default') == 'default'

    def test_set_value(self, config_file):
        """Test setting configuration values."""
        config = ConfigLoader(str(config_file))

        # Set new value
        config.set('training.new_param', 123)
        assert config.get('training.new_param') == 123

        # Update existing value
        config.set('training.batch_size', 16)
        assert config.get('training.batch_size') == 16

    def test_update_config(self, config_file):
        """Test updating configuration with dictionary."""
        config = ConfigLoader(str(config_file))

        updates = {
            'training': {
                'batch_size': 32,
                'new_field': 'value'
            }
        }

        config.update(updates)
        assert config.get('training.batch_size') == 32
        assert config.get('training.new_field') == 'value'

    def test_env_variable_override(self, config_file, monkeypatch):
        """Test environment variable overrides."""
        # Set environment variable
        monkeypatch.setenv('PYTORCH_TEST_TRAINING_BATCH_SIZE', '128')

        config = ConfigLoader(str(config_file))
        assert config.get('training.batch_size') == 128

    def test_env_variable_type_conversion(self, config_file, monkeypatch):
        """Test environment variable type conversion."""
        # Boolean
        monkeypatch.setenv('PYTORCH_TEST_GPU_MIXED_PRECISION', 'true')
        config = ConfigLoader(str(config_file))
        assert config.get('gpu.mixed_precision') is True

        # Integer
        monkeypatch.setenv('PYTORCH_TEST_TRAINING_EPOCHS', '10')
        config = ConfigLoader(str(config_file))
        assert config.get('training.epochs') == 10

        # Float
        monkeypatch.setenv('PYTORCH_TEST_TRAINING_LEARNING_RATE', '0.01')
        config = ConfigLoader(str(config_file))
        assert config.get('training.learning_rate') == 0.01

    def test_save_config(self, config_file, temp_dir):
        """Test saving configuration to file."""
        config = ConfigLoader(str(config_file))

        # Modify config
        config.set('training.batch_size', 64)

        # Save to new file
        output_path = temp_dir / 'saved_config.yaml'
        config.save(str(output_path))

        # Load and verify
        with open(output_path, 'r') as f:
            saved_config = yaml.safe_load(f)

        assert saved_config['training']['batch_size'] == 64

    def test_dict_access(self, config_file):
        """Test dictionary-style access."""
        config = ConfigLoader(str(config_file))

        # Get
        assert config['general']['log_level'] == 'INFO'

        # Set
        config['training']['batch_size'] = 128
        assert config['training']['batch_size'] == 128

    def test_load_config_helper(self, config_file):
        """Test load_config helper function."""
        config = load_config(str(config_file))
        assert isinstance(config, ConfigLoader)
        assert config.config is not None


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config_structure(self, sample_config):
        """Test that sample config has valid structure."""
        assert 'general' in sample_config
        assert 'gpu' in sample_config
        assert 'training' in sample_config
        assert 'workloads' in sample_config

    def test_required_fields(self, config_file):
        """Test that required fields are present."""
        config = ConfigLoader(str(config_file))

        # General fields
        assert config.get('general.log_level') is not None
        assert config.get('general.output_dir') is not None

        # Training fields
        assert config.get('training.batch_size') is not None
        assert config.get('training.epochs') is not None
        assert config.get('training.learning_rate') is not None

    def test_default_values(self, config_file):
        """Test default value handling."""
        config = ConfigLoader(str(config_file))

        # Should return default for missing key
        assert config.get('missing.key', 'default_value') == 'default_value'

        # Should return None for missing key without default
        assert config.get('missing.key') is None


@pytest.mark.unit
class TestConfigLoaderAdvanced:
    """Advanced tests for config loader to improve coverage."""

    def test_contains_method(self, config_file):
        """Test __contains__ method."""
        config = ConfigLoader(str(config_file))

        assert 'general' in config
        assert 'nonexistent' not in config

    def test_iter_method(self, config_file):
        """Test __iter__ method."""
        config = ConfigLoader(str(config_file))

        keys = list(config)
        assert 'general' in keys
        assert 'training' in keys

    def test_len_method(self, config_file):
        """Test __len__ method."""
        config = ConfigLoader(str(config_file))

        # Should return number of top-level keys
        assert len(config) > 0

    def test_set_deep_nested_value(self, config_file):
        """Test setting deeply nested values."""
        config = ConfigLoader(str(config_file))

        # Set a deeply nested value
        config.set('workloads.cnn.advanced.nested.value', 123)
        assert config.get('workloads.cnn.advanced.nested.value') == 123

    def test_update_preserves_existing(self, config_file):
        """Test that update preserves existing nested keys."""
        config = ConfigLoader(str(config_file))

        original_lr = config.get('training.learning_rate')
        config.update({'training': {'new_key': 'new_value'}})

        # Original key should still exist
        assert config.get('training.learning_rate') == original_lr
        assert config.get('training.new_key') == 'new_value'

    def test_env_override_multiple_vars(self, config_file, monkeypatch):
        """Test multiple environment variable overrides."""
        monkeypatch.setenv('PYTORCH_TEST_TRAINING_BATCH_SIZE', '64')
        monkeypatch.setenv('PYTORCH_TEST_TRAINING_EPOCHS', '20')
        monkeypatch.setenv('PYTORCH_TEST_GENERAL_LOG_LEVEL', 'DEBUG')

        config = ConfigLoader(str(config_file))

        assert config.get('training.batch_size') == 64
        assert config.get('training.epochs') == 20
        assert config.get('general.log_level') == 'DEBUG'

    def test_get_entire_section(self, config_file):
        """Test getting an entire section."""
        config = ConfigLoader(str(config_file))

        training = config.get('training')
        assert isinstance(training, dict)
        assert 'batch_size' in training
        assert 'epochs' in training

    def test_empty_config(self, temp_dir):
        """Test loading empty config file."""
        empty_config = temp_dir / 'empty.yaml'
        with open(empty_config, 'w') as f:
            yaml.dump({}, f)

        config = ConfigLoader(str(empty_config))
        assert config.config == {}

    def test_list_values(self, temp_dir):
        """Test config with list values."""
        config_data = {
            'devices': [0, 1, 2],
            'operations': ['matmul', 'conv2d']
        }

        config_path = temp_dir / 'list_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = ConfigLoader(str(config_path))
        assert config.get('devices') == [0, 1, 2]
        assert 'matmul' in config.get('operations')

    def test_complex_nested_structure(self, temp_dir):
        """Test complex nested configuration structure."""
        config_data = {
            'model': {
                'architecture': {
                    'encoder': {
                        'layers': 6,
                        'hidden_dim': 512
                    },
                    'decoder': {
                        'layers': 6,
                        'hidden_dim': 512
                    }
                },
                'training': {
                    'optimizer': {
                        'type': 'adam',
                        'params': {
                            'lr': 0.001,
                            'betas': [0.9, 0.999]
                        }
                    }
                }
            }
        }

        config_path = temp_dir / 'complex_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = ConfigLoader(str(config_path))
        assert config.get('model.architecture.encoder.layers') == 6
        assert config.get('model.training.optimizer.params.lr') == 0.001

    def test_str_representation(self, config_file):
        """Test string representation of config."""
        config = ConfigLoader(str(config_file))
        str_repr = str(config)
        assert len(str_repr) > 0

    def test_repr_representation(self, config_file):
        """Test repr of config."""
        config = ConfigLoader(str(config_file))
        repr_str = repr(config)
        assert 'ConfigLoader' in repr_str
