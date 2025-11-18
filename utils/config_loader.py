"""
Configuration loader for PyTorch Testing Framework.
Handles loading, validation, and merging of configuration files.
"""

import os
import yaml
from typing import Any, Dict, List, Optional
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            # Default to config/config.yaml in the repo root
            repo_root = Path(__file__).parent.parent
            config_path = repo_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Apply environment variable overrides
        config = self._apply_env_overrides(config)

        return config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        Environment variables follow the pattern: PYTORCH_TEST_<SECTION>_<KEY>
        Example: PYTORCH_TEST_TRAINING_BATCH_SIZE=128
        """
        prefix = "PYTORCH_TEST_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Parse environment variable
            key_parts = env_key[len(prefix):].lower().split('_')

            # Find the best match in config structure
            # Try different combinations of joining parts with underscores
            path = self._find_config_path(config, key_parts)

            if path:
                # Navigate to the correct config section
                current = config
                for part in path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the value with type conversion
                final_key = path[-1]
                current[final_key] = self._convert_type(env_value)

        return config

    def _find_config_path(self, config: Dict[str, Any], parts: List[str]) -> Optional[List[str]]:
        """
        Find the correct path in config by trying different combinations of joining parts.

        Args:
            config: Configuration dictionary
            parts: List of parts from environment variable name

        Returns:
            List of keys representing the path, or None if not found
        """
        if not parts:
            return []

        # Try different split points
        for i in range(1, len(parts) + 1):
            # Join first i parts as a potential key
            potential_key = '_'.join(parts[:i])

            if potential_key in config:
                if i == len(parts):
                    # This is the final key
                    return [potential_key]
                elif isinstance(config[potential_key], dict):
                    # Recursively find the rest of the path
                    rest = self._find_config_path(config[potential_key], parts[i:])
                    if rest is not None:
                        return [potential_key] + rest

        # If no match found in existing config, use the simple approach
        # (first part as section, rest joined as key)
        if len(parts) >= 2:
            return [parts[0], '_'.join(parts[1:])]

        return parts

    @staticmethod
    def _convert_type(value: str) -> Any:
        """Convert string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # None
        if value.lower() in ('none', 'null'):
            return None

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # List (comma-separated)
        if ',' in value:
            return [v.strip() for v in value.split(',')]

        # String
        return value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated path to config value (e.g., 'training.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            config.get('training.batch_size')  # Returns 64
            config.get('training.epochs', 10)  # Returns 10 if not set
        """
        keys = key_path.split('.')
        current = self.config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary of values.

        Args:
            updates: Dictionary of updates to apply
        """
        self._deep_update(self.config, updates)

    @staticmethod
    def _deep_update(base: Dict, updates: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                ConfigLoader._deep_update(base[key], value)
            else:
                base[key] = value

    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save config. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self.config[key] = value

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigLoader(config_path={self.config_path})"


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)


if __name__ == "__main__":
    # Example usage
    config = load_config()

    print("Configuration loaded successfully!")
    print(f"Batch size: {config.get('training.batch_size')}")
    print(f"Learning rate: {config.get('training.learning_rate')}")
    print(f"GPU device: {config.get('gpu.device')}")
    print(f"Mixed precision: {config.get('gpu.mixed_precision')}")
