"""
Synthetic data generators for PyTorch Testing Framework.
Generates data for various workloads without requiring external datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List, Any
import math


class SyntheticImageDataset(Dataset):
    """Generate synthetic image data for CNN training."""

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 224,
        num_channels: int = 3,
        num_classes: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic image dataset.

        Args:
            num_samples: Number of samples to generate
            image_size: Image height/width
            num_channels: Number of color channels
            num_classes: Number of classes
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a synthetic image and label.

        Returns:
            Tuple of (image, label)
        """
        # Generate random image
        image = torch.randn(self.num_channels, self.image_size, self.image_size)

        # Generate random label
        label = torch.randint(0, self.num_classes, (1,)).item()

        return image, label


class SyntheticSequenceDataset(Dataset):
    """Generate synthetic sequence data for transformer training."""

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 512,
        vocab_size: int = 30000,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic sequence dataset.

        Args:
            num_samples: Number of samples to generate
            seq_length: Sequence length
            vocab_size: Vocabulary size
            seed: Random seed
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a synthetic sequence and target.

        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        # Generate random input sequence
        input_seq = torch.randint(0, self.vocab_size, (self.seq_length,))

        # Generate target sequence (shifted by 1 for language modeling)
        target_seq = torch.randint(0, self.vocab_size, (self.seq_length,))

        return input_seq, target_seq


class SyntheticRegressionDataset(Dataset):
    """Generate synthetic data for regression tasks."""

    def __init__(
        self,
        num_samples: int = 10000,
        input_dim: int = 100,
        output_dim: int = 1,
        noise_std: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic regression dataset.

        Args:
            num_samples: Number of samples
            input_dim: Input dimension
            output_dim: Output dimension
            noise_std: Standard deviation of noise
            seed: Random seed
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_std = noise_std

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate fixed weights for consistency
        self.weights = torch.randn(input_dim, output_dim)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a synthetic regression sample.

        Returns:
            Tuple of (input, target)
        """
        # Generate input
        x = torch.randn(self.input_dim)

        # Generate target with noise
        y = torch.matmul(x, self.weights) + self.noise_std * torch.randn(self.output_dim)

        return x, y.squeeze()


class SyntheticReinforcementEnvironment:
    """Synthetic environment for reinforcement learning."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        continuous_action: bool = True,
        episode_length: int = 200,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic RL environment.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            continuous_action: Whether actions are continuous
            episode_length: Maximum episode length
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_action = continuous_action
        self.episode_length = episode_length
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        self.state = None
        self.reset()

    def reset(self) -> torch.Tensor:
        """
        Reset environment.

        Returns:
            Initial state
        """
        self.current_step = 0
        self.state = torch.randn(self.state_dim)
        return self.state.clone()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.current_step += 1

        # Synthetic dynamics: next_state = f(state, action) + noise
        action_effect = action.mean() if self.continuous_action else action
        self.state = 0.9 * self.state + 0.1 * action_effect + 0.1 * torch.randn(self.state_dim)

        # Synthetic reward: negative distance from origin
        reward = -torch.norm(self.state).item()

        # Episode termination
        done = self.current_step >= self.episode_length

        info = {'step': self.current_step}

        return self.state.clone(), reward, done, info

    def sample_action(self) -> torch.Tensor:
        """
        Sample a random action.

        Returns:
            Random action
        """
        if self.continuous_action:
            return torch.randn(self.action_dim)
        else:
            return torch.randint(0, self.action_dim, (1,)).item()


def create_synthetic_dataloader(
    dataset_type: str = "image",
    batch_size: int = 32,
    num_samples: int = 10000,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with synthetic data.

    Args:
        dataset_type: Type of dataset ('image', 'sequence', 'regression')
        batch_size: Batch size
        num_samples: Number of samples
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory
        **kwargs: Additional arguments for the dataset

    Returns:
        DataLoader instance
    """
    if dataset_type == "image":
        dataset = SyntheticImageDataset(num_samples=num_samples, **kwargs)
    elif dataset_type == "sequence":
        dataset = SyntheticSequenceDataset(num_samples=num_samples, **kwargs)
    elif dataset_type == "regression":
        dataset = SyntheticRegressionDataset(num_samples=num_samples, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    return dataloader


def estimate_dataloader_size(
    dataloader: DataLoader,
    samples: int = 10
) -> dict:
    """
    Estimate memory footprint of dataloader batches.

    Args:
        dataloader: DataLoader to analyze
        samples: Number of batches to sample

    Returns:
        Dictionary with size estimates
    """
    total_size = 0
    num_batches = 0

    for i, batch in enumerate(dataloader):
        if i >= samples:
            break

        # Calculate batch size
        batch_size = 0
        if isinstance(batch, (list, tuple)):
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_size += item.element_size() * item.nelement()
        elif isinstance(batch, torch.Tensor):
            batch_size = batch.element_size() * batch.nelement()

        total_size += batch_size
        num_batches += 1

    avg_size = total_size / num_batches if num_batches > 0 else 0

    return {
        'avg_batch_size_mb': avg_size / (1024 * 1024),
        'estimated_total_mb': avg_size * len(dataloader) / (1024 * 1024),
    }


class InfiniteDataLoader:
    """Wrapper for infinite data loading (useful for burn-in tests)."""

    def __init__(self, dataloader: DataLoader):
        """
        Initialize infinite dataloader.

        Args:
            dataloader: Base DataLoader to wrap
        """
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch, restarting if needed."""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


def generate_batch_on_device(
    batch_size: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch directly on device (fastest for burn-in).

    Args:
        batch_size: Batch size
        input_shape: Shape of input (excluding batch dimension)
        num_classes: Number of classes
        device: Device to generate on

    Returns:
        Tuple of (inputs, labels)
    """
    inputs = torch.randn(batch_size, *input_shape, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return inputs, labels


class SyntheticBurnInGenerator:
    """High-performance generator for GPU burn-in tests."""

    def __init__(
        self,
        batch_size: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        device: torch.device
    ):
        """
        Initialize burn-in generator.

        Args:
            batch_size: Batch size
            input_shape: Input shape (excluding batch)
            num_classes: Number of classes
            device: Device
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device

        # Pre-allocate buffers for maximum performance
        self.input_buffer = torch.empty(batch_size, *input_shape, device=device)
        self.label_buffer = torch.empty(batch_size, dtype=torch.long, device=device)

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch (reuses buffers).

        Returns:
            Tuple of (inputs, labels)
        """
        self.input_buffer.normal_()
        self.label_buffer.random_(0, self.num_classes)
        return self.input_buffer, self.label_buffer


if __name__ == "__main__":
    # Example usage
    print("Creating synthetic image dataloader...")
    image_loader = create_synthetic_dataloader(
        dataset_type="image",
        batch_size=32,
        num_samples=1000,
        image_size=224,
        num_classes=1000
    )

    print(f"DataLoader created with {len(image_loader)} batches")

    # Test a batch
    images, labels = next(iter(image_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

    # Estimate size
    size_info = estimate_dataloader_size(image_loader, samples=5)
    print(f"Average batch size: {size_info['avg_batch_size_mb']:.2f} MB")

    print("\nCreating synthetic sequence dataloader...")
    seq_loader = create_synthetic_dataloader(
        dataset_type="sequence",
        batch_size=16,
        num_samples=1000,
        seq_length=512,
        vocab_size=30000
    )

    input_seq, target_seq = next(iter(seq_loader))
    print(f"Sequence batch shape: {input_seq.shape}")

    print("\nCreating synthetic RL environment...")
    env = SyntheticReinforcementEnvironment(
        state_dim=4,
        action_dim=2,
        continuous_action=True
    )

    state = env.reset()
    print(f"Initial state: {state}")

    action = env.sample_action()
    next_state, reward, done, info = env.step(action)
    print(f"After step - Reward: {reward:.3f}, Done: {done}")
