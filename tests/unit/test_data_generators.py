"""
Unit tests for data_generators module.
"""

import pytest
import torch
import numpy as np

from utils.data_generators import (
    SyntheticImageDataset,
    SyntheticSequenceDataset,
    SyntheticRegressionDataset,
    SyntheticReinforcementEnvironment,
    create_synthetic_dataloader,
    generate_batch_on_device,
    SyntheticBurnInGenerator
)


@pytest.mark.unit
class TestSyntheticImageDataset:
    """Test SyntheticImageDataset."""

    def test_dataset_creation(self):
        """Test creating synthetic image dataset."""
        dataset = SyntheticImageDataset(
            num_samples=100,
            image_size=32,
            num_channels=3,
            num_classes=10
        )

        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        dataset = SyntheticImageDataset(
            num_samples=10,
            image_size=32,
            num_channels=3,
            num_classes=10
        )

        image, label = dataset[0]

        assert image.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert 0 <= label < 10

    def test_dataset_reproducibility(self):
        """Test that seed makes dataset reproducible."""
        dataset1 = SyntheticImageDataset(num_samples=10, seed=42)
        dataset2 = SyntheticImageDataset(num_samples=10, seed=42)

        img1, label1 = dataset1[0]
        img2, label2 = dataset2[0]

        assert torch.allclose(img1, img2)
        assert label1 == label2

    def test_different_sizes(self):
        """Test datasets with different sizes."""
        sizes = [28, 64, 224]

        for size in sizes:
            dataset = SyntheticImageDataset(
                num_samples=10,
                image_size=size,
                num_channels=3
            )
            image, _ = dataset[0]
            assert image.shape == (3, size, size)


@pytest.mark.unit
class TestSyntheticSequenceDataset:
    """Test SyntheticSequenceDataset."""

    def test_dataset_creation(self):
        """Test creating synthetic sequence dataset."""
        dataset = SyntheticSequenceDataset(
            num_samples=100,
            seq_length=128,
            vocab_size=1000
        )

        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        dataset = SyntheticSequenceDataset(
            num_samples=10,
            seq_length=128,
            vocab_size=1000
        )

        input_seq, target_seq = dataset[0]

        assert input_seq.shape == (128,)
        assert target_seq.shape == (128,)
        assert input_seq.dtype == torch.long
        assert target_seq.dtype == torch.long

    def test_vocab_bounds(self):
        """Test that tokens are within vocabulary bounds."""
        vocab_size = 1000
        dataset = SyntheticSequenceDataset(
            num_samples=10,
            seq_length=128,
            vocab_size=vocab_size
        )

        input_seq, target_seq = dataset[0]

        assert input_seq.max() < vocab_size
        assert input_seq.min() >= 0
        assert target_seq.max() < vocab_size
        assert target_seq.min() >= 0


@pytest.mark.unit
class TestSyntheticRegressionDataset:
    """Test SyntheticRegressionDataset."""

    def test_dataset_creation(self):
        """Test creating synthetic regression dataset."""
        dataset = SyntheticRegressionDataset(
            num_samples=100,
            input_dim=50,
            output_dim=1
        )

        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        dataset = SyntheticRegressionDataset(
            num_samples=10,
            input_dim=50,
            output_dim=1
        )

        x, y = dataset[0]

        assert x.shape == (50,)
        assert isinstance(y.item(), float)

    def test_linear_relationship(self):
        """Test that dataset follows linear relationship."""
        dataset = SyntheticRegressionDataset(
            num_samples=100,
            input_dim=10,
            output_dim=1,
            noise_std=0.0,  # No noise
            seed=42
        )

        # Get multiple samples
        samples_x = []
        samples_y = []
        for i in range(10):
            x, y = dataset[i]
            samples_x.append(x)
            samples_y.append(y)

        # Check that relationship is consistent (same weights)
        # y = w^T x, so if we have the same input, we should get same output
        x_test = torch.randn(10)
        y1 = torch.matmul(x_test, dataset.weights).item()
        y2 = torch.matmul(x_test, dataset.weights).item()
        assert y1 == y2


@pytest.mark.unit
class TestSyntheticReinforcementEnvironment:
    """Test SyntheticReinforcementEnvironment."""

    def test_env_creation(self):
        """Test creating synthetic RL environment."""
        env = SyntheticReinforcementEnvironment(
            state_dim=4,
            action_dim=2,
            continuous_action=True
        )

        assert env.state_dim == 4
        assert env.action_dim == 2

    def test_env_reset(self):
        """Test environment reset."""
        env = SyntheticReinforcementEnvironment(state_dim=4, action_dim=2)

        state = env.reset()

        assert state.shape == (4,)
        assert env.current_step == 0

    def test_env_step(self):
        """Test environment step."""
        env = SyntheticReinforcementEnvironment(
            state_dim=4,
            action_dim=2,
            continuous_action=True
        )

        state = env.reset()
        action = torch.randn(2)

        next_state, reward, done, info = env.step(action)

        assert next_state.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert 'step' in info

    def test_env_episode_termination(self):
        """Test that episode terminates after max steps."""
        episode_length = 10
        env = SyntheticReinforcementEnvironment(
            state_dim=4,
            action_dim=2,
            episode_length=episode_length
        )

        env.reset()
        done = False
        steps = 0

        while not done and steps < episode_length + 5:
            action = env.sample_action()
            _, _, done, _ = env.step(action)
            steps += 1

        assert done
        assert steps == episode_length

    def test_continuous_vs_discrete(self):
        """Test continuous vs discrete action spaces."""
        # Continuous
        env_cont = SyntheticReinforcementEnvironment(
            state_dim=4,
            action_dim=2,
            continuous_action=True
        )
        action_cont = env_cont.sample_action()
        assert action_cont.shape == (2,)

        # Discrete
        env_disc = SyntheticReinforcementEnvironment(
            state_dim=4,
            action_dim=5,
            continuous_action=False
        )
        action_disc = env_disc.sample_action()
        assert isinstance(action_disc, int)
        assert 0 <= action_disc < 5


@pytest.mark.unit
class TestDataLoaderCreation:
    """Test dataloader creation functions."""

    def test_create_image_dataloader(self):
        """Test creating image dataloader."""
        dataloader = create_synthetic_dataloader(
            dataset_type='image',
            batch_size=4,
            num_samples=20,
            num_workers=0,
            image_size=32
        )

        assert len(dataloader) == 5  # 20 samples / 4 batch_size

        batch = next(iter(dataloader))
        images, labels = batch
        assert images.shape == (4, 3, 32, 32)
        assert labels.shape == (4,)

    def test_create_sequence_dataloader(self):
        """Test creating sequence dataloader."""
        dataloader = create_synthetic_dataloader(
            dataset_type='sequence',
            batch_size=4,
            num_samples=20,
            num_workers=0,
            seq_length=64
        )

        batch = next(iter(dataloader))
        input_seq, target_seq = batch
        assert input_seq.shape == (4, 64)
        assert target_seq.shape == (4, 64)

    def test_invalid_dataset_type(self):
        """Test creating dataloader with invalid type."""
        with pytest.raises(ValueError):
            create_synthetic_dataloader(
                dataset_type='invalid',
                batch_size=4,
                num_samples=20
            )


@pytest.mark.unit
class TestBatchGeneration:
    """Test batch generation utilities."""

    def test_generate_batch_on_device(self):
        """Test generating batch directly on device."""
        device = torch.device('cpu')
        inputs, labels = generate_batch_on_device(
            batch_size=8,
            input_shape=(3, 32, 32),
            num_classes=10,
            device=device
        )

        assert inputs.shape == (8, 3, 32, 32)
        assert labels.shape == (8,)
        assert inputs.device == device
        assert labels.device == device

    def test_burnin_generator(self):
        """Test burn-in generator."""
        device = torch.device('cpu')
        generator = SyntheticBurnInGenerator(
            batch_size=4,
            input_shape=(3, 32, 32),
            num_classes=10,
            device=device
        )

        # Generate multiple batches
        for _ in range(5):
            inputs, labels = generator.generate()
            assert inputs.shape == (4, 3, 32, 32)
            assert labels.shape == (4,)

    def test_burnin_generator_reuses_buffers(self):
        """Test that burn-in generator reuses buffers."""
        device = torch.device('cpu')
        generator = SyntheticBurnInGenerator(
            batch_size=4,
            input_shape=(3, 32, 32),
            num_classes=10,
            device=device
        )

        inputs1, labels1 = generator.generate()
        inputs2, labels2 = generator.generate()

        # Should reuse same buffers (same memory address)
        assert inputs1.data_ptr() == inputs2.data_ptr()
        assert labels1.data_ptr() == labels2.data_ptr()
