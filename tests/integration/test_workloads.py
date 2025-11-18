"""
Integration tests for workloads.
Tests actual training runs on CPU with small configurations.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestCNNTraining:
    """Integration tests for CNN training workload."""

    def test_cnn_training_imports(self):
        """Test that CNN training module can be imported."""
        from workloads.single_node import cnn_training
        assert hasattr(cnn_training, 'CNNTrainer')

    def test_cnn_trainer_initialization(self, config_file):
        """Test CNN trainer can be initialized."""
        from workloads.single_node.cnn_training import CNNTrainer

        trainer = CNNTrainer(config_path=str(config_file))
        assert trainer is not None
        assert trainer.device is not None

    def test_cnn_model_creation(self, config_file):
        """Test CNN model creation."""
        from workloads.single_node.cnn_training import CNNTrainer

        trainer = CNNTrainer(config_path=str(config_file))
        model = trainer.create_model()

        assert model is not None
        assert hasattr(model, 'forward')

    def test_cnn_dataloader_creation(self, config_file):
        """Test dataloader creation."""
        from workloads.single_node.cnn_training import CNNTrainer

        trainer = CNNTrainer(config_path=str(config_file))
        dataloader = trainer.create_dataloader()

        assert dataloader is not None
        assert len(dataloader) > 0

    @pytest.mark.slow
    def test_cnn_training_epoch(self, config_file, small_dataset_config, temp_dir):
        """Test running one training epoch."""
        import yaml
        from workloads.single_node.cnn_training import CNNTrainer

        # Create minimal config
        config_path = temp_dir / 'minimal_config.yaml'
        minimal_config = small_dataset_config.copy()
        minimal_config['training']['epochs'] = 1
        minimal_config['general']['output_dir'] = str(temp_dir)

        with open(config_path, 'w') as f:
            yaml.dump(minimal_config, f)

        trainer = CNNTrainer(config_path=str(config_path))
        model = trainer.create_model()
        dataloader = trainer.create_dataloader()

        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Run one epoch
        try:
            trainer.train_epoch(model, dataloader, optimizer, epoch=1)
            success = True
        except Exception as e:
            success = False
            print(f"Training failed: {e}")

        assert success


@pytest.mark.integration
class TestTransformerTraining:
    """Integration tests for Transformer training workload."""

    def test_transformer_training_imports(self):
        """Test that Transformer training module can be imported."""
        from workloads.single_node import transformer_training
        assert hasattr(transformer_training, 'TransformerTrainer')

    def test_transformer_trainer_initialization(self, config_file):
        """Test Transformer trainer can be initialized."""
        from workloads.single_node.transformer_training import TransformerTrainer

        trainer = TransformerTrainer(config_path=str(config_file))
        assert trainer is not None

    def test_transformer_model_creation(self, config_file):
        """Test Transformer model creation."""
        from workloads.single_node.transformer_training import TransformerTrainer

        trainer = TransformerTrainer(config_path=str(config_file))
        model = trainer.create_model()

        assert model is not None
        assert hasattr(model, 'forward')

    @pytest.mark.slow
    def test_transformer_training_epoch(self, config_file, small_dataset_config, temp_dir):
        """Test running one training epoch."""
        import yaml
        from workloads.single_node.transformer_training import TransformerTrainer

        # Create minimal config
        config_path = temp_dir / 'minimal_config.yaml'
        minimal_config = small_dataset_config.copy()
        minimal_config['training']['epochs'] = 1
        minimal_config['general']['output_dir'] = str(temp_dir)

        with open(config_path, 'w') as f:
            yaml.dump(minimal_config, f)

        trainer = TransformerTrainer(config_path=str(config_path))
        model = trainer.create_model()
        dataloader = trainer.create_dataloader()

        import torch.optim as optim
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)

        # Run one epoch
        try:
            trainer.train_epoch(model, dataloader, optimizer, epoch=1)
            success = True
        except Exception as e:
            success = False
            print(f"Training failed: {e}")

        assert success


@pytest.mark.integration
class TestGPUBurnIn:
    """Integration tests for GPU burn-in workload."""

    def test_burnin_imports(self):
        """Test that GPU burn-in module can be imported."""
        from workloads.single_node import gpu_burnin
        assert hasattr(gpu_burnin, 'GPUBurnIn')

    def test_burnin_initialization(self, config_file):
        """Test GPU burn-in can be initialized."""
        from workloads.single_node.gpu_burnin import GPUBurnIn

        burnin = GPUBurnIn(config_path=str(config_file))
        assert burnin is not None

    def test_matrix_multiply_stress(self):
        """Test matrix multiplication stress function."""
        from workloads.single_node.gpu_burnin import matrix_multiply_stress
        import torch

        device = torch.device('cpu')

        # Should not raise exception
        try:
            matrix_multiply_stress(size=128, iterations=2, device=device)
            success = True
        except Exception as e:
            success = False
            print(f"Matrix multiply stress failed: {e}")

        assert success

    def test_attention_stress(self):
        """Test attention stress function."""
        from workloads.single_node.gpu_burnin import attention_stress
        import torch

        device = torch.device('cpu')

        # Should not raise exception
        try:
            result = attention_stress(
                batch_size=2,
                seq_len=32,
                dim=64,
                device=device
            )
            success = True
        except Exception as e:
            success = False
            print(f"Attention stress failed: {e}")

        assert success


@pytest.mark.integration
class TestMixedPrecision:
    """Integration tests for mixed precision workload."""

    def test_mixed_precision_imports(self):
        """Test that mixed precision module can be imported."""
        from workloads.single_node import mixed_precision
        assert hasattr(mixed_precision, 'MixedPrecisionTrainer')

    def test_mixed_precision_initialization(self, config_file):
        """Test mixed precision trainer can be initialized."""
        from workloads.single_node.mixed_precision import MixedPrecisionTrainer

        trainer = MixedPrecisionTrainer(config_path=str(config_file))
        assert trainer is not None


@pytest.mark.integration
class TestReinforcementLearning:
    """Integration tests for RL workloads."""

    def test_ppo_imports(self):
        """Test that PPO module can be imported."""
        from workloads.reinforcement_learning import ppo_training
        assert hasattr(ppo_training, 'PPOTrainer')
        assert hasattr(ppo_training, 'ActorCritic')

    def test_ppo_trainer_initialization(self, config_file):
        """Test PPO trainer can be initialized."""
        from workloads.reinforcement_learning.ppo_training import PPOTrainer

        trainer = PPOTrainer(config_path=str(config_file))
        assert trainer is not None

    def test_actor_critic_model(self):
        """Test Actor-Critic model creation."""
        from workloads.reinforcement_learning.ppo_training import ActorCritic
        import torch

        model = ActorCritic(state_dim=4, action_dim=2, continuous=True)
        assert model is not None

        # Test forward pass
        state = torch.randn(1, 4)
        action_mean, action_std, value = model(state)

        assert action_mean.shape == (1, 2)
        assert action_std.shape == (2,)
        assert value.shape == (1, 1)

    def test_ppo_trajectory_collection(self, config_file):
        """Test PPO trajectory collection."""
        from workloads.reinforcement_learning.ppo_training import PPOTrainer

        trainer = PPOTrainer(config_path=str(config_file))
        env, model = trainer.create_env_and_model()

        # Should be able to collect trajectories
        assert env is not None
        assert model is not None


@pytest.mark.integration
@pytest.mark.distributed
class TestDistributedWorkloads:
    """Integration tests for distributed workloads."""

    def test_ddp_imports(self):
        """Test that DDP module can be imported."""
        from workloads.multi_node import ddp_training
        assert hasattr(ddp_training, 'DDPTrainer')

    def test_fsdp_imports(self):
        """Test that FSDP module can be imported."""
        from workloads.multi_node import fsdp_training
        assert hasattr(fsdp_training, 'FSDPTrainer')


@pytest.mark.integration
class TestEndToEndSmoke:
    """Smoke tests for end-to-end workload execution."""

    @pytest.mark.smoke
    def test_cnn_smoke(self, config_file, small_dataset_config, temp_dir):
        """Quick smoke test for CNN training."""
        import yaml
        from workloads.single_node.cnn_training import CNNTrainer

        # Ultra minimal config
        config_path = temp_dir / 'smoke_config.yaml'
        config = small_dataset_config.copy()
        config['training']['epochs'] = 1
        config['training']['batch_size'] = 2
        config['workloads']['cnn']['dataset_size'] = 4
        config['general']['output_dir'] = str(temp_dir)

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        trainer = CNNTrainer(config_path=str(config_path))

        try:
            # Just test initialization and one batch
            model = trainer.create_model()
            dataloader = trainer.create_dataloader()
            batch = next(iter(dataloader))
            success = True
        except Exception as e:
            success = False
            print(f"Smoke test failed: {e}")

        assert success

    @pytest.mark.smoke
    def test_all_workloads_importable(self):
        """Test that all workloads can be imported without errors."""
        workloads = [
            'workloads.single_node.cnn_training',
            'workloads.single_node.transformer_training',
            'workloads.single_node.gpu_burnin',
            'workloads.single_node.mixed_precision',
            'workloads.multi_node.ddp_training',
            'workloads.multi_node.fsdp_training',
            'workloads.reinforcement_learning.ppo_training'
        ]

        for workload in workloads:
            try:
                __import__(workload)
                success = True
            except ImportError as e:
                success = False
                print(f"Failed to import {workload}: {e}")

            assert success, f"Failed to import {workload}"
