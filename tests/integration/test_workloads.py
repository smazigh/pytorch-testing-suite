"""
Integration tests for workloads.
Tests actual training runs on CPU with small configurations.
"""

import pytest
import sys
import time
import torch
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.benchmark import PerformanceBenchmark, BenchmarkMetrics
from utils.config_loader import ConfigLoader, load_config
from utils.data_generators import (
    SyntheticImageDataset,
    SyntheticSequenceDataset,
    SyntheticRegressionDataset,
    SyntheticReinforcementEnvironment,
    create_synthetic_dataloader,
    generate_batch_on_device,
    SyntheticBurnInGenerator,
    InfiniteDataLoader,
    estimate_dataloader_size
)
from utils.logger import PerformanceLogger, get_logger, ColoredFormatter
from utils.gpu_monitor import GPUMonitor, GPUMetrics, check_gpu_availability


@pytest.mark.integration
class TestUtilsIntegration:
    """Integration tests for utility modules."""

    @pytest.mark.smoke
    def test_benchmark_full_workflow(self, temp_dir):
        """Test complete benchmark workflow."""
        benchmark = PerformanceBenchmark(
            name='integration_test',
            output_dir=str(temp_dir)
        )
        benchmark.configure(batch_size=16, sequence_length=64, model_flops=1e9)

        # Simulate training
        for epoch in range(2):
            benchmark.start_epoch()
            for i in range(5):
                benchmark.start_iteration()
                time.sleep(0.001)  # Small delay
                benchmark.end_iteration()
                benchmark.record_loss(1.0 - i * 0.1)
                benchmark.record_accuracy(0.5 + i * 0.1)
                benchmark.record_gpu_metrics(
                    utilization=50.0,
                    memory_used=4.0,
                    memory_allocated=3.5,
                    temperature=60.0
                )
            benchmark.end_epoch()

        # Get summary
        summary = benchmark.get_summary()
        assert summary['num_iterations'] == 10
        assert summary['num_epochs'] == 2
        assert 'avg_iteration_time' in summary
        assert 'final_loss' in summary
        assert 'final_accuracy' in summary

        # Test throughput calculation
        throughput = benchmark.get_throughput()
        assert throughput > 0

        # Save results
        benchmark.save_results(format='json')
        benchmark.save_results(format='csv')

        # Check files exist
        assert (temp_dir / 'integration_test_summary.json').exists()
        assert (temp_dir / 'integration_test_metrics.csv').exists()

    @pytest.mark.smoke
    def test_config_loader_full_workflow(self, temp_dir, sample_config):
        """Test complete config loader workflow."""
        # Create config file
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        # Load config
        config = ConfigLoader(str(config_path))

        # Test getting values
        assert config.get('general.log_level') == 'INFO'
        assert config.get('training.batch_size') == 4
        assert config.get('missing.key', 'default') == 'default'

        # Test setting values
        config.set('training.new_param', 123)
        assert config.get('training.new_param') == 123

        # Test updating
        config.update({'training': {'batch_size': 32}})
        assert config.get('training.batch_size') == 32

        # Test dict access
        assert config['general']['log_level'] == 'INFO'
        config['training']['test_value'] = 'test'
        assert config['training']['test_value'] == 'test'

        # Test saving
        output_path = temp_dir / 'saved_config.yaml'
        config.save(str(output_path))
        assert output_path.exists()

        # Load and verify
        loaded = load_config(str(output_path))
        assert loaded.get('training.batch_size') == 32

    @pytest.mark.smoke
    def test_data_generators_full_workflow(self):
        """Test data generators end-to-end."""
        # Image dataset
        image_ds = SyntheticImageDataset(num_samples=20, image_size=32, num_classes=10, seed=42)
        assert len(image_ds) == 20
        img, label = image_ds[0]
        assert img.shape == (3, 32, 32)
        assert 0 <= label < 10

        # Sequence dataset
        seq_ds = SyntheticSequenceDataset(num_samples=20, seq_length=64, vocab_size=1000, seed=42)
        assert len(seq_ds) == 20
        inp, tgt = seq_ds[0]
        assert inp.shape == (64,)
        assert tgt.shape == (64,)

        # Regression dataset
        reg_ds = SyntheticRegressionDataset(num_samples=20, input_dim=32, output_dim=1, seed=42)
        assert len(reg_ds) == 20
        x, y = reg_ds[0]
        assert x.shape == (32,)

        # RL environment
        env = SyntheticReinforcementEnvironment(state_dim=4, action_dim=2, episode_length=10)
        state = env.reset()
        assert state.shape == (4,)

        for _ in range(10):
            action = env.sample_action()
            next_state, reward, done, info = env.step(action)
            assert next_state.shape == (4,)
            if done:
                break

        # Dataloaders
        img_loader = create_synthetic_dataloader(
            dataset_type='image',
            batch_size=4,
            num_samples=16,
            num_workers=0,
            pin_memory=False,
            image_size=32
        )
        assert len(img_loader) == 4

        seq_loader = create_synthetic_dataloader(
            dataset_type='sequence',
            batch_size=4,
            num_samples=16,
            num_workers=0,
            pin_memory=False,
            seq_length=32
        )
        batch = next(iter(seq_loader))
        assert batch[0].shape == (4, 32)

        reg_loader = create_synthetic_dataloader(
            dataset_type='regression',
            batch_size=4,
            num_samples=16,
            num_workers=0,
            pin_memory=False,
            input_dim=16
        )
        batch = next(iter(reg_loader))
        assert batch[0].shape == (4, 16)

        # Generate batch on device
        device = torch.device('cpu')
        inputs, labels = generate_batch_on_device(8, (3, 32, 32), 10, device)
        assert inputs.shape == (8, 3, 32, 32)
        assert labels.shape == (8,)

        # Burn-in generator
        gen = SyntheticBurnInGenerator(4, (3, 32, 32), 10, device)
        inputs1, labels1 = gen.generate()
        inputs2, labels2 = gen.generate()
        assert inputs1.data_ptr() == inputs2.data_ptr()  # Reuses buffers

        # Infinite dataloader
        base_loader = create_synthetic_dataloader(
            dataset_type='image',
            batch_size=2,
            num_samples=4,
            num_workers=0,
            pin_memory=False
        )
        inf_loader = InfiniteDataLoader(base_loader)
        for i, batch in enumerate(inf_loader):
            if i >= 5:  # More than the dataset size
                break
        assert i == 5

        # Estimate dataloader size
        size_info = estimate_dataloader_size(img_loader, samples=2)
        assert 'avg_batch_size_mb' in size_info
        assert 'estimated_total_mb' in size_info

    @pytest.mark.smoke
    def test_logger_full_workflow(self, temp_dir):
        """Test logger end-to-end."""
        logger = get_logger(
            name='integration_test_logger',
            log_dir=str(temp_dir),
            level='DEBUG',
            rank=0
        )

        # Test all log levels
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')

        # Test headers and config logging
        logger.log_header('Test Header')
        logger.log_config({'param1': 'value1', 'nested': {'param2': 'value2'}})

        # Test metrics logging
        logger.log_metrics(
            epoch=1,
            iteration=10,
            metrics={'loss': 0.5, 'accuracy': 0.9},
            prefix='train'
        )

        # Test progress logging
        logger.log_progress(50, 100, metrics={'loss': 0.3}, prefix='Training')

        # Test system and GPU info logging
        logger.log_system_info({'python': '3.8', 'torch': '2.0'})
        logger.log_gpu_info({0: {'name': 'Test GPU', 'memory': '16GB'}})

        # Test iteration timing
        for _ in range(5):
            logger.start_iteration()
            time.sleep(0.001)
            logger.end_iteration()

        avg_time = logger.get_avg_iteration_time()
        assert avg_time > 0

        throughput = logger.get_throughput(batch_size=32)
        assert throughput > 0

        # Test progress bar
        pbar = logger.create_progress_bar(total=10, desc='Test')
        for i in range(10):
            pbar.update(1)
        pbar.close()

        # Test summary
        logger.log_summary({'final_loss': 0.1, 'epochs': 5})

        # Check metrics file was saved
        assert logger.metrics_file is not None

    @pytest.mark.smoke
    def test_gpu_monitor_workflow(self):
        """Test GPU monitor workflow."""
        # Check availability
        info = check_gpu_availability()
        assert 'cuda_available' in info
        assert 'num_gpus' in info

        # Create monitor
        monitor = GPUMonitor(device_ids=[0] if info['num_gpus'] > 0 else None)

        # Get metrics (will be None or actual metrics depending on GPU)
        if info['cuda_available']:
            metrics = monitor.get_metrics(0)
            if metrics:
                assert isinstance(metrics, GPUMetrics)
                assert metrics.memory_total >= 0
        else:
            # Just test it doesn't crash
            monitor.get_metrics(0)
            monitor.get_memory_summary(0)
            monitor.empty_cache()


@pytest.mark.integration
@pytest.mark.smoke
class TestBenchmarkMetrics:
    """Test BenchmarkMetrics in integration context."""

    def test_metrics_dataclass(self):
        """Test BenchmarkMetrics dataclass."""
        metrics = BenchmarkMetrics()
        metrics.iteration_times.append(0.1)
        metrics.losses.append(0.5)
        metrics.accuracies.append(0.9)
        metrics.gpu_utilization.append(80.0)
        metrics.gpu_memory_used.append(8.0)
        metrics.gpu_temperature.append(65.0)
        metrics.tflops.append(1.5)
        metrics.samples_per_sec.append(1000.0)
        metrics.tokens_per_sec.append(50000.0)
        metrics.epoch_times.append(30.0)

        assert len(metrics.iteration_times) == 1
        assert len(metrics.losses) == 1


@pytest.mark.integration
@pytest.mark.smoke
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
@pytest.mark.smoke
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
@pytest.mark.smoke
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
@pytest.mark.smoke
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
@pytest.mark.smoke
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
@pytest.mark.smoke
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


@pytest.mark.integration
@pytest.mark.smoke
class TestMixedPrecisionWorkload:
    """Additional tests for mixed precision workload."""

    def test_mixed_precision_model_creation(self, config_file):
        """Test mixed precision model creation."""
        from workloads.single_node.mixed_precision import MixedPrecisionTrainer

        trainer = MixedPrecisionTrainer(config_path=str(config_file))
        model = trainer.create_model()

        assert model is not None
        assert hasattr(model, 'forward')

    def test_mixed_precision_dataloader(self, config_file):
        """Test mixed precision dataloader creation."""
        from workloads.single_node.mixed_precision import MixedPrecisionTrainer

        trainer = MixedPrecisionTrainer(config_path=str(config_file))
        dataloader = trainer.create_dataloader()

        assert dataloader is not None
        assert len(dataloader) > 0




@pytest.mark.integration
@pytest.mark.smoke
class TestPPOWorkload:
    """Additional tests for PPO workload."""

    def test_ppo_discrete_action(self):
        """Test PPO with discrete action space."""
        from workloads.reinforcement_learning.ppo_training import ActorCritic

        model = ActorCritic(state_dim=8, action_dim=4, continuous=False)
        state = torch.randn(1, 8)
        action_logits, value = model(state)

        assert action_logits.shape == (1, 4)
        assert value.shape == (1, 1)

    def test_ppo_continuous_action(self):
        """Test PPO with continuous action space."""
        from workloads.reinforcement_learning.ppo_training import ActorCritic

        model = ActorCritic(state_dim=8, action_dim=4, continuous=True)
        state = torch.randn(1, 8)
        action_mean, action_std, value = model(state)

        assert action_mean.shape == (1, 4)
        assert action_std.shape == (4,)
        assert value.shape == (1, 1)


@pytest.mark.integration
@pytest.mark.smoke
class TestGPUBurnInWorkload:
    """Additional tests for GPU burn-in workload."""

    def test_burnin_model(self):
        """Test burn-in model creation."""
        from workloads.single_node.gpu_burnin import BurnInModel

        model = BurnInModel(channels=32, num_blocks=2)
        x = torch.randn(2, 3, 64, 64)
        output = model(x)

        assert output is not None
        assert output.shape[0] == 2


@pytest.mark.integration
@pytest.mark.smoke
class TestTransformerWorkload:
    """Additional tests for Transformer workload."""

    def test_transformer_dataloader(self, config_file):
        """Test transformer dataloader creation."""
        from workloads.single_node.transformer_training import TransformerTrainer

        trainer = TransformerTrainer(config_path=str(config_file))
        dataloader = trainer.create_dataloader()

        assert dataloader is not None
        assert len(dataloader) > 0

        # Get a batch
        batch = next(iter(dataloader))
        assert len(batch) == 2  # input and target

    def test_transformer_forward_pass(self, config_file):
        """Test transformer forward pass."""
        from workloads.single_node.transformer_training import TransformerTrainer

        trainer = TransformerTrainer(config_path=str(config_file))
        model = trainer.create_model()
        dataloader = trainer.create_dataloader()

        # Get a batch and do forward pass
        input_seq, target_seq = next(iter(dataloader))
        output = model(input_seq)

        assert output is not None
        assert output.shape[0] == input_seq.shape[0]


@pytest.mark.integration
@pytest.mark.smoke
class TestCNNWorkload:
    """Additional tests for CNN workload."""

    def test_cnn_forward_pass(self, config_file):
        """Test CNN forward pass."""
        from workloads.single_node.cnn_training import CNNTrainer

        trainer = CNNTrainer(config_path=str(config_file))
        model = trainer.create_model()
        dataloader = trainer.create_dataloader()

        # Get a batch and do forward pass
        images, labels = next(iter(dataloader))
        output = model(images)

        assert output is not None
        assert output.shape[0] == images.shape[0]

    def test_cnn_loss_computation(self, config_file):
        """Test CNN loss computation."""
        import torch.nn as nn
        from workloads.single_node.cnn_training import CNNTrainer

        trainer = CNNTrainer(config_path=str(config_file))
        model = trainer.create_model()
        dataloader = trainer.create_dataloader()

        # Get a batch and compute loss
        images, labels = next(iter(dataloader))
        output = model(images)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)

        assert loss is not None
        assert loss.item() > 0


@pytest.mark.integration
@pytest.mark.smoke
class TestMixedPrecisionExtended:
    """Extended tests for mixed precision training to improve coverage."""

    def test_mixed_precision_train_fp32(self, config_file, temp_dir):
        """Test FP32 training method."""
        import yaml
        from workloads.single_node.mixed_precision import MixedPrecisionTrainer

        # Create minimal config
        config_path = temp_dir / 'mp_config.yaml'
        with open(config_file) as f:
            config = yaml.safe_load(f)

        config['training']['batch_size'] = 2
        config['workloads']['cnn']['dataset_size'] = 8
        config['general']['output_dir'] = str(temp_dir)

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        trainer = MixedPrecisionTrainer(config_path=str(config_path))
        trainer.num_iterations = 2  # Very few iterations

        model = trainer.create_model()
        dataloader = trainer.create_dataloader()

        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Run FP32 training
        trainer.train_fp32(model, dataloader, optimizer)

        # Check metrics were recorded
        assert len(trainer.benchmark_fp32.metrics.iteration_times) == 2

    def test_mixed_precision_compare_results(self, config_file, temp_dir):
        """Test compare_results method."""
        import yaml
        from workloads.single_node.mixed_precision import MixedPrecisionTrainer

        config_path = temp_dir / 'mp_compare_config.yaml'
        with open(config_file) as f:
            config = yaml.safe_load(f)

        config['training']['batch_size'] = 2
        config['workloads']['cnn']['dataset_size'] = 4
        config['general']['output_dir'] = str(temp_dir)

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        trainer = MixedPrecisionTrainer(config_path=str(config_path))

        # Add some mock data to benchmarks
        trainer.benchmark_fp32.configure(batch_size=2)
        trainer.benchmark_amp.configure(batch_size=2)

        # Record some iteration times
        for _ in range(3):
            trainer.benchmark_fp32.start_iteration()
            trainer.benchmark_fp32.end_iteration()
            trainer.benchmark_amp.start_iteration()
            trainer.benchmark_amp.end_iteration()

        # This should not crash
        trainer.compare_results()


@pytest.mark.integration
@pytest.mark.smoke
class TestGPUBurnInExtended:
    """Extended tests for GPU burn-in to improve coverage."""

    def test_burnin_log_configuration(self, config_file):
        """Test _log_configuration method."""
        from workloads.single_node.gpu_burnin import GPUBurnIn

        burnin = GPUBurnIn(config_path=str(config_file))
        # _log_configuration is called in __init__, just verify it didn't crash
        assert burnin.logger is not None
        assert burnin.duration_minutes is not None

    def test_burnin_run_matmul(self, config_file, temp_dir):
        """Test run_matmul_burnin method."""
        import yaml
        from workloads.single_node.gpu_burnin import GPUBurnIn

        config_path = temp_dir / 'burnin_config.yaml'
        with open(config_file) as f:
            config = yaml.safe_load(f)

        config['workloads']['gpu_burnin'] = {
            'duration_minutes': 0.01,  # Very short
            'matrix_size': 64,
            'operations': ['matmul']
        }
        config['general']['output_dir'] = str(temp_dir)

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        burnin = GPUBurnIn(config_path=str(config_path))
        burnin.duration_minutes = 0.01  # Override to be very short
        burnin.matrix_size = 64

        # Run matmul burnin
        burnin.run_matmul_burnin()

        assert len(burnin.benchmark.metrics.iteration_times) > 0

    def test_burnin_run_attention(self, config_file, temp_dir):
        """Test run_attention_burnin method."""
        import yaml
        from workloads.single_node.gpu_burnin import GPUBurnIn

        config_path = temp_dir / 'burnin_attn_config.yaml'
        with open(config_file) as f:
            config = yaml.safe_load(f)

        config['workloads']['gpu_burnin'] = {
            'duration_minutes': 0.01,
            'matrix_size': 64,
            'operations': ['attention']
        }
        config['general']['output_dir'] = str(temp_dir)

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        burnin = GPUBurnIn(config_path=str(config_path))
        burnin.duration_minutes = 0.01

        # Run attention burnin
        burnin.run_attention_burnin()

        assert len(burnin.benchmark.metrics.iteration_times) > 0

    def test_burnin_model_forward(self):
        """Test BurnInModel forward pass with various inputs."""
        from workloads.single_node.gpu_burnin import BurnInModel

        model = BurnInModel(channels=16, num_blocks=2)

        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = model(x)
            assert output.shape == (batch_size, 1000)


@pytest.mark.integration
@pytest.mark.smoke
class TestPPOExtended:
    """Extended tests for PPO training to improve coverage."""

    def test_ppo_get_action_continuous(self):
        """Test get_action method for continuous actions."""
        from workloads.reinforcement_learning.ppo_training import ActorCritic

        model = ActorCritic(state_dim=8, action_dim=4, continuous=True)
        state = torch.randn(1, 8)

        action, log_prob, value = model.get_action(state)

        assert action.shape == (1, 4)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)

    def test_ppo_get_action_discrete(self):
        """Test get_action method for discrete actions."""
        from workloads.reinforcement_learning.ppo_training import ActorCritic

        model = ActorCritic(state_dim=8, action_dim=4, continuous=False)
        state = torch.randn(1, 8)

        action, log_prob, value = model.get_action(state)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)

    def test_ppo_compute_returns(self, config_file):
        """Test compute_returns method."""
        from workloads.reinforcement_learning.ppo_training import PPOTrainer

        trainer = PPOTrainer(config_path=str(config_file))

        # Create test data
        rewards = [1.0, 0.5, 0.2, 0.1, 0.0]
        values = [0.9, 0.5, 0.3, 0.1, 0.05]
        dones = [False, False, False, False, True]

        returns, advantages = trainer.compute_returns(rewards, values, dones)

        assert len(returns) == 5
        assert len(advantages) == 5
        assert returns.device == trainer.device

    def test_ppo_collect_trajectories(self, config_file):
        """Test collect_trajectories method."""
        from workloads.reinforcement_learning.ppo_training import PPOTrainer

        trainer = PPOTrainer(config_path=str(config_file))
        trainer.num_steps = 10  # Very few steps

        env, model = trainer.create_env_and_model()
        trajectory = trainer.collect_trajectories(env, model)

        assert 'states' in trajectory
        assert 'actions' in trajectory
        assert 'rewards' in trajectory
        assert len(trajectory['rewards']) == 10

    def test_ppo_trainer_log_configuration(self, config_file):
        """Test _log_configuration method."""
        from workloads.reinforcement_learning.ppo_training import PPOTrainer

        trainer = PPOTrainer(config_path=str(config_file))
        # _log_configuration is called in __init__
        assert trainer.num_steps > 0
        assert trainer.num_epochs > 0
        assert trainer.clip_epsilon > 0


@pytest.mark.integration
@pytest.mark.smoke
class TestConfigLoaderExtended:
    """Extended tests for config loader to improve coverage."""

    def test_config_deep_nesting(self, temp_dir):
        """Test deeply nested configuration."""
        import yaml

        config = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 42
                    }
                }
            }
        }

        config_path = temp_dir / 'nested_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        loader = ConfigLoader(str(config_path))

        assert loader.get('level1.level2.level3.value') == 42
        assert loader.get('level1.level2.missing', 'default') == 'default'

    def test_config_type_conversion(self, temp_dir):
        """Test various type conversions."""
        import yaml

        config = {
            'int_val': 42,
            'float_val': 3.14,
            'bool_val': True,
            'str_val': 'hello',
            'list_val': [1, 2, 3],
            'none_val': None
        }

        config_path = temp_dir / 'types_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        loader = ConfigLoader(str(config_path))

        assert isinstance(loader.get('int_val'), int)
        assert isinstance(loader.get('float_val'), float)
        assert isinstance(loader.get('bool_val'), bool)
        assert isinstance(loader.get('list_val'), list)

    def test_config_update_nested(self, temp_dir, sample_config):
        """Test updating nested configuration."""
        import yaml

        config_path = temp_dir / 'update_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        loader = ConfigLoader(str(config_path))

        # Update nested value
        loader.update({
            'training': {
                'batch_size': 128,
                'new_key': 'new_value'
            }
        })

        assert loader.get('training.batch_size') == 128
        assert loader.get('training.new_key') == 'new_value'
        # Original keys should be preserved
        assert loader.get('training.learning_rate') == 0.001


@pytest.mark.integration
@pytest.mark.smoke
class TestDataGeneratorsExtended:
    """Extended tests for data generators to improve coverage."""

    def test_rl_environment_discrete(self):
        """Test RL environment with discrete actions."""
        env = SyntheticReinforcementEnvironment(
            state_dim=4,
            action_dim=2,
            continuous_action=False,
            episode_length=10
        )

        state = env.reset()
        assert state.shape == (4,)

        # Take discrete action
        action = 0
        next_state, reward, done, info = env.step(action)
        assert next_state.shape == (4,)

    def test_rl_environment_multiple_episodes(self):
        """Test RL environment over multiple episodes."""
        env = SyntheticReinforcementEnvironment(
            state_dim=4,
            action_dim=2,
            episode_length=5,
            seed=42
        )

        total_steps = 0
        for episode in range(3):
            state = env.reset()
            done = False
            while not done:
                action = env.sample_action()
                state, reward, done, info = env.step(action)
                total_steps += 1

        assert total_steps == 15  # 3 episodes * 5 steps

    def test_burnin_generator_reuse(self):
        """Test SyntheticBurnInGenerator buffer reuse."""
        device = torch.device('cpu')
        gen = SyntheticBurnInGenerator(
            batch_size=2,
            input_shape=(3, 16, 16),
            num_classes=10,
            device=device
        )

        # Generate multiple times
        inputs1, labels1 = gen.generate()
        inputs2, labels2 = gen.generate()

        # Buffers should be reused
        assert inputs1.data_ptr() == inputs2.data_ptr()
        assert labels1.data_ptr() == labels2.data_ptr()

    def test_regression_dataset_output_dim(self):
        """Test SyntheticRegressionDataset with different output dims."""
        # Multi-output regression
        ds = SyntheticRegressionDataset(
            num_samples=10,
            input_dim=16,
            output_dim=5,
            seed=42
        )

        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (5,)


@pytest.mark.integration
@pytest.mark.smoke
class TestBenchmarkExtended:
    """Extended tests for benchmark utilities."""

    def test_benchmark_tflops_recording(self, temp_dir):
        """Test TFLOPS recording."""
        benchmark = PerformanceBenchmark(
            name='tflops_test',
            output_dir=str(temp_dir)
        )
        benchmark.configure(batch_size=16, model_flops=1e12)

        # Record some iterations
        for _ in range(3):
            benchmark.start_iteration()
            time.sleep(0.001)
            benchmark.end_iteration()

        summary = benchmark.get_summary()
        assert 'avg_iteration_time' in summary

    def test_benchmark_save_csv(self, temp_dir):
        """Test saving results to CSV."""
        benchmark = PerformanceBenchmark(
            name='csv_test',
            output_dir=str(temp_dir)
        )
        benchmark.configure(batch_size=8)

        # Record some data
        for i in range(5):
            benchmark.start_iteration()
            time.sleep(0.001)
            benchmark.end_iteration()
            benchmark.record_loss(1.0 - i * 0.1)

        # Save to CSV
        benchmark.save_results(format='csv')

        csv_file = temp_dir / 'csv_test_metrics.csv'
        assert csv_file.exists()

    def test_benchmark_epoch_tracking(self, temp_dir):
        """Test epoch tracking."""
        benchmark = PerformanceBenchmark(
            name='epoch_test',
            output_dir=str(temp_dir)
        )
        benchmark.configure(batch_size=8)

        # Run two epochs
        for epoch in range(2):
            benchmark.start_epoch()
            for _ in range(3):
                benchmark.start_iteration()
                time.sleep(0.001)
                benchmark.end_iteration()
            benchmark.end_epoch()

        summary = benchmark.get_summary()
        assert summary['num_epochs'] == 2
        assert summary['num_iterations'] == 6
