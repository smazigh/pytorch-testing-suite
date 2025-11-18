"""
Unit tests for distributed training modules (DDP and FSDP).
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import os


@pytest.mark.unit
class TestDDPSetupFunctions:
    """Test DDP setup and cleanup functions."""

    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=2)
    @patch('torch.cuda.set_device')
    @patch.dict(os.environ, {'LOCAL_RANK': '0'})
    def test_setup_distributed(self, mock_set_device, mock_world_size, mock_rank, mock_init):
        """Test setup_distributed function."""
        from workloads.multi_node.ddp_training import setup_distributed

        rank, world_size, local_rank = setup_distributed()

        mock_init.assert_called_once_with(backend='nccl')
        assert rank == 0
        assert world_size == 2
        assert local_rank == 0
        mock_set_device.assert_called_once_with(0)

    @patch('torch.distributed.destroy_process_group')
    def test_cleanup_distributed(self, mock_destroy):
        """Test cleanup_distributed function."""
        from workloads.multi_node.ddp_training import cleanup_distributed

        cleanup_distributed()
        mock_destroy.assert_called_once()


@pytest.mark.unit
class TestFSDPSetupFunctions:
    """Test FSDP setup and cleanup functions."""

    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.get_rank', return_value=1)
    @patch('torch.distributed.get_world_size', return_value=4)
    @patch('torch.cuda.set_device')
    @patch.dict(os.environ, {'LOCAL_RANK': '1'})
    def test_fsdp_setup_distributed(self, mock_set_device, mock_world_size, mock_rank, mock_init):
        """Test FSDP setup_distributed function."""
        from workloads.multi_node.fsdp_training import setup_distributed

        rank, world_size, local_rank = setup_distributed()

        mock_init.assert_called_once_with(backend='nccl')
        assert rank == 1
        assert world_size == 4
        assert local_rank == 1

    @patch('torch.distributed.destroy_process_group')
    def test_fsdp_cleanup_distributed(self, mock_destroy):
        """Test FSDP cleanup_distributed function."""
        from workloads.multi_node.fsdp_training import cleanup_distributed

        cleanup_distributed()
        mock_destroy.assert_called_once()


@pytest.mark.unit
class TestDDPTrainerCreation:
    """Test DDPTrainer initialization and model creation."""

    @patch('workloads.multi_node.ddp_training.setup_distributed')
    @patch('workloads.multi_node.ddp_training.load_config')
    @patch('workloads.multi_node.ddp_training.get_logger')
    @patch('workloads.multi_node.ddp_training.GPUMonitor')
    @patch('workloads.multi_node.ddp_training.PerformanceBenchmark')
    def test_ddp_trainer_init_rank_0(self, mock_bench, mock_monitor, mock_logger,
                                      mock_config, mock_setup):
        """Test DDPTrainer initialization for rank 0."""
        # Setup mocks
        mock_setup.return_value = (0, 2, 0)  # rank, world_size, local_rank

        mock_config_instance = MagicMock()
        mock_config_instance.get.side_effect = lambda key, default=None: {
            'general.output_dir': './results',
            'general.log_level': 'INFO',
            'workloads.ddp.model': 'resnet50',
            'workloads.cnn.num_classes': 1000,
            'workloads.cnn.image_size': 224,
            'training.batch_size': 64,
            'training.epochs': 10,
            'training.learning_rate': 0.001,
            'gpu.mixed_precision': True,
        }.get(key, default)
        mock_config.return_value = mock_config_instance

        from workloads.multi_node.ddp_training import DDPTrainer

        trainer = DDPTrainer(config_path='test.yaml')

        assert trainer.rank == 0
        assert trainer.world_size == 2
        assert trainer.local_rank == 0
        assert trainer.model_name == 'resnet50'
        assert trainer.batch_size == 64
        mock_bench.assert_called_once()

    @patch('workloads.multi_node.ddp_training.setup_distributed')
    @patch('workloads.multi_node.ddp_training.load_config')
    @patch('workloads.multi_node.ddp_training.get_logger')
    @patch('workloads.multi_node.ddp_training.GPUMonitor')
    @patch('workloads.multi_node.ddp_training.PerformanceBenchmark')
    def test_ddp_trainer_init_rank_1(self, mock_bench, mock_monitor, mock_logger,
                                      mock_config, mock_setup):
        """Test DDPTrainer initialization for non-zero rank."""
        mock_setup.return_value = (1, 2, 1)  # rank 1

        mock_config_instance = MagicMock()
        mock_config_instance.get.return_value = None
        mock_config.return_value = mock_config_instance

        from workloads.multi_node.ddp_training import DDPTrainer

        trainer = DDPTrainer()

        assert trainer.rank == 1
        # Benchmark should be None for non-rank-0
        assert trainer.benchmark is None


@pytest.mark.unit
class TestFSDPTrainerCreation:
    """Test FSDPTrainer initialization."""

    @patch('workloads.multi_node.fsdp_training.setup_distributed')
    @patch('workloads.multi_node.fsdp_training.load_config')
    @patch('workloads.multi_node.fsdp_training.get_logger')
    @patch('workloads.multi_node.fsdp_training.GPUMonitor')
    @patch('workloads.multi_node.fsdp_training.PerformanceBenchmark')
    def test_fsdp_trainer_init(self, mock_bench, mock_monitor, mock_logger,
                                mock_config, mock_setup):
        """Test FSDPTrainer initialization."""
        mock_setup.return_value = (0, 4, 0)

        mock_config_instance = MagicMock()
        mock_config_instance.get.side_effect = lambda key, default=None: {
            'general.output_dir': './results',
            'general.log_level': 'INFO',
            'workloads.fsdp.model': 'resnet101',
            'training.batch_size': 32,
            'training.epochs': 5,
            'training.learning_rate': 0.001,
        }.get(key, default)
        mock_config.return_value = mock_config_instance

        from workloads.multi_node.fsdp_training import FSDPTrainer

        trainer = FSDPTrainer()

        assert trainer.rank == 0
        assert trainer.world_size == 4
        assert trainer.model_name == 'resnet101'
        assert trainer.batch_size == 32


@pytest.mark.unit
class TestDDPModelCreation:
    """Test DDP model creation methods."""

    @patch('workloads.multi_node.ddp_training.setup_distributed')
    @patch('workloads.multi_node.ddp_training.load_config')
    @patch('workloads.multi_node.ddp_training.get_logger')
    @patch('workloads.multi_node.ddp_training.GPUMonitor')
    @patch('workloads.multi_node.ddp_training.PerformanceBenchmark')
    @patch('torch.cuda.is_available', return_value=False)
    def test_ddp_model_types(self, mock_cuda, mock_bench, mock_monitor, mock_logger,
                              mock_config, mock_setup):
        """Test DDPTrainer supports different model types."""
        mock_setup.return_value = (0, 2, 0)

        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        from workloads.multi_node.ddp_training import DDPTrainer

        # Test different model names
        for model_name in ['resnet18', 'resnet50', 'resnet101', 'unknown']:
            mock_config_instance.get.side_effect = lambda key, default=None, mn=model_name: {
                'workloads.ddp.model': mn,
                'workloads.cnn.num_classes': 100,
                'distributed.find_unused_parameters': False,
                'distributed.gradient_as_bucket_view': True,
            }.get(key, default)

            trainer = DDPTrainer()
            assert trainer.model_name == model_name


@pytest.mark.unit
class TestFSDPModelCreation:
    """Test FSDP model creation methods."""

    @patch('workloads.multi_node.fsdp_training.setup_distributed')
    @patch('workloads.multi_node.fsdp_training.load_config')
    @patch('workloads.multi_node.fsdp_training.get_logger')
    @patch('workloads.multi_node.fsdp_training.GPUMonitor')
    @patch('workloads.multi_node.fsdp_training.PerformanceBenchmark')
    def test_fsdp_model_types(self, mock_bench, mock_monitor, mock_logger,
                               mock_config, mock_setup):
        """Test FSDPTrainer supports different model types."""
        mock_setup.return_value = (0, 2, 0)

        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        from workloads.multi_node.fsdp_training import FSDPTrainer

        for model_name in ['resnet50', 'resnet101', 'unknown']:
            mock_config_instance.get.side_effect = lambda key, default=None, mn=model_name: {
                'workloads.fsdp.model': mn,
            }.get(key, default)

            trainer = FSDPTrainer()
            assert trainer.model_name == model_name


@pytest.mark.unit
class TestDDPDataLoader:
    """Test DDP dataloader creation."""

    @patch('workloads.multi_node.ddp_training.setup_distributed')
    @patch('workloads.multi_node.ddp_training.load_config')
    @patch('workloads.multi_node.ddp_training.get_logger')
    @patch('workloads.multi_node.ddp_training.GPUMonitor')
    @patch('workloads.multi_node.ddp_training.PerformanceBenchmark')
    def test_ddp_dataloader_creation(self, mock_bench, mock_monitor, mock_logger,
                                      mock_config, mock_setup):
        """Test DDPTrainer.create_dataloader method."""
        mock_setup.return_value = (0, 2, 0)

        mock_config_instance = MagicMock()
        mock_config_instance.get.side_effect = lambda key, default=None: {
            'workloads.cnn.dataset_size': 1000,
            'workloads.cnn.image_size': 32,
            'workloads.cnn.num_classes': 10,
            'general.seed': 42,
            'training.num_workers': 0,
            'training.pin_memory': False,
            'training.batch_size': 8,
        }.get(key, default)
        mock_config.return_value = mock_config_instance

        from workloads.multi_node.ddp_training import DDPTrainer

        trainer = DDPTrainer()
        trainer.image_size = 32
        trainer.num_classes = 10
        trainer.batch_size = 8

        dataloader, sampler = trainer.create_dataloader()

        assert dataloader is not None
        assert sampler is not None
        assert len(dataloader) > 0


@pytest.mark.unit
class TestFSDPDataLoader:
    """Test FSDP dataloader creation."""

    @patch('workloads.multi_node.fsdp_training.setup_distributed')
    @patch('workloads.multi_node.fsdp_training.load_config')
    @patch('workloads.multi_node.fsdp_training.get_logger')
    @patch('workloads.multi_node.fsdp_training.GPUMonitor')
    @patch('workloads.multi_node.fsdp_training.PerformanceBenchmark')
    def test_fsdp_dataloader_creation(self, mock_bench, mock_monitor, mock_logger,
                                       mock_config, mock_setup):
        """Test FSDPTrainer.create_dataloader method."""
        mock_setup.return_value = (0, 2, 0)

        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        from workloads.multi_node.fsdp_training import FSDPTrainer

        trainer = FSDPTrainer()
        trainer.batch_size = 4

        dataloader, sampler = trainer.create_dataloader()

        assert dataloader is not None
        assert sampler is not None


@pytest.mark.unit
class TestDistributedMainFunctions:
    """Test main entry point functions."""

    @patch('workloads.multi_node.ddp_training.DDPTrainer')
    @patch('argparse.ArgumentParser.parse_args')
    def test_ddp_main(self, mock_args, mock_trainer):
        """Test DDP main function."""
        mock_args.return_value = MagicMock(config='test.yaml')
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        from workloads.multi_node.ddp_training import main

        main()

        mock_trainer.assert_called_once_with(config_path='test.yaml')
        mock_trainer_instance.train.assert_called_once()

    @patch('workloads.multi_node.fsdp_training.FSDPTrainer')
    @patch('argparse.ArgumentParser.parse_args')
    def test_fsdp_main(self, mock_args, mock_trainer):
        """Test FSDP main function."""
        mock_args.return_value = MagicMock(config=None)
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        from workloads.multi_node.fsdp_training import main

        main()

        mock_trainer.assert_called_once_with(config_path=None)
        mock_trainer_instance.train.assert_called_once()


@pytest.mark.unit
class TestDDPTrainingLoop:
    """Test DDP training loop components."""

    @patch('workloads.multi_node.ddp_training.setup_distributed')
    @patch('workloads.multi_node.ddp_training.load_config')
    @patch('workloads.multi_node.ddp_training.get_logger')
    @patch('workloads.multi_node.ddp_training.GPUMonitor')
    @patch('workloads.multi_node.ddp_training.PerformanceBenchmark')
    def test_ddp_benchmark_configuration(self, mock_bench, mock_monitor, mock_logger,
                                          mock_config, mock_setup):
        """Test DDPTrainer benchmark is configured correctly."""
        mock_setup.return_value = (0, 2, 0)

        mock_config_instance = MagicMock()
        mock_config_instance.get.side_effect = lambda key, default=None: {
            'training.batch_size': 32,
            'training.epochs': 5,
        }.get(key, default)
        mock_config.return_value = mock_config_instance

        mock_bench_instance = MagicMock()
        mock_bench.return_value = mock_bench_instance

        from workloads.multi_node.ddp_training import DDPTrainer

        trainer = DDPTrainer()

        assert trainer.batch_size == 32
        assert trainer.epochs == 5
        # For rank 0, benchmark should be created
        mock_bench.assert_called_once()


@pytest.mark.unit
class TestFSDPTrainingLoop:
    """Test FSDP training loop components."""

    @patch('workloads.multi_node.fsdp_training.setup_distributed')
    @patch('workloads.multi_node.fsdp_training.load_config')
    @patch('workloads.multi_node.fsdp_training.get_logger')
    @patch('workloads.multi_node.fsdp_training.GPUMonitor')
    @patch('workloads.multi_node.fsdp_training.PerformanceBenchmark')
    def test_fsdp_training_epochs(self, mock_bench, mock_monitor, mock_logger,
                                   mock_config, mock_setup):
        """Test FSDPTrainer training epoch configuration."""
        mock_setup.return_value = (0, 4, 0)

        mock_config_instance = MagicMock()
        mock_config_instance.get.side_effect = lambda key, default=None: {
            'training.epochs': 10,
            'training.batch_size': 16,
            'training.learning_rate': 0.0001,
        }.get(key, default)
        mock_config.return_value = mock_config_instance

        from workloads.multi_node.fsdp_training import FSDPTrainer

        trainer = FSDPTrainer()

        assert trainer.epochs == 10
        assert trainer.batch_size == 16
        assert trainer.lr == 0.0001
