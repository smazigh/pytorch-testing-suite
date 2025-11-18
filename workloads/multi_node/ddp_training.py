#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) Training Workload - Multi Node
Trains models using PyTorch Distributed Data Parallel across multiple GPUs/nodes.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    load_config,
    get_logger,
    PerformanceBenchmark,
    GPUMonitor,
    create_synthetic_dataloader
)


def setup_distributed():
    """Initialize distributed training environment."""
    # Initialize process group
    dist.init_process_group(backend='nccl')

    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set device
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


class DDPTrainer:
    """Trainer for DDP training."""

    def __init__(self, config_path: str = None):
        """
        Initialize DDP trainer.

        Args:
            config_path: Path to configuration file
        """
        # Setup distributed
        self.rank, self.world_size, self.local_rank = setup_distributed()

        # Load configuration
        self.config = load_config(config_path)

        # Setup logging (only rank 0 logs to file)
        self.logger = get_logger(
            name="ddp_training",
            log_dir=self.config.get('general.output_dir', './results'),
            level=self.config.get('general.log_level', 'INFO'),
            rank=self.rank
        )

        # Setup device
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Setup GPU monitor
        self.gpu_monitor = GPUMonitor(device_ids=[self.local_rank])

        # Setup benchmark (only rank 0)
        if self.rank == 0:
            self.benchmark = PerformanceBenchmark(
                name="ddp_training",
                output_dir=self.config.get('general.output_dir', './results')
            )
        else:
            self.benchmark = None

        # Model configuration
        self.model_name = self.config.get('workloads.ddp.model', 'resnet50')
        self.num_classes = self.config.get('workloads.cnn.num_classes', 1000)
        self.image_size = self.config.get('workloads.cnn.image_size', 224)

        # Training configuration
        self.batch_size = self.config.get('training.batch_size', 64)  # Per-GPU batch size
        self.epochs = self.config.get('training.epochs', 10)
        self.lr = self.config.get('training.learning_rate', 0.001)

        # Mixed precision
        self.use_amp = self.config.get('gpu.mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        if self.rank == 0:
            self.logger.log_header(f"DDP Training - {self.model_name}")
            self._log_configuration()

    def _log_configuration(self):
        """Log configuration details."""
        config_dict = {
            'Model': self.model_name,
            'World Size': self.world_size,
            'Rank': self.rank,
            'Local Rank': self.local_rank,
            'Per-GPU Batch Size': self.batch_size,
            'Global Batch Size': self.batch_size * self.world_size,
            'Epochs': self.epochs,
            'Learning Rate': self.lr,
            'Image Size': self.image_size,
            'Num Classes': self.num_classes,
            'Mixed Precision': self.use_amp,
        }
        self.logger.log_config(config_dict)

        # Log GPU info
        gpu_info = {self.local_rank: self.gpu_monitor.get_gpu_info(self.local_rank)}
        self.logger.log_gpu_info(gpu_info)

    def create_model(self) -> nn.Module:
        """
        Create and wrap model with DDP.

        Returns:
            DDP-wrapped PyTorch model
        """
        if self.rank == 0:
            self.logger.info(f"Creating model: {self.model_name}")

        # Select model architecture
        if self.model_name == 'resnet18':
            model = models.resnet18(num_classes=self.num_classes)
        elif self.model_name == 'resnet50':
            model = models.resnet50(num_classes=self.num_classes)
        elif self.model_name == 'resnet101':
            model = models.resnet101(num_classes=self.num_classes)
        else:
            model = models.resnet50(num_classes=self.num_classes)

        model = model.to(self.device)

        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.config.get('distributed.find_unused_parameters', False),
            gradient_as_bucket_view=self.config.get('distributed.gradient_as_bucket_view', True)
        )

        if self.rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        return model

    def create_dataloader(self):
        """Create distributed data loader with synthetic data."""
        if self.rank == 0:
            self.logger.info("Creating distributed synthetic dataloader...")

        # Create base dataset
        from utils.data_generators import SyntheticImageDataset

        dataset = SyntheticImageDataset(
            num_samples=self.config.get('workloads.cnn.dataset_size', 50000),
            image_size=self.image_size,
            num_channels=3,
            num_classes=self.num_classes,
            seed=self.config.get('general.seed', 42)
        )

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.config.get('training.num_workers', 4),
            pin_memory=self.config.get('training.pin_memory', True),
            drop_last=True
        )

        if self.rank == 0:
            self.logger.info(f"DataLoader created with {len(dataloader)} batches per GPU")
            self.logger.info(f"Total global batches: {len(dataloader)} batches")

        return dataloader, sampler

    def train_epoch(self, model, dataloader, sampler, optimizer, epoch):
        """
        Train for one epoch.

        Args:
            model: DDP-wrapped model
            dataloader: Distributed dataloader
            sampler: Distributed sampler
            optimizer: Optimizer
            epoch: Current epoch number
        """
        model.train()
        sampler.set_epoch(epoch)  # Ensure different shuffling each epoch
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar (only rank 0)
        if self.rank == 0:
            pbar = self.logger.create_progress_bar(
                total=len(dataloader),
                desc=f"Epoch {epoch}/{self.epochs}"
            )
            self.benchmark.start_epoch()

        for batch_idx, (images, labels) in enumerate(dataloader):
            if self.rank == 0:
                self.benchmark.start_iteration()

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update metrics (only rank 0)
            if self.rank == 0:
                total_loss += loss.item()
                iter_time = self.benchmark.end_iteration()

                self.benchmark.record_loss(loss.item())
                self.benchmark.record_accuracy(100.0 * correct / total)

                # Record GPU metrics
                if batch_idx % 10 == 0:
                    gpu_metrics = self.gpu_monitor.get_metrics(self.local_rank)
                    if gpu_metrics:
                        self.benchmark.record_gpu_metrics(
                            utilization=gpu_metrics.utilization,
                            memory_used=gpu_metrics.memory_used,
                            memory_allocated=gpu_metrics.memory_allocated,
                            temperature=gpu_metrics.temperature
                        )

                # Log progress
                if batch_idx % self.config.get('logging.console_log_interval', 10) == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100.0 * correct / total
                    # Global throughput = per-GPU throughput * world_size
                    throughput = self.benchmark.get_throughput(self.batch_size) * self.world_size

                    metrics = {
                        'loss': avg_loss,
                        'acc': accuracy,
                        'global_samples/s': throughput,
                    }

                    gpu_summary = self.gpu_monitor.log_metrics_summary()
                    pbar.set_postfix_str(f"{metrics} | {gpu_summary}")

                pbar.update(1)

        if self.rank == 0:
            pbar.close()
            epoch_time = self.benchmark.end_epoch()
            avg_loss = total_loss / len(dataloader)
            accuracy = 100.0 * correct / total

            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%"
            )

    def train(self):
        """Run distributed training."""
        if self.rank == 0:
            self.logger.log_header("Starting Distributed Training")

            # Configure benchmark
            self.benchmark.configure(
                batch_size=self.batch_size * self.world_size,  # Global batch size
                sequence_length=0,
                model_flops=0
            )

        # Create model
        model = self.create_model()

        # Create dataloader
        dataloader, sampler = self.create_dataloader()

        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.config.get('training.weight_decay', 0.0001)
        )

        # Training loop
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(model, dataloader, sampler, optimizer, epoch)

        # Print summary (only rank 0)
        if self.rank == 0:
            self.logger.log_header("Training Complete")
            self.benchmark.print_summary()
            self.benchmark.save_results()
            self.gpu_monitor.print_all_metrics()

        # Cleanup
        cleanup_distributed()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='DDP Training Workload')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Run training
    trainer = DDPTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
