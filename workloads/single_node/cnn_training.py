#!/usr/bin/env python3
"""
CNN Training Workload - Single Node
Trains various CNN architectures on synthetic data for testing and burn-in.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
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


class CNNTrainer:
    """Trainer for CNN models."""

    def __init__(self, config_path: str = None):
        """
        Initialize CNN trainer.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self.logger = get_logger(
            name="cnn_training",
            log_dir=self.config.get('general.output_dir', './results'),
            level=self.config.get('general.log_level', 'INFO')
        )

        # Setup device
        self.device = torch.device(
            self.config.get('gpu.device', 'cuda')
            if torch.cuda.is_available() else 'cpu'
        )

        # Setup GPU monitor
        self.gpu_monitor = GPUMonitor(
            device_ids=self.config.get('gpu.device_ids', [0])
        )

        # Setup benchmark
        self.benchmark = PerformanceBenchmark(
            name="cnn_training",
            output_dir=self.config.get('general.output_dir', './results')
        )

        # Model configuration
        self.model_name = self.config.get('workloads.cnn.model', 'resnet50')
        self.num_classes = self.config.get('workloads.cnn.num_classes', 1000)
        self.image_size = self.config.get('workloads.cnn.image_size', 224)

        # Training configuration
        self.batch_size = self.config.get('training.batch_size', 64)
        self.epochs = self.config.get('training.epochs', 10)
        self.lr = self.config.get('training.learning_rate', 0.001)

        # Mixed precision
        self.use_amp = self.config.get('gpu.mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        self.logger.log_header(f"CNN Training - {self.model_name}")
        self._log_configuration()

    def _log_configuration(self):
        """Log configuration details."""
        config_dict = {
            'Model': self.model_name,
            'Device': str(self.device),
            'Batch Size': self.batch_size,
            'Epochs': self.epochs,
            'Learning Rate': self.lr,
            'Image Size': self.image_size,
            'Num Classes': self.num_classes,
            'Mixed Precision': self.use_amp,
        }
        self.logger.log_config(config_dict)

        # Log GPU info
        if torch.cuda.is_available():
            gpu_info = {}
            for device_id in self.config.get('gpu.device_ids', [0]):
                gpu_info[device_id] = self.gpu_monitor.get_gpu_info(device_id)
            self.logger.log_gpu_info(gpu_info)

    def create_model(self) -> nn.Module:
        """
        Create CNN model.

        Returns:
            PyTorch model
        """
        self.logger.info(f"Creating model: {self.model_name}")

        # Select model architecture
        if self.model_name == 'resnet18':
            model = models.resnet18(num_classes=self.num_classes)
        elif self.model_name == 'resnet50':
            model = models.resnet50(num_classes=self.num_classes)
        elif self.model_name == 'resnet101':
            model = models.resnet101(num_classes=self.num_classes)
        elif self.model_name == 'vgg16':
            model = models.vgg16(num_classes=self.num_classes)
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        return model

    def create_dataloader(self):
        """Create data loader with synthetic data."""
        self.logger.info("Creating synthetic dataloader...")

        dataloader = create_synthetic_dataloader(
            dataset_type="image",
            batch_size=self.batch_size,
            num_samples=self.config.get('workloads.cnn.dataset_size', 50000),
            num_workers=self.config.get('training.num_workers', 4),
            pin_memory=self.config.get('training.pin_memory', True),
            image_size=self.image_size,
            num_channels=3,
            num_classes=self.num_classes,
            seed=self.config.get('general.seed', 42)
        )

        self.logger.info(f"DataLoader created with {len(dataloader)} batches")
        return dataloader

    def train_epoch(self, model, dataloader, optimizer, epoch):
        """
        Train for one epoch.

        Args:
            model: Model to train
            dataloader: Training dataloader
            optimizer: Optimizer
            epoch: Current epoch number
        """
        model.train()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = self.logger.create_progress_bar(
            total=len(dataloader),
            desc=f"Epoch {epoch}/{self.epochs}"
        )

        self.benchmark.start_epoch()

        for batch_idx, (images, labels) in enumerate(dataloader):
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

            # Update metrics
            total_loss += loss.item()
            iter_time = self.benchmark.end_iteration()

            self.benchmark.record_loss(loss.item())
            self.benchmark.record_accuracy(100.0 * correct / total)

            # Record GPU metrics
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                gpu_metrics = self.gpu_monitor.get_metrics(0)
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
                throughput = self.benchmark.get_throughput(self.batch_size)

                metrics = {
                    'loss': avg_loss,
                    'acc': accuracy,
                    'samples/s': throughput,
                }

                # Add GPU metrics
                if torch.cuda.is_available():
                    gpu_summary = self.gpu_monitor.log_metrics_summary()
                    pbar.set_postfix_str(f"{metrics} | {gpu_summary}")
                else:
                    pbar.set_postfix(metrics)

            pbar.update(1)

        pbar.close()

        epoch_time = self.benchmark.end_epoch()
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        self.logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s | "
            f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%"
        )

        return avg_loss, accuracy

    def train(self):
        """Run training."""
        self.logger.log_header("Starting Training")

        # Configure benchmark
        self.benchmark.configure(
            batch_size=self.batch_size,
            sequence_length=0,
            model_flops=0
        )

        # Create model
        model = self.create_model()

        # Create dataloader
        dataloader = self.create_dataloader()

        # Create optimizer
        optimizer_type = self.config.get('optimizer.type', 'adamw')
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=self.config.get('optimizer.momentum', 0.9),
                weight_decay=self.config.get('training.weight_decay', 0.0001)
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.config.get('training.weight_decay', 0.0001)
            )
        else:  # adamw
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.config.get('training.weight_decay', 0.0001)
            )

        # Training loop
        for epoch in range(1, self.epochs + 1):
            loss, accuracy = self.train_epoch(model, dataloader, optimizer, epoch)

        # Print summary
        self.logger.log_header("Training Complete")
        self.benchmark.print_summary()
        self.benchmark.save_results()

        # Final GPU metrics
        if torch.cuda.is_available():
            self.gpu_monitor.print_all_metrics()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CNN Training Workload')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Run training
    trainer = CNNTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
