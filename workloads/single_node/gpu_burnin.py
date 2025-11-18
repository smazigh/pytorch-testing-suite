#!/usr/bin/env python3
"""
GPU Burn-in Workload - Single Node
Maximizes GPU utilization for stress testing and validation.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    load_config,
    get_logger,
    PerformanceBenchmark,
    GPUMonitor,
    SyntheticBurnInGenerator
)


class BurnInModel(nn.Module):
    """High-compute model for GPU burn-in."""

    def __init__(self, channels: int = 64, num_blocks: int = 4):
        """
        Initialize burn-in model with heavy computation.

        Args:
            channels: Number of channels
            num_blocks: Number of residual blocks
        """
        super().__init__()

        layers = []
        in_channels = 3

        # Build deep convolutional network
        for i in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            in_channels = channels
            channels *= 2

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, 1000)

    def forward(self, x):
        """Forward pass with intensive computation."""
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def matrix_multiply_stress(size: int, iterations: int, device: torch.device):
    """
    Perform intensive matrix multiplications.

    Args:
        size: Matrix size
        iterations: Number of iterations
        device: Device to run on
    """
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    for _ in range(iterations):
        C = torch.matmul(A, B)
        A = C


def attention_stress(batch_size: int, seq_len: int, dim: int, device: torch.device):
    """
    Perform intensive attention computations.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        dim: Model dimension
        device: Device to run on
    """
    # Generate query, key, value
    Q = torch.randn(batch_size, seq_len, dim, device=device)
    K = torch.randn(batch_size, seq_len, dim, device=device)
    V = torch.randn(batch_size, seq_len, dim, device=device)

    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)

    return output


class GPUBurnIn:
    """GPU burn-in stress test."""

    def __init__(self, config_path: str = None):
        """
        Initialize GPU burn-in.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self.logger = get_logger(
            name="gpu_burnin",
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
            name="gpu_burnin",
            output_dir=self.config.get('general.output_dir', './results')
        )

        # Burn-in configuration
        self.duration_minutes = self.config.get('burnin.duration_minutes', 30)
        self.stress_level = self.config.get('workloads.gpu_burnin.stress_level', 100)
        self.matrix_size = self.config.get('workloads.gpu_burnin.matrix_size', 8192)
        self.operations = self.config.get('workloads.gpu_burnin.operations', ['matmul', 'conv2d', 'attention'])

        # Batch configuration
        self.batch_size = self.config.get('training.batch_size', 128)
        self.image_size = 224

        self.logger.log_header("GPU Burn-in Test")
        self._log_configuration()

    def _log_configuration(self):
        """Log configuration details."""
        config_dict = {
            'Device': str(self.device),
            'Duration': f"{self.duration_minutes} minutes",
            'Stress Level': f"{self.stress_level}%",
            'Matrix Size': self.matrix_size,
            'Operations': ', '.join(self.operations),
            'Batch Size': self.batch_size,
            'Image Size': self.image_size,
        }
        self.logger.log_config(config_dict)

        # Log GPU info
        if torch.cuda.is_available():
            gpu_info = {}
            for device_id in self.config.get('gpu.device_ids', [0]):
                gpu_info[device_id] = self.gpu_monitor.get_gpu_info(device_id)
            self.logger.log_gpu_info(gpu_info)

    def run_cnn_burnin(self):
        """Run CNN-based burn-in."""
        self.logger.info("Running CNN burn-in...")

        # Create model
        model = BurnInModel(channels=128, num_blocks=6).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create data generator
        generator = SyntheticBurnInGenerator(
            batch_size=self.batch_size,
            input_shape=(3, self.image_size, self.image_size),
            num_classes=1000,
            device=self.device
        )

        # Run for specified duration
        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)
        iteration = 0

        self.logger.info(f"Starting burn-in for {self.duration_minutes} minutes...")

        while time.time() < end_time:
            self.benchmark.start_iteration()

            # Generate data
            images, labels = generator.generate()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            iter_time = self.benchmark.end_iteration()
            self.benchmark.record_loss(loss.item())

            # Record GPU metrics
            if iteration % 10 == 0:
                gpu_metrics = self.gpu_monitor.get_metrics(0)
                if gpu_metrics:
                    self.benchmark.record_gpu_metrics(
                        utilization=gpu_metrics.utilization,
                        memory_used=gpu_metrics.memory_used,
                        memory_allocated=gpu_metrics.memory_allocated,
                        temperature=gpu_metrics.temperature
                    )

            # Log progress
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                remaining = end_time - time.time()
                throughput = self.benchmark.get_throughput(self.batch_size)

                self.logger.info(
                    f"Iteration {iteration} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Throughput: {throughput:.1f} samples/s | "
                    f"Elapsed: {elapsed/60:.1f}m | "
                    f"Remaining: {remaining/60:.1f}m | "
                    f"{self.gpu_monitor.log_metrics_summary()}"
                )

            iteration += 1

        self.logger.info(f"CNN burn-in completed: {iteration} iterations")

    def run_matmul_burnin(self):
        """Run matrix multiplication burn-in."""
        self.logger.info("Running matrix multiplication burn-in...")

        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)
        iteration = 0

        while time.time() < end_time:
            self.benchmark.start_iteration()

            # Perform intensive matrix multiplications
            matrix_multiply_stress(
                size=self.matrix_size,
                iterations=5,
                device=self.device
            )

            iter_time = self.benchmark.end_iteration()

            # Record GPU metrics
            if iteration % 5 == 0:
                gpu_metrics = self.gpu_monitor.get_metrics(0)
                if gpu_metrics:
                    self.benchmark.record_gpu_metrics(
                        utilization=gpu_metrics.utilization,
                        memory_used=gpu_metrics.memory_used,
                        memory_allocated=gpu_metrics.memory_allocated,
                        temperature=gpu_metrics.temperature
                    )

            # Log progress
            if iteration % 50 == 0:
                elapsed = time.time() - start_time
                remaining = end_time - time.time()

                self.logger.info(
                    f"Iteration {iteration} | "
                    f"Elapsed: {elapsed/60:.1f}m | "
                    f"Remaining: {remaining/60:.1f}m | "
                    f"{self.gpu_monitor.log_metrics_summary()}"
                )

            iteration += 1

        self.logger.info(f"Matrix multiplication burn-in completed: {iteration} iterations")

    def run_attention_burnin(self):
        """Run attention mechanism burn-in."""
        self.logger.info("Running attention burn-in...")

        batch_size = 32
        seq_len = 512
        dim = 768

        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)
        iteration = 0

        while time.time() < end_time:
            self.benchmark.start_iteration()

            # Perform attention computations
            _ = attention_stress(batch_size, seq_len, dim, self.device)

            iter_time = self.benchmark.end_iteration()

            # Record GPU metrics
            if iteration % 10 == 0:
                gpu_metrics = self.gpu_monitor.get_metrics(0)
                if gpu_metrics:
                    self.benchmark.record_gpu_metrics(
                        utilization=gpu_metrics.utilization,
                        memory_used=gpu_metrics.memory_used,
                        memory_allocated=gpu_metrics.memory_allocated,
                        temperature=gpu_metrics.temperature
                    )

            # Log progress
            if iteration % 50 == 0:
                elapsed = time.time() - start_time
                remaining = end_time - time.time()

                self.logger.info(
                    f"Iteration {iteration} | "
                    f"Elapsed: {elapsed/60:.1f}m | "
                    f"Remaining: {remaining/60:.1f}m | "
                    f"{self.gpu_monitor.log_metrics_summary()}"
                )

            iteration += 1

        self.logger.info(f"Attention burn-in completed: {iteration} iterations")

    def run(self):
        """Run burn-in tests."""
        self.logger.log_header("Starting GPU Burn-in")

        if not torch.cuda.is_available():
            self.logger.error("CUDA not available. Cannot run GPU burn-in.")
            return

        # Configure benchmark
        self.benchmark.configure(batch_size=self.batch_size)

        # Run selected operations
        for operation in self.operations:
            if operation == 'conv2d':
                self.run_cnn_burnin()
            elif operation == 'matmul':
                self.run_matmul_burnin()
            elif operation == 'attention':
                self.run_attention_burnin()
            else:
                self.logger.warning(f"Unknown operation: {operation}")

        # Print summary
        self.logger.log_header("Burn-in Complete")
        self.benchmark.print_summary()
        self.benchmark.save_results()

        # Final GPU metrics
        self.gpu_monitor.print_all_metrics()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GPU Burn-in Workload')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration in minutes (overrides config)'
    )
    args = parser.parse_args()

    # Run burn-in
    burnin = GPUBurnIn(config_path=args.config)

    # Override duration if specified
    if args.duration is not None:
        burnin.duration_minutes = args.duration
        burnin.logger.info(f"Duration overridden to {args.duration} minutes")

    burnin.run()


if __name__ == "__main__":
    main()
