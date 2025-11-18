#!/usr/bin/env python3
"""
Mixed Precision Training Workload - Single Node
Demonstrates and benchmarks automatic mixed precision (AMP) training.
"""

import os
import sys
import argparse
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


class MixedPrecisionTrainer:
    """Trainer demonstrating mixed precision training."""

    def __init__(self, config_path: str = None):
        """
        Initialize mixed precision trainer.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self.logger = get_logger(
            name="mixed_precision",
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

        # Setup benchmarks (one for FP32, one for AMP)
        self.benchmark_fp32 = PerformanceBenchmark(
            name="mixed_precision_fp32",
            output_dir=self.config.get('general.output_dir', './results')
        )

        self.benchmark_amp = PerformanceBenchmark(
            name="mixed_precision_amp",
            output_dir=self.config.get('general.output_dir', './results')
        )

        # Model configuration
        self.model_name = self.config.get('workloads.cnn.model', 'resnet50')
        self.num_classes = self.config.get('workloads.cnn.num_classes', 1000)
        self.image_size = self.config.get('workloads.cnn.image_size', 224)

        # Training configuration
        self.batch_size = self.config.get('training.batch_size', 64)
        self.num_iterations = 100  # Fixed number of iterations for fair comparison
        self.lr = self.config.get('training.learning_rate', 0.001)

        self.logger.log_header("Mixed Precision Training Benchmark")
        self._log_configuration()

    def _log_configuration(self):
        """Log configuration details."""
        config_dict = {
            'Model': self.model_name,
            'Device': str(self.device),
            'Batch Size': self.batch_size,
            'Iterations': self.num_iterations,
            'Learning Rate': self.lr,
            'Image Size': self.image_size,
            'Num Classes': self.num_classes,
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
        Create model.

        Returns:
            PyTorch model
        """
        if self.model_name == 'resnet50':
            model = models.resnet50(num_classes=self.num_classes)
        elif self.model_name == 'resnet101':
            model = models.resnet101(num_classes=self.num_classes)
        else:
            model = models.resnet50(num_classes=self.num_classes)

        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model parameters: {total_params:,}")

        return model

    def create_dataloader(self):
        """Create data loader with synthetic data."""
        dataloader = create_synthetic_dataloader(
            dataset_type="image",
            batch_size=self.batch_size,
            num_samples=self.num_iterations * self.batch_size,
            num_workers=self.config.get('training.num_workers', 4),
            pin_memory=self.config.get('training.pin_memory', True),
            image_size=self.image_size,
            num_channels=3,
            num_classes=self.num_classes,
            seed=self.config.get('general.seed', 42)
        )

        return dataloader

    def train_fp32(self, model, dataloader, optimizer):
        """
        Train with FP32 (full precision).

        Args:
            model: Model to train
            dataloader: Training dataloader
            optimizer: Optimizer
        """
        self.logger.log_header("Training with FP32")
        model.train()
        criterion = nn.CrossEntropyLoss()

        pbar = self.logger.create_progress_bar(
            total=self.num_iterations,
            desc="FP32 Training"
        )

        self.benchmark_fp32.configure(batch_size=self.batch_size)

        for i, (images, labels) in enumerate(dataloader):
            if i >= self.num_iterations:
                break

            self.benchmark_fp32.start_iteration()

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass (FP32)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            iter_time = self.benchmark_fp32.end_iteration()
            self.benchmark_fp32.record_loss(loss.item())

            # Record GPU metrics
            if i % 10 == 0:
                gpu_metrics = self.gpu_monitor.get_metrics(0)
                if gpu_metrics:
                    self.benchmark_fp32.record_gpu_metrics(
                        utilization=gpu_metrics.utilization,
                        memory_used=gpu_metrics.memory_used,
                        memory_allocated=gpu_metrics.memory_allocated,
                        temperature=gpu_metrics.temperature
                    )

            pbar.update(1)

        pbar.close()
        self.logger.info("FP32 training completed")

    def train_amp(self, model, dataloader, optimizer):
        """
        Train with Automatic Mixed Precision (AMP).

        Args:
            model: Model to train
            dataloader: Training dataloader
            optimizer: Optimizer
        """
        self.logger.log_header("Training with AMP (Automatic Mixed Precision)")
        model.train()
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        pbar = self.logger.create_progress_bar(
            total=self.num_iterations,
            desc="AMP Training"
        )

        self.benchmark_amp.configure(batch_size=self.batch_size)

        for i, (images, labels) in enumerate(dataloader):
            if i >= self.num_iterations:
                break

            self.benchmark_amp.start_iteration()

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with autocast (FP16)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iter_time = self.benchmark_amp.end_iteration()
            self.benchmark_amp.record_loss(loss.item())

            # Record GPU metrics
            if i % 10 == 0:
                gpu_metrics = self.gpu_monitor.get_metrics(0)
                if gpu_metrics:
                    self.benchmark_amp.record_gpu_metrics(
                        utilization=gpu_metrics.utilization,
                        memory_used=gpu_metrics.memory_used,
                        memory_allocated=gpu_metrics.memory_allocated,
                        temperature=gpu_metrics.temperature
                    )

            pbar.update(1)

        pbar.close()
        self.logger.info("AMP training completed")

    def compare_results(self):
        """Compare FP32 vs AMP results."""
        self.logger.log_header("Performance Comparison: FP32 vs AMP")

        summary_fp32 = self.benchmark_fp32.get_summary()
        summary_amp = self.benchmark_amp.get_summary()

        # Timing comparison
        avg_time_fp32 = summary_fp32.get('avg_iteration_time', 0)
        avg_time_amp = summary_amp.get('avg_iteration_time', 0)
        speedup = avg_time_fp32 / avg_time_amp if avg_time_amp > 0 else 0

        self.logger.info("\n[Timing Comparison]")
        self.logger.info(f"  FP32 avg iteration time: {avg_time_fp32*1000:.2f}ms")
        self.logger.info(f"  AMP avg iteration time:  {avg_time_amp*1000:.2f}ms")
        self.logger.info(f"  Speedup: {speedup:.2f}x")

        # Throughput comparison
        throughput_fp32 = summary_fp32.get('avg_samples_per_sec', 0)
        throughput_amp = summary_amp.get('avg_samples_per_sec', 0)

        self.logger.info("\n[Throughput Comparison]")
        self.logger.info(f"  FP32 throughput: {throughput_fp32:.2f} samples/sec")
        self.logger.info(f"  AMP throughput:  {throughput_amp:.2f} samples/sec")

        # Memory comparison
        memory_fp32 = summary_fp32.get('peak_gpu_memory_gb', 0)
        memory_amp = summary_amp.get('peak_gpu_memory_gb', 0)
        memory_savings = memory_fp32 - memory_amp

        self.logger.info("\n[Memory Comparison]")
        self.logger.info(f"  FP32 peak memory: {memory_fp32:.2f} GB")
        self.logger.info(f"  AMP peak memory:  {memory_amp:.2f} GB")
        self.logger.info(f"  Memory savings:   {memory_savings:.2f} GB")

        # GPU utilization
        util_fp32 = summary_fp32.get('avg_gpu_utilization', 0)
        util_amp = summary_amp.get('avg_gpu_utilization', 0)

        self.logger.info("\n[GPU Utilization]")
        self.logger.info(f"  FP32 avg utilization: {util_fp32:.1f}%")
        self.logger.info(f"  AMP avg utilization:  {util_amp:.1f}%")

        self.logger.info("")

    def run(self):
        """Run mixed precision benchmark."""
        if not torch.cuda.is_available():
            self.logger.error("CUDA not available. Mixed precision training requires CUDA.")
            return

        # Create dataloader (shared for both runs)
        dataloader = self.create_dataloader()

        # Benchmark FP32
        self.logger.info("\n" + "="*80)
        self.logger.info("Running FP32 benchmark...")
        self.logger.info("="*80 + "\n")

        model_fp32 = self.create_model()
        optimizer_fp32 = optim.AdamW(model_fp32.parameters(), lr=self.lr)
        self.gpu_monitor.reset_peak_memory_stats()
        self.train_fp32(model_fp32, dataloader, optimizer_fp32)

        # Clear GPU memory
        del model_fp32, optimizer_fp32
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Benchmark AMP
        self.logger.info("\n" + "="*80)
        self.logger.info("Running AMP benchmark...")
        self.logger.info("="*80 + "\n")

        # Recreate dataloader for fair comparison
        dataloader = self.create_dataloader()

        model_amp = self.create_model()
        optimizer_amp = optim.AdamW(model_amp.parameters(), lr=self.lr)
        self.gpu_monitor.reset_peak_memory_stats()
        self.train_amp(model_amp, dataloader, optimizer_amp)

        # Compare results
        self.compare_results()

        # Save results
        self.benchmark_fp32.save_results()
        self.benchmark_amp.save_results()

        self.logger.log_header("Benchmark Complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Mixed Precision Training Benchmark')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Run benchmark
    trainer = MixedPrecisionTrainer(config_path=args.config)
    trainer.run()


if __name__ == "__main__":
    main()
