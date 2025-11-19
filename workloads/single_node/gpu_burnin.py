#!/usr/bin/env python3
"""
GPU Burn-in Workload - Single Node
Maximizes GPU utilization for stress testing and validation.
Supports multi-GPU stress testing with configurable GPU selection.
"""

import os
import sys
import argparse
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

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


def gpu_worker(gpu_id: int, config: dict, duration_minutes: int, operations: list,
               results_queue: mp.Queue, stop_event: mp.Event):
    """
    Worker function that runs burn-in on a single GPU.

    Args:
        gpu_id: GPU device ID
        config: Configuration dictionary
        duration_minutes: Duration in minutes
        operations: List of operations to run
        results_queue: Queue to report results
        stop_event: Event to signal stop
    """
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    # Get configuration values
    batch_size = config.get('training.batch_size', 128)
    matrix_size = config.get('workloads.gpu_burnin.matrix_size', 8192)
    image_size = 224

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    iteration = 0
    total_iterations = 0

    for operation in operations:
        if stop_event.is_set():
            break

        if operation == 'conv2d':
            # CNN burn-in
            model = BurnInModel(channels=128, num_blocks=6).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            generator = SyntheticBurnInGenerator(
                batch_size=batch_size,
                input_shape=(3, image_size, image_size),
                num_classes=1000,
                device=device
            )

            while time.time() < end_time and not stop_event.is_set():
                images, labels = generator.generate()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                iteration += 1
                total_iterations += 1

                # Report metrics periodically
                if iteration % 50 == 0:
                    gpu_util = 0
                    gpu_mem = 0
                    gpu_temp = 0
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu_util = util.gpu
                        gpu_mem = mem.used / (1024**3)
                        gpu_temp = temp
                    except:
                        gpu_mem = torch.cuda.memory_allocated(device) / (1024**3)

                    results_queue.put({
                        'gpu_id': gpu_id,
                        'operation': operation,
                        'iteration': iteration,
                        'utilization': gpu_util,
                        'memory_gb': gpu_mem,
                        'temperature': gpu_temp,
                        'loss': loss.item() if 'loss' in dir() else 0
                    })

            del model, optimizer, generator
            torch.cuda.empty_cache()

        elif operation == 'matmul':
            # Matrix multiplication burn-in
            A = torch.randn(matrix_size, matrix_size, device=device)
            B = torch.randn(matrix_size, matrix_size, device=device)

            while time.time() < end_time and not stop_event.is_set():
                for _ in range(5):
                    C = torch.matmul(A, B)
                    A = C
                iteration += 1
                total_iterations += 1

                if iteration % 20 == 0:
                    gpu_mem = torch.cuda.memory_allocated(device) / (1024**3)
                    results_queue.put({
                        'gpu_id': gpu_id,
                        'operation': operation,
                        'iteration': iteration,
                        'utilization': 0,
                        'memory_gb': gpu_mem,
                        'temperature': 0,
                        'loss': 0
                    })

            del A, B
            torch.cuda.empty_cache()

        elif operation == 'attention':
            # Attention burn-in
            attn_batch = 32
            seq_len = 512
            dim = 768

            while time.time() < end_time and not stop_event.is_set():
                Q = torch.randn(attn_batch, seq_len, dim, device=device)
                K = torch.randn(attn_batch, seq_len, dim, device=device)
                V = torch.randn(attn_batch, seq_len, dim, device=device)

                scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)
                attn = F.softmax(scores, dim=-1)
                output = torch.matmul(attn, V)

                iteration += 1
                total_iterations += 1

                if iteration % 30 == 0:
                    gpu_mem = torch.cuda.memory_allocated(device) / (1024**3)
                    results_queue.put({
                        'gpu_id': gpu_id,
                        'operation': operation,
                        'iteration': iteration,
                        'utilization': 0,
                        'memory_gb': gpu_mem,
                        'temperature': 0,
                        'loss': 0
                    })

                del Q, K, V, scores, attn, output

            torch.cuda.empty_cache()

    # Final report
    results_queue.put({
        'gpu_id': gpu_id,
        'operation': 'DONE',
        'iteration': total_iterations,
        'utilization': 0,
        'memory_gb': 0,
        'temperature': 0,
        'loss': 0
    })


class GPUBurnIn:
    """GPU burn-in stress test with multi-GPU support."""

    def __init__(self, config_path: str = None, num_gpus: int = None):
        """
        Initialize GPU burn-in.

        Args:
            config_path: Path to configuration file
            num_gpus: Number of GPUs to use (None = use config or 1)
        """
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self.logger = get_logger(
            name="gpu_burnin",
            log_dir=self.config.get('general.output_dir', './results'),
            level=self.config.get('general.log_level', 'INFO')
        )

        # Determine number of GPUs
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus is not None:
            self.num_gpus = min(num_gpus, available_gpus)
        else:
            config_gpus = len(self.config.get('gpu.device_ids', [0]))
            self.num_gpus = min(config_gpus, available_gpus) if available_gpus > 0 else 0

        self.gpu_ids = list(range(self.num_gpus))

        # Setup device (primary)
        self.device = torch.device(
            f'cuda:0' if torch.cuda.is_available() else 'cpu'
        )

        # Setup GPU monitor for all GPUs
        self.gpu_monitor = GPUMonitor(
            device_ids=self.gpu_ids if self.gpu_ids else [0]
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

        # Multi-GPU state
        self.gpu_metrics = {gpu_id: {'iteration': 0, 'utilization': 0, 'memory_gb': 0, 'temperature': 0, 'operation': 'idle'}
                           for gpu_id in self.gpu_ids}

        self.logger.log_header("GPU Burn-in Test")
        self._log_configuration()

    def _log_configuration(self):
        """Log configuration details."""
        config_dict = {
            'Mode': f"Multi-GPU ({self.num_gpus} GPUs)" if self.num_gpus > 1 else "Single GPU",
            'GPU IDs': ', '.join(map(str, self.gpu_ids)) if self.gpu_ids else 'None',
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
            for device_id in self.gpu_ids:
                gpu_info[device_id] = self.gpu_monitor.get_gpu_info(device_id)
            self.logger.log_gpu_info(gpu_info)

    def _format_gpu_status(self):
        """Format current GPU status for display."""
        lines = []
        for gpu_id in self.gpu_ids:
            metrics = self.gpu_metrics.get(gpu_id, {})
            op = metrics.get('operation', 'idle')[:8]
            iters = metrics.get('iteration', 0)
            util = metrics.get('utilization', 0)
            mem = metrics.get('memory_gb', 0)
            temp = metrics.get('temperature', 0)

            if temp > 0:
                lines.append(f"GPU{gpu_id}: {op:8s} | iter:{iters:5d} | {util:3.0f}% | {mem:.1f}GB | {temp}°C")
            else:
                lines.append(f"GPU{gpu_id}: {op:8s} | iter:{iters:5d} | {mem:.1f}GB")

        return '\n'.join(lines)

    def run_multi_gpu(self):
        """Run burn-in on multiple GPUs simultaneously."""
        self.logger.info(f"Starting multi-GPU burn-in on {self.num_gpus} GPUs...")
        self.logger.info(f"GPU IDs: {self.gpu_ids}")
        self.logger.info(f"Operations: {', '.join(self.operations)}")
        self.logger.info(f"Duration: {self.duration_minutes} minutes")
        self.logger.info("")

        # Create multiprocessing context
        mp.set_start_method('spawn', force=True)
        results_queue = mp.Queue()
        stop_event = mp.Event()

        # Convert config to dict for passing to workers
        config_dict = {
            'training.batch_size': self.batch_size,
            'workloads.gpu_burnin.matrix_size': self.matrix_size,
        }

        # Start worker processes
        processes = []
        for gpu_id in self.gpu_ids:
            p = mp.Process(
                target=gpu_worker,
                args=(gpu_id, config_dict, self.duration_minutes, self.operations,
                      results_queue, stop_event)
            )
            p.start()
            processes.append(p)
            self.logger.info(f"Started worker process for GPU {gpu_id}")

        # Monitor progress
        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)
        completed_gpus = set()
        last_log_time = 0

        self.logger.info("")
        self.logger.log_header("GPU Status Monitor")

        try:
            while len(completed_gpus) < self.num_gpus:
                # Process results from queue
                while not results_queue.empty():
                    try:
                        result = results_queue.get_nowait()
                        gpu_id = result['gpu_id']

                        if result['operation'] == 'DONE':
                            completed_gpus.add(gpu_id)
                            self.gpu_metrics[gpu_id]['operation'] = 'done'
                            self.gpu_metrics[gpu_id]['iteration'] = result['iteration']
                            self.logger.info(f"GPU {gpu_id} completed: {result['iteration']} total iterations")
                        else:
                            self.gpu_metrics[gpu_id] = {
                                'operation': result['operation'],
                                'iteration': result['iteration'],
                                'utilization': result['utilization'],
                                'memory_gb': result['memory_gb'],
                                'temperature': result['temperature']
                            }
                    except:
                        break

                # Log status periodically
                current_time = time.time()
                if current_time - last_log_time >= 5:  # Log every 5 seconds
                    elapsed = current_time - start_time
                    remaining = max(0, end_time - current_time)

                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"Elapsed: {elapsed/60:.1f}m | Remaining: {remaining/60:.1f}m")
                    self.logger.info(f"{'='*60}")
                    self.logger.info(self._format_gpu_status())

                    last_log_time = current_time

                time.sleep(0.5)

        except KeyboardInterrupt:
            self.logger.warning("Interrupt received, stopping workers...")
            stop_event.set()

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        # Final summary
        self.logger.info("")
        self.logger.log_header("Multi-GPU Burn-in Complete")

        total_iterations = sum(m.get('iteration', 0) for m in self.gpu_metrics.values())
        elapsed_time = time.time() - start_time

        self.logger.info(f"Total GPUs: {self.num_gpus}")
        self.logger.info(f"Total iterations: {total_iterations}")
        self.logger.info(f"Elapsed time: {elapsed_time/60:.1f} minutes")
        self.logger.info(f"Avg iterations/GPU: {total_iterations/self.num_gpus:.0f}")

        self.logger.info("\nPer-GPU Summary:")
        for gpu_id in self.gpu_ids:
            iters = self.gpu_metrics[gpu_id].get('iteration', 0)
            self.logger.info(f"  GPU {gpu_id}: {iters} iterations")

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

        # Use multi-GPU mode if more than one GPU
        if self.num_gpus > 1:
            self.run_multi_gpu()
            return

        # Single GPU mode
        self.logger.info(f"Running single-GPU burn-in on GPU 0...")

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
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use for stress testing (default: 1, use 0 or "all" for all available)'
    )
    parser.add_argument(
        '--all-gpus',
        action='store_true',
        help='Use all available GPUs'
    )
    args = parser.parse_args()

    # Determine number of GPUs
    num_gpus = args.num_gpus
    if args.all_gpus:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Run burn-in
    burnin = GPUBurnIn(config_path=args.config, num_gpus=num_gpus)

    # Override duration if specified
    if args.duration is not None:
        burnin.duration_minutes = args.duration
        burnin.logger.info(f"Duration overridden to {args.duration} minutes")

    burnin.run()


if __name__ == "__main__":
    main()
