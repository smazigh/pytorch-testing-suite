"""
Benchmarking utilities for PyTorch Testing Framework.
Tracks performance metrics, throughput, and generates reports.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    # Timing metrics
    iteration_times: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

    # Throughput metrics
    samples_per_sec: List[float] = field(default_factory=list)
    tokens_per_sec: List[float] = field(default_factory=list)

    # Loss and accuracy metrics
    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)

    # GPU metrics
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_allocated: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)

    # Compute metrics
    tflops: List[float] = field(default_factory=list)

    # Metadata
    timestamps: List[float] = field(default_factory=list)


class PerformanceBenchmark:
    """Track and analyze performance metrics."""

    def __init__(self, name: str = "benchmark", output_dir: str = "./results"):
        """
        Initialize benchmark tracker.

        Args:
            name: Benchmark name
            output_dir: Directory to save benchmark results
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = BenchmarkMetrics()
        self.start_time = time.time()
        self.iteration_start = None
        self.epoch_start = None

        # Configuration
        self.batch_size = 0
        self.sequence_length = 0
        self.model_flops = 0

    def configure(
        self,
        batch_size: int,
        sequence_length: int = 0,
        model_flops: float = 0
    ) -> None:
        """
        Configure benchmark parameters.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length (for transformers)
            model_flops: Model FLOPs per forward pass
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.model_flops = model_flops

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self.epoch_start = time.time()

    def end_epoch(self) -> float:
        """
        Mark the end of an epoch.

        Returns:
            Epoch time in seconds
        """
        if self.epoch_start is None:
            return 0.0

        epoch_time = time.time() - self.epoch_start
        self.metrics.epoch_times.append(epoch_time)
        return epoch_time

    def start_iteration(self) -> None:
        """Mark the start of an iteration."""
        self.iteration_start = time.time()

    def end_iteration(self) -> float:
        """
        Mark the end of an iteration.

        Returns:
            Iteration time in seconds
        """
        if self.iteration_start is None:
            return 0.0

        iteration_time = time.time() - self.iteration_start
        self.metrics.iteration_times.append(iteration_time)
        self.metrics.timestamps.append(time.time())

        # Calculate throughput
        if self.batch_size > 0 and iteration_time > 0:
            samples_per_sec = self.batch_size / iteration_time
            self.metrics.samples_per_sec.append(samples_per_sec)

            # Tokens per second for sequence models
            if self.sequence_length > 0:
                tokens_per_sec = (self.batch_size * self.sequence_length) / iteration_time
                self.metrics.tokens_per_sec.append(tokens_per_sec)

        # Calculate TFLOPS if model_flops is provided
        if self.model_flops > 0 and iteration_time > 0:
            tflops = (self.model_flops * self.batch_size * 2) / (iteration_time * 1e12)
            self.metrics.tflops.append(tflops)

        return iteration_time

    def record_loss(self, loss: float) -> None:
        """Record loss value."""
        self.metrics.losses.append(float(loss))

    def record_accuracy(self, accuracy: float) -> None:
        """Record accuracy value."""
        self.metrics.accuracies.append(float(accuracy))

    def record_gpu_metrics(
        self,
        utilization: Optional[float] = None,
        memory_used: Optional[float] = None,
        memory_allocated: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> None:
        """
        Record GPU metrics.

        Args:
            utilization: GPU utilization percentage (0-100)
            memory_used: GPU memory used in GB
            memory_allocated: GPU memory allocated in GB
            temperature: GPU temperature in Celsius
        """
        if utilization is not None:
            self.metrics.gpu_utilization.append(utilization)
        if memory_used is not None:
            self.metrics.gpu_memory_used.append(memory_used)
        if memory_allocated is not None:
            self.metrics.gpu_memory_allocated.append(memory_allocated)
        if temperature is not None:
            self.metrics.gpu_temperature.append(temperature)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get benchmark summary statistics.

        Returns:
            Dictionary of summary statistics
        """
        summary = {
            'total_time': time.time() - self.start_time,
            'num_iterations': len(self.metrics.iteration_times),
            'num_epochs': len(self.metrics.epoch_times),
        }

        # Timing statistics
        if self.metrics.iteration_times:
            summary['avg_iteration_time'] = np.mean(self.metrics.iteration_times)
            summary['std_iteration_time'] = np.std(self.metrics.iteration_times)
            summary['min_iteration_time'] = np.min(self.metrics.iteration_times)
            summary['max_iteration_time'] = np.max(self.metrics.iteration_times)

        if self.metrics.epoch_times:
            summary['avg_epoch_time'] = np.mean(self.metrics.epoch_times)
            summary['total_epoch_time'] = np.sum(self.metrics.epoch_times)

        # Throughput statistics
        if self.metrics.samples_per_sec:
            summary['avg_samples_per_sec'] = np.mean(self.metrics.samples_per_sec)
            summary['peak_samples_per_sec'] = np.max(self.metrics.samples_per_sec)

        if self.metrics.tokens_per_sec:
            summary['avg_tokens_per_sec'] = np.mean(self.metrics.tokens_per_sec)
            summary['peak_tokens_per_sec'] = np.max(self.metrics.tokens_per_sec)

        # TFLOPS statistics
        if self.metrics.tflops:
            summary['avg_tflops'] = np.mean(self.metrics.tflops)
            summary['peak_tflops'] = np.max(self.metrics.tflops)

        # Loss and accuracy statistics
        if self.metrics.losses:
            summary['final_loss'] = self.metrics.losses[-1]
            summary['min_loss'] = np.min(self.metrics.losses)
            summary['avg_loss'] = np.mean(self.metrics.losses)

        if self.metrics.accuracies:
            summary['final_accuracy'] = self.metrics.accuracies[-1]
            summary['max_accuracy'] = np.max(self.metrics.accuracies)
            summary['avg_accuracy'] = np.mean(self.metrics.accuracies)

        # GPU statistics
        if self.metrics.gpu_utilization:
            summary['avg_gpu_utilization'] = np.mean(self.metrics.gpu_utilization)
            summary['max_gpu_utilization'] = np.max(self.metrics.gpu_utilization)

        if self.metrics.gpu_memory_used:
            summary['avg_gpu_memory_gb'] = np.mean(self.metrics.gpu_memory_used)
            summary['peak_gpu_memory_gb'] = np.max(self.metrics.gpu_memory_used)

        if self.metrics.gpu_memory_allocated:
            summary['avg_gpu_allocated_gb'] = np.mean(self.metrics.gpu_memory_allocated)
            summary['peak_gpu_allocated_gb'] = np.max(self.metrics.gpu_memory_allocated)

        if self.metrics.gpu_temperature:
            summary['avg_gpu_temp'] = np.mean(self.metrics.gpu_temperature)
            summary['max_gpu_temp'] = np.max(self.metrics.gpu_temperature)

        return summary

    def print_summary(self) -> None:
        """Print formatted benchmark summary."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print(f"  Benchmark Summary: {self.name}")
        print("=" * 80)

        print("\n[Timing]")
        print(f"  Total time: {summary.get('total_time', 0):.2f}s")
        if 'avg_iteration_time' in summary:
            print(f"  Avg iteration time: {summary['avg_iteration_time']*1000:.2f}ms")
            print(f"  Std iteration time: {summary['std_iteration_time']*1000:.2f}ms")
        if 'avg_epoch_time' in summary:
            print(f"  Avg epoch time: {summary['avg_epoch_time']:.2f}s")

        print("\n[Throughput]")
        if 'avg_samples_per_sec' in summary:
            print(f"  Avg throughput: {summary['avg_samples_per_sec']:.2f} samples/sec")
            print(f"  Peak throughput: {summary['peak_samples_per_sec']:.2f} samples/sec")
        if 'avg_tokens_per_sec' in summary:
            print(f"  Avg tokens/sec: {summary['avg_tokens_per_sec']:.2f}")
            print(f"  Peak tokens/sec: {summary['peak_tokens_per_sec']:.2f}")
        if 'avg_tflops' in summary:
            print(f"  Avg TFLOPS: {summary['avg_tflops']:.2f}")
            print(f"  Peak TFLOPS: {summary['peak_tflops']:.2f}")

        print("\n[Training Metrics]")
        if 'final_loss' in summary:
            print(f"  Final loss: {summary['final_loss']:.4f}")
            print(f"  Min loss: {summary['min_loss']:.4f}")
        if 'final_accuracy' in summary:
            print(f"  Final accuracy: {summary['final_accuracy']:.4f}")
            print(f"  Max accuracy: {summary['max_accuracy']:.4f}")

        print("\n[GPU Metrics]")
        if 'avg_gpu_utilization' in summary:
            print(f"  Avg GPU utilization: {summary['avg_gpu_utilization']:.1f}%")
            print(f"  Max GPU utilization: {summary['max_gpu_utilization']:.1f}%")
        if 'avg_gpu_memory_gb' in summary:
            print(f"  Avg GPU memory: {summary['avg_gpu_memory_gb']:.2f} GB")
            print(f"  Peak GPU memory: {summary['peak_gpu_memory_gb']:.2f} GB")
        if 'avg_gpu_temp' in summary:
            print(f"  Avg GPU temp: {summary['avg_gpu_temp']:.1f}°C")
            print(f"  Max GPU temp: {summary['max_gpu_temp']:.1f}°C")

        print("\n" + "=" * 80 + "\n")

    def save_results(self, format: str = 'all') -> None:
        """
        Save benchmark results to files.

        Args:
            format: Output format ('json', 'csv', 'all')
        """
        # Save summary as JSON
        if format in ['json', 'all']:
            summary = self.get_summary()
            summary_file = self.output_dir / f"{self.name}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to {summary_file}")

        # Save detailed metrics as CSV
        if format in ['csv', 'all']:
            csv_file = self.output_dir / f"{self.name}_metrics.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                header = ['iteration', 'timestamp', 'iteration_time']
                if self.metrics.samples_per_sec:
                    header.append('samples_per_sec')
                if self.metrics.tokens_per_sec:
                    header.append('tokens_per_sec')
                if self.metrics.tflops:
                    header.append('tflops')
                if self.metrics.losses:
                    header.append('loss')
                if self.metrics.accuracies:
                    header.append('accuracy')
                if self.metrics.gpu_utilization:
                    header.append('gpu_utilization')
                if self.metrics.gpu_memory_used:
                    header.append('gpu_memory_gb')

                writer.writerow(header)

                # Write data
                num_rows = len(self.metrics.iteration_times)
                for i in range(num_rows):
                    row = [i, self.metrics.timestamps[i], self.metrics.iteration_times[i]]

                    if self.metrics.samples_per_sec and i < len(self.metrics.samples_per_sec):
                        row.append(self.metrics.samples_per_sec[i])
                    if self.metrics.tokens_per_sec and i < len(self.metrics.tokens_per_sec):
                        row.append(self.metrics.tokens_per_sec[i])
                    if self.metrics.tflops and i < len(self.metrics.tflops):
                        row.append(self.metrics.tflops[i])
                    if self.metrics.losses and i < len(self.metrics.losses):
                        row.append(self.metrics.losses[i])
                    if self.metrics.accuracies and i < len(self.metrics.accuracies):
                        row.append(self.metrics.accuracies[i])
                    if self.metrics.gpu_utilization and i < len(self.metrics.gpu_utilization):
                        row.append(self.metrics.gpu_utilization[i])
                    if self.metrics.gpu_memory_used and i < len(self.metrics.gpu_memory_used):
                        row.append(self.metrics.gpu_memory_used[i])

                    writer.writerow(row)

            print(f"Detailed metrics saved to {csv_file}")


if __name__ == "__main__":
    # Example usage
    benchmark = PerformanceBenchmark("test_run")
    benchmark.configure(batch_size=32, sequence_length=512, model_flops=1e9)

    # Simulate training
    for epoch in range(3):
        benchmark.start_epoch()

        for i in range(100):
            benchmark.start_iteration()
            time.sleep(0.01)  # Simulate work
            benchmark.end_iteration()

            benchmark.record_loss(1.0 - i * 0.001)
            benchmark.record_accuracy(0.5 + i * 0.001)
            benchmark.record_gpu_metrics(utilization=85.0, memory_used=8.5, temperature=65.0)

        benchmark.end_epoch()

    benchmark.print_summary()
    benchmark.save_results()
