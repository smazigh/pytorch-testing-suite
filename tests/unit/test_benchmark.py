"""
Unit tests for benchmark module.
"""

import time
import pytest
import numpy as np
from pathlib import Path

from utils.benchmark import PerformanceBenchmark, BenchmarkMetrics


@pytest.mark.unit
class TestBenchmarkMetrics:
    """Test BenchmarkMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating BenchmarkMetrics."""
        metrics = BenchmarkMetrics()

        assert metrics.iteration_times == []
        assert metrics.losses == []
        assert metrics.accuracies == []
        assert metrics.gpu_utilization == []

    def test_metrics_with_data(self):
        """Test BenchmarkMetrics with data."""
        metrics = BenchmarkMetrics()

        metrics.iteration_times.append(0.1)
        metrics.losses.append(0.5)
        metrics.accuracies.append(0.9)

        assert len(metrics.iteration_times) == 1
        assert metrics.losses[0] == 0.5


@pytest.mark.unit
class TestPerformanceBenchmark:
    """Test PerformanceBenchmark class."""

    def test_benchmark_creation(self, temp_dir):
        """Test creating benchmark."""
        benchmark = PerformanceBenchmark(
            name='test_benchmark',
            output_dir=str(temp_dir)
        )

        assert benchmark.name == 'test_benchmark'
        assert benchmark.metrics is not None

    def test_configure(self, temp_dir):
        """Test configuring benchmark."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))

        benchmark.configure(
            batch_size=32,
            sequence_length=128,
            model_flops=1e9
        )

        assert benchmark.batch_size == 32
        assert benchmark.sequence_length == 128
        assert benchmark.model_flops == 1e9

    def test_iteration_timing(self, temp_dir):
        """Test iteration timing."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))

        benchmark.start_iteration()
        time.sleep(0.01)  # Simulate work
        iter_time = benchmark.end_iteration()

        assert iter_time > 0
        assert len(benchmark.metrics.iteration_times) == 1

    def test_epoch_timing(self, temp_dir):
        """Test epoch timing."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))

        benchmark.start_epoch()
        time.sleep(0.01)  # Simulate work
        epoch_time = benchmark.end_epoch()

        assert epoch_time > 0
        assert len(benchmark.metrics.epoch_times) == 1

    def test_record_metrics(self, temp_dir):
        """Test recording various metrics."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))

        benchmark.record_loss(0.5)
        benchmark.record_accuracy(0.95)
        benchmark.record_gpu_metrics(
            utilization=85.0,
            memory_used=8.5,
            memory_allocated=10.0,
            temperature=70.0
        )

        assert len(benchmark.metrics.losses) == 1
        assert benchmark.metrics.losses[0] == 0.5
        assert benchmark.metrics.accuracies[0] == 0.95
        assert benchmark.metrics.gpu_utilization[0] == 85.0

    def test_throughput_calculation(self, temp_dir):
        """Test throughput calculation."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))
        benchmark.configure(batch_size=32)

        # Simulate iterations
        for _ in range(10):
            benchmark.start_iteration()
            time.sleep(0.01)
            benchmark.end_iteration()

        throughput = benchmark.get_throughput(32)
        assert throughput > 0

    def test_get_summary(self, temp_dir):
        """Test getting summary statistics."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))
        benchmark.configure(batch_size=32, sequence_length=128)

        # Simulate some iterations
        for i in range(10):
            benchmark.start_iteration()
            time.sleep(0.01)
            benchmark.end_iteration()
            benchmark.record_loss(1.0 - i * 0.05)
            benchmark.record_accuracy(0.5 + i * 0.05)

        summary = benchmark.get_summary()

        assert 'total_time' in summary
        assert 'num_iterations' in summary
        assert 'avg_iteration_time' in summary
        assert 'avg_samples_per_sec' in summary
        assert 'final_loss' in summary
        assert 'final_accuracy' in summary

    def test_summary_statistics(self, temp_dir):
        """Test correctness of summary statistics."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))

        # Add known values
        benchmark.metrics.iteration_times = [0.1, 0.2, 0.3]
        benchmark.metrics.losses = [1.0, 0.5, 0.3]
        benchmark.metrics.accuracies = [0.5, 0.7, 0.9]

        summary = benchmark.get_summary()

        assert summary['avg_iteration_time'] == pytest.approx(0.2)
        assert summary['min_iteration_time'] == 0.1
        assert summary['max_iteration_time'] == 0.3
        assert summary['final_loss'] == 0.3
        assert summary['final_accuracy'] == 0.9

    def test_save_results_json(self, temp_dir):
        """Test saving results as JSON."""
        benchmark = PerformanceBenchmark(
            name='test',
            output_dir=str(temp_dir)
        )

        # Add some data
        benchmark.start_iteration()
        benchmark.end_iteration()
        benchmark.record_loss(0.5)

        # Save
        benchmark.save_results(format='json')

        # Check file exists
        json_file = temp_dir / 'test_summary.json'
        assert json_file.exists()

    def test_save_results_csv(self, temp_dir):
        """Test saving results as CSV."""
        benchmark = PerformanceBenchmark(
            name='test',
            output_dir=str(temp_dir)
        )

        # Add some data
        for _ in range(5):
            benchmark.start_iteration()
            time.sleep(0.01)
            benchmark.end_iteration()
            benchmark.record_loss(0.5)

        # Save
        benchmark.save_results(format='csv')

        # Check file exists
        csv_file = temp_dir / 'test_metrics.csv'
        assert csv_file.exists()

    def test_tflops_calculation(self, temp_dir):
        """Test TFLOPS calculation."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))
        benchmark.configure(
            batch_size=32,
            model_flops=1e12  # 1 TFLOP
        )

        benchmark.start_iteration()
        time.sleep(0.01)  # 10ms
        benchmark.end_iteration()

        # Should have calculated TFLOPS
        assert len(benchmark.metrics.tflops) > 0
        assert benchmark.metrics.tflops[0] > 0

    def test_tokens_per_sec_calculation(self, temp_dir):
        """Test tokens/sec calculation for sequence models."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))
        benchmark.configure(
            batch_size=16,
            sequence_length=512
        )

        benchmark.start_iteration()
        time.sleep(0.01)
        benchmark.end_iteration()

        assert len(benchmark.metrics.tokens_per_sec) > 0
        # tokens/sec = (batch_size * seq_length) / time
        expected_min = (16 * 512) / 0.02  # Conservative estimate
        assert benchmark.metrics.tokens_per_sec[0] > expected_min


@pytest.mark.unit
class TestBenchmarkEdgeCases:
    """Test edge cases for benchmark."""

    def test_empty_metrics(self, temp_dir):
        """Test summary with no metrics."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))
        summary = benchmark.get_summary()

        assert summary['total_time'] >= 0
        assert summary['num_iterations'] == 0

    def test_zero_batch_size(self, temp_dir):
        """Test with zero batch size."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))
        benchmark.configure(batch_size=0)

        benchmark.start_iteration()
        benchmark.end_iteration()

        throughput = benchmark.get_throughput(0)
        assert throughput == 0

    def test_negative_values_handled(self, temp_dir):
        """Test that negative values don't break calculations."""
        benchmark = PerformanceBenchmark(output_dir=str(temp_dir))

        # Record some metrics
        benchmark.record_loss(-0.1)  # Shouldn't happen, but test handling
        benchmark.record_accuracy(-0.5)

        summary = benchmark.get_summary()
        assert 'final_loss' in summary
        assert 'final_accuracy' in summary
