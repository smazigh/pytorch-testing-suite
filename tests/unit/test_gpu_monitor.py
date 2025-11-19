"""
Unit tests for gpu_monitor module.
"""

import pytest
from unittest.mock import patch, MagicMock

from utils.gpu_monitor import (
    GPUMetrics,
    GPUMonitor,
    check_gpu_availability,
    wait_for_gpu_memory,
)


@pytest.mark.unit
class TestGPUMetrics:
    """Test GPUMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating GPUMetrics."""
        metrics = GPUMetrics(
            utilization=85.0,
            memory_used=8.5,
            memory_total=16.0,
            memory_allocated=7.0,
            temperature=70.0,
            power_usage=250.0,
            clock_speed=1500.0
        )

        assert metrics.utilization == 85.0
        assert metrics.memory_used == 8.5
        assert metrics.memory_total == 16.0
        assert metrics.memory_allocated == 7.0
        assert metrics.temperature == 70.0
        assert metrics.power_usage == 250.0
        assert metrics.clock_speed == 1500.0

    def test_metrics_default_values(self):
        """Test GPUMetrics with zero values."""
        metrics = GPUMetrics(
            utilization=0.0,
            memory_used=0.0,
            memory_total=0.0,
            memory_allocated=0.0,
            temperature=0.0,
            power_usage=0.0,
            clock_speed=0.0
        )

        assert metrics.utilization == 0.0
        assert metrics.memory_total == 0.0


@pytest.mark.unit
class TestGPUMonitor:
    """Test GPUMonitor class."""

    def test_monitor_creation_no_cuda(self, mock_gpu_available):
        """Test creating monitor when CUDA not available."""
        monitor = GPUMonitor()
        # Should not crash even without CUDA
        assert monitor is not None

    def test_get_gpu_info_no_cuda(self, mock_gpu_available):
        """Test get_gpu_info returns empty dict when no CUDA."""
        monitor = GPUMonitor()
        info = monitor.get_gpu_info()
        assert info == {}

    def test_get_metrics_no_cuda(self, mock_gpu_available):
        """Test get_metrics returns None when no CUDA."""
        monitor = GPUMonitor()
        metrics = monitor.get_metrics()
        assert metrics is None

    def test_get_all_metrics_no_cuda(self, mock_gpu_available):
        """Test get_all_metrics with no CUDA."""
        monitor = GPUMonitor()
        # Set device_ids to empty list since CUDA is not available
        monitor.device_ids = []
        metrics = monitor.get_all_metrics()
        assert metrics == {}

    def test_print_metrics_no_cuda(self, mock_gpu_available, capsys):
        """Test print_metrics when no CUDA."""
        monitor = GPUMonitor()
        monitor.print_metrics(0)
        captured = capsys.readouterr()
        assert "No metrics available" in captured.out

    def test_log_metrics_summary_no_cuda(self, mock_gpu_available):
        """Test log_metrics_summary when no CUDA."""
        monitor = GPUMonitor()
        monitor.device_ids = []
        summary = monitor.log_metrics_summary()
        assert summary == "No GPU metrics"

    def test_get_memory_summary_no_cuda(self, mock_gpu_available):
        """Test get_memory_summary when no CUDA."""
        monitor = GPUMonitor()
        summary = monitor.get_memory_summary()
        assert summary == "CUDA not available"

    def test_empty_cache_no_cuda(self, mock_gpu_available):
        """Test empty_cache doesn't crash when no CUDA."""
        monitor = GPUMonitor()
        # Should not raise any exception
        monitor.empty_cache()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    def test_monitor_with_device_ids(self, mock_count, mock_available):
        """Test monitor with specific device IDs."""
        monitor = GPUMonitor(device_ids=[0, 1])
        assert monitor.device_ids == [0, 1]

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.memory_allocated', return_value=4e9)
    @patch('torch.cuda.get_device_properties')
    def test_get_metrics_with_cuda(self, mock_props, mock_mem, mock_count, mock_available):
        """Test get_metrics with mocked CUDA."""
        # Setup mock device properties
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0])
        metrics = monitor.get_metrics(0)

        assert metrics is not None
        assert metrics.memory_total == 16.0
        assert metrics.memory_allocated == 4.0

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name', return_value='Test GPU')
    @patch('torch.cuda.get_device_capability', return_value=(8, 0))
    @patch('torch.cuda.get_device_properties')
    def test_get_gpu_info_with_cuda(self, mock_props, mock_cap, mock_name, mock_count, mock_available):
        """Test get_gpu_info with mocked CUDA."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0])
        info = monitor.get_gpu_info(0)

        assert info['name'] == 'Test GPU'
        assert info['capability'] == (8, 0)
        assert info['memory_total_gb'] == 16.0

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.memory_allocated', return_value=4e9)
    @patch('torch.cuda.get_device_properties')
    def test_get_all_metrics_with_cuda(self, mock_props, mock_mem, mock_count, mock_available):
        """Test get_all_metrics with mocked CUDA."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0, 1])
        all_metrics = monitor.get_all_metrics()

        assert len(all_metrics) == 2
        assert 0 in all_metrics
        assert 1 in all_metrics

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=4e9)
    @patch('torch.cuda.get_device_properties')
    def test_log_metrics_summary_with_cuda(self, mock_props, mock_mem, mock_count, mock_available):
        """Test log_metrics_summary with mocked CUDA."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0])
        summary = monitor.log_metrics_summary()

        assert "GPU0" in summary
        assert "16.0GB" in summary

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    @patch('torch.cuda.device_count', return_value=1)
    def test_reset_peak_memory_stats(self, mock_count, mock_reset, mock_available):
        """Test reset_peak_memory_stats."""
        monitor = GPUMonitor(device_ids=[0])
        monitor.reset_peak_memory_stats(0)
        mock_reset.assert_called_once_with(0)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    @patch('torch.cuda.device_count', return_value=2)
    def test_reset_peak_memory_stats_all(self, mock_count, mock_reset, mock_available):
        """Test reset_peak_memory_stats for all devices."""
        monitor = GPUMonitor(device_ids=[0, 1])
        monitor.reset_peak_memory_stats(None)
        assert mock_reset.call_count == 2

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.device_count', return_value=1)
    def test_synchronize(self, mock_count, mock_sync, mock_available):
        """Test synchronize."""
        monitor = GPUMonitor(device_ids=[0])
        monitor.synchronize(0)
        mock_sync.assert_called_once_with(0)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.device_count', return_value=1)
    def test_synchronize_all(self, mock_count, mock_sync, mock_available):
        """Test synchronize all devices."""
        monitor = GPUMonitor(device_ids=[0])
        monitor.synchronize(None)
        mock_sync.assert_called_once_with()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.device_count', return_value=1)
    def test_empty_cache_with_cuda(self, mock_count, mock_empty, mock_available):
        """Test empty_cache with CUDA."""
        monitor = GPUMonitor(device_ids=[0])
        monitor.empty_cache()
        mock_empty.assert_called_once()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_summary', return_value="Memory Summary")
    @patch('torch.cuda.device_count', return_value=1)
    def test_get_memory_summary_with_cuda(self, mock_count, mock_summary, mock_available):
        """Test get_memory_summary with CUDA."""
        monitor = GPUMonitor(device_ids=[0])
        summary = monitor.get_memory_summary(0)
        assert summary == "Memory Summary"


@pytest.mark.unit
class TestCheckGPUAvailability:
    """Test check_gpu_availability function."""

    def test_check_no_cuda(self, mock_gpu_available):
        """Test check_gpu_availability when no CUDA."""
        info = check_gpu_availability()

        assert info['cuda_available'] is False
        assert info['cuda_version'] is None
        assert info['cudnn_version'] is None
        assert info['num_gpus'] == 0
        assert info['gpu_names'] == []

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.get_device_name')
    @patch('torch.version.cuda', '11.8')
    @patch('torch.backends.cudnn.version', return_value=8600)
    def test_check_with_cuda(self, mock_cudnn, mock_name, mock_count, mock_available):
        """Test check_gpu_availability with CUDA."""
        mock_name.side_effect = ['GPU 0', 'GPU 1']

        info = check_gpu_availability()

        assert info['cuda_available'] is True
        assert info['num_gpus'] == 2
        assert len(info['gpu_names']) == 2


@pytest.mark.unit
class TestWaitForGPUMemory:
    """Test wait_for_gpu_memory function."""

    def test_wait_no_cuda(self, mock_gpu_available):
        """Test wait_for_gpu_memory when no CUDA."""
        result = wait_for_gpu_memory(required_gb=1.0, timeout=1)
        assert result is False

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=4e9)
    @patch('torch.cuda.get_device_properties')
    def test_wait_memory_available(self, mock_props, mock_mem, mock_count, mock_available):
        """Test wait_for_gpu_memory when memory is available."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        result = wait_for_gpu_memory(
            required_gb=1.0,
            device_id=0,
            timeout=1,
            check_interval=0.1
        )
        # With 16GB total and 4GB used, we have 12GB free
        assert result is True

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=15e9)
    @patch('torch.cuda.get_device_properties')
    def test_wait_memory_not_available(self, mock_props, mock_mem, mock_count, mock_available):
        """Test wait_for_gpu_memory when memory is not available."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        result = wait_for_gpu_memory(
            required_gb=10.0,
            device_id=0,
            timeout=0.2,
            check_interval=0.1
        )
        # With 16GB total and 15GB used, we only have 1GB free
        assert result is False


@pytest.mark.unit
class TestGPUMonitorPrintAllMetrics:
    """Test print_all_metrics method."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.memory_allocated', return_value=4e9)
    @patch('torch.cuda.get_device_properties')
    def test_print_all_metrics(self, mock_props, mock_mem, mock_count, mock_available, capsys):
        """Test print_all_metrics output."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0, 1])
        monitor.print_all_metrics()

        captured = capsys.readouterr()
        assert "GPU Metrics" in captured.out
        assert "GPU 0 Metrics" in captured.out
        assert "GPU 1 Metrics" in captured.out


@pytest.mark.unit
class TestGPUMonitorAdvanced:
    """Advanced tests for GPUMonitor to improve coverage."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=4e9)
    @patch('torch.cuda.get_device_properties')
    def test_print_metrics_with_cuda(self, mock_props, mock_mem, mock_count, mock_available, capsys):
        """Test print_metrics with mocked CUDA."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0])
        monitor.print_metrics(0)

        captured = capsys.readouterr()
        # Should print metrics without crashing
        assert len(captured.out) > 0

    def test_print_all_metrics_no_cuda(self, mock_gpu_available, capsys):
        """Test print_all_metrics when no CUDA."""
        monitor = GPUMonitor()
        monitor.device_ids = []
        monitor.print_all_metrics()

        captured = capsys.readouterr()
        assert "GPU Metrics" in captured.out

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=0)
    def test_monitor_with_no_devices(self, mock_count, mock_available):
        """Test monitor when CUDA is available but no devices."""
        monitor = GPUMonitor()
        assert monitor.device_ids == []

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=4e9)
    @patch('torch.cuda.get_device_properties')
    def test_get_metrics_default_device(self, mock_props, mock_mem, mock_count, mock_available):
        """Test get_metrics with default device (None)."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0])
        metrics = monitor.get_metrics()  # Default device

        assert metrics is not None
        assert metrics.memory_total == 16.0

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    def test_monitor_auto_detect_devices(self, mock_count, mock_available):
        """Test monitor auto-detects all devices when not specified."""
        monitor = GPUMonitor()
        assert len(monitor.device_ids) == 2
        assert monitor.device_ids == [0, 1]

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_summary', return_value="Detailed Memory Summary")
    @patch('torch.cuda.device_count', return_value=1)
    def test_get_memory_summary_default_device(self, mock_count, mock_summary, mock_available):
        """Test get_memory_summary with default device."""
        monitor = GPUMonitor(device_ids=[0])
        summary = monitor.get_memory_summary()  # Default device
        assert summary == "Detailed Memory Summary"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.device_count', return_value=2)
    def test_synchronize_all_devices(self, mock_count, mock_sync, mock_available):
        """Test synchronize with device_id=None synchronizes all."""
        monitor = GPUMonitor(device_ids=[0, 1])
        monitor.synchronize(None)
        mock_sync.assert_called_once()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=8e9)
    @patch('torch.cuda.get_device_properties')
    def test_log_metrics_summary_format(self, mock_props, mock_mem, mock_count, mock_available):
        """Test log_metrics_summary returns properly formatted string."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0])
        summary = monitor.log_metrics_summary()

        assert "GPU0" in summary
        assert "8.0" in summary  # allocated memory
        assert "16.0" in summary  # total memory


@pytest.mark.unit
class TestGPUMonitorNVML:
    """Tests for NVML-related functionality."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    def test_monitor_nvml_not_initialized(self, mock_count, mock_available):
        """Test monitor when NVML is not initialized."""
        monitor = GPUMonitor(device_ids=[0])
        monitor.nvml_initialized = False
        assert monitor.device_ids == [0]

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=2e9)
    @patch('torch.cuda.get_device_properties')
    def test_get_metrics_no_nvml(self, mock_props, mock_mem, mock_count, mock_available):
        """Test get_metrics falls back to PyTorch when NVML unavailable."""
        mock_props.return_value = MagicMock(total_memory=8e9)

        monitor = GPUMonitor(device_ids=[0])
        monitor.nvml_initialized = False

        metrics = monitor.get_metrics(0)

        assert metrics is not None
        assert metrics.memory_allocated == 2.0
        assert metrics.memory_total == 8.0
        assert metrics.utilization == 0.0

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name', return_value='Test GPU')
    @patch('torch.cuda.get_device_capability', return_value=(7, 5))
    @patch('torch.cuda.get_device_properties')
    def test_get_gpu_info_no_nvml(self, mock_props, mock_cap, mock_name, mock_count, mock_available):
        """Test get_gpu_info without NVML."""
        mock_props.return_value = MagicMock(total_memory=12e9)

        monitor = GPUMonitor(device_ids=[0])
        monitor.nvml_initialized = False

        info = monitor.get_gpu_info(0)

        assert info['name'] == 'Test GPU'
        assert info['capability'] == (7, 5)
        assert info['memory_total_gb'] == 12.0

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.memory_allocated', return_value=4e9)
    @patch('torch.cuda.get_device_properties')
    def test_print_metrics_with_zeros(self, mock_props, mock_mem, mock_count, mock_available, capsys):
        """Test print_metrics when temperature/power are zero."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0])
        monitor.nvml_initialized = False

        monitor.print_metrics(0)
        captured = capsys.readouterr()

        assert "GPU 0 Metrics" in captured.out
        assert "Memory Used" in captured.out

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.memory_allocated', return_value=3e9)
    @patch('torch.cuda.get_device_properties')
    def test_log_metrics_summary_multiple_gpus(self, mock_props, mock_mem, mock_count, mock_available):
        """Test log_metrics_summary with multiple GPUs."""
        mock_props.return_value = MagicMock(total_memory=16e9)

        monitor = GPUMonitor(device_ids=[0, 1])
        summary = monitor.log_metrics_summary()

        assert "GPU0" in summary
        assert "GPU1" in summary

    def test_destructor_no_crash(self, mock_gpu_available):
        """Test that destructor doesn't crash."""
        monitor = GPUMonitor()
        monitor.nvml_initialized = False
        del monitor


@pytest.mark.unit
class TestGPUMonitorExtendedCoverage:
    """Extended tests to improve coverage for GPUMonitor."""

    def test_print_metrics_with_temperature_power_clock(self, capsys):
        """Test print_metrics displays temperature, power, and clock when non-zero."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=4e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value = MagicMock(total_memory=16e9)

            monitor = GPUMonitor(device_ids=[0])

            # Create a mock metrics object with non-zero values
            mock_metrics = GPUMetrics(
                utilization=75.0,
                memory_used=8.0,
                memory_total=16.0,
                memory_allocated=4.0,
                temperature=65.0,
                power_usage=200.0,
                clock_speed=1800.0
            )

            # Patch get_metrics to return our custom metrics
            with patch.object(monitor, 'get_metrics', return_value=mock_metrics):
                monitor.print_metrics(0)

            captured = capsys.readouterr()
            assert "Temperature" in captured.out
            assert "65.0" in captured.out
            assert "Power Usage" in captured.out
            assert "200.0" in captured.out
            assert "Clock Speed" in captured.out
            assert "1800" in captured.out

    def test_monitor_init_with_cuda_available(self):
        """Test GPUMonitor initialization when CUDA is available."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=2), \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            monitor = GPUMonitor()
            assert monitor.device_ids == [0, 1]
            assert monitor.nvml_initialized is False

    def test_monitor_init_with_specific_device_ids(self):
        """Test GPUMonitor with specific device IDs."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=4), \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            monitor = GPUMonitor(device_ids=[1, 3])
            assert monitor.device_ids == [1, 3]

    def test_get_gpu_info_with_cuda(self):
        """Test get_gpu_info returns correct info when CUDA available."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.get_device_name', return_value='NVIDIA RTX 3080'), \
             patch('utils.gpu_monitor.torch.cuda.get_device_capability', return_value=(8, 6)), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            mock_props.return_value = MagicMock(total_memory=10e9)

            monitor = GPUMonitor(device_ids=[0])
            info = monitor.get_gpu_info(0)

            assert info['name'] == 'NVIDIA RTX 3080'
            assert info['capability'] == (8, 6)
            assert info['memory_total_gb'] == 10.0

    def test_get_metrics_with_cuda_no_nvml(self):
        """Test get_metrics returns PyTorch-only metrics when NVML unavailable."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=2e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            mock_props.return_value = MagicMock(total_memory=8e9)

            monitor = GPUMonitor(device_ids=[0])
            metrics = monitor.get_metrics(0)

            assert metrics is not None
            assert metrics.memory_allocated == 2.0
            assert metrics.memory_total == 8.0
            assert metrics.utilization == 0.0  # No NVML

    def test_get_all_metrics_with_multiple_devices(self):
        """Test get_all_metrics returns metrics for all devices."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=2), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=1e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            mock_props.return_value = MagicMock(total_memory=4e9)

            monitor = GPUMonitor(device_ids=[0, 1])
            all_metrics = monitor.get_all_metrics()

            assert len(all_metrics) == 2
            assert 0 in all_metrics
            assert 1 in all_metrics

    def test_print_metrics_with_cuda(self, capsys):
        """Test print_metrics displays correct output with CUDA."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=4e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            mock_props.return_value = MagicMock(total_memory=16e9)

            monitor = GPUMonitor(device_ids=[0])
            monitor.print_metrics(0)

            captured = capsys.readouterr()
            assert "GPU 0 Metrics" in captured.out
            assert "Utilization" in captured.out
            assert "Memory Used" in captured.out

    def test_print_all_metrics_with_cuda(self, capsys):
        """Test print_all_metrics displays all GPUs."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=2), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=2e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            mock_props.return_value = MagicMock(total_memory=8e9)

            monitor = GPUMonitor(device_ids=[0, 1])
            monitor.print_all_metrics()

            captured = capsys.readouterr()
            assert "GPU Metrics" in captured.out
            assert "GPU 0 Metrics" in captured.out
            assert "GPU 1 Metrics" in captured.out

    def test_log_metrics_summary_with_cuda(self):
        """Test log_metrics_summary returns formatted string."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=3e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            mock_props.return_value = MagicMock(total_memory=12e9)

            monitor = GPUMonitor(device_ids=[0])
            summary = monitor.log_metrics_summary()

            assert "GPU0" in summary
            assert "12.0GB" in summary

    def test_reset_peak_memory_stats_single_device(self):
        """Test reset_peak_memory_stats for single device."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.reset_peak_memory_stats') as mock_reset, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            monitor = GPUMonitor(device_ids=[0])
            monitor.reset_peak_memory_stats(0)
            mock_reset.assert_called_once_with(0)

    def test_reset_peak_memory_stats_all_devices(self):
        """Test reset_peak_memory_stats for all devices."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=3), \
             patch('utils.gpu_monitor.torch.cuda.reset_peak_memory_stats') as mock_reset, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            monitor = GPUMonitor(device_ids=[0, 1, 2])
            monitor.reset_peak_memory_stats(None)
            assert mock_reset.call_count == 3

    def test_get_memory_summary_with_cuda(self):
        """Test get_memory_summary returns memory summary string."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.memory_summary', return_value="Test Memory Summary"), \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            monitor = GPUMonitor(device_ids=[0])
            summary = monitor.get_memory_summary(0)
            assert summary == "Test Memory Summary"

    def test_synchronize_single_device(self):
        """Test synchronize for single device."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.synchronize') as mock_sync, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            monitor = GPUMonitor(device_ids=[0])
            monitor.synchronize(0)
            mock_sync.assert_called_once_with(0)

    def test_synchronize_all_devices(self):
        """Test synchronize for all devices."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=2), \
             patch('utils.gpu_monitor.torch.cuda.synchronize') as mock_sync, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            monitor = GPUMonitor(device_ids=[0, 1])
            monitor.synchronize(None)
            mock_sync.assert_called_once_with()

    def test_empty_cache_with_cuda(self):
        """Test empty_cache calls torch.cuda.empty_cache."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.empty_cache') as mock_empty, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            monitor = GPUMonitor(device_ids=[0])
            monitor.empty_cache()
            mock_empty.assert_called_once()

    def test_destructor_with_nvml(self):
        """Test destructor shuts down NVML."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.NVML_AVAILABLE', True), \
             patch('utils.gpu_monitor.nvml.nvmlInit'), \
             patch('utils.gpu_monitor.nvml.nvmlDeviceGetHandleByIndex'), \
             patch('utils.gpu_monitor.nvml.nvmlShutdown') as mock_shutdown:
            monitor = GPUMonitor(device_ids=[0])
            monitor.nvml_initialized = True
            del monitor
            # Note: __del__ may be called by garbage collector

    def test_get_metrics_with_nvml_data(self):
        """Test get_metrics with NVML providing full data."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=4e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', True), \
             patch('utils.gpu_monitor.nvml.nvmlInit'), \
             patch('utils.gpu_monitor.nvml.nvmlDeviceGetHandleByIndex') as mock_handle:
            mock_props.return_value = MagicMock(total_memory=16e9)
            mock_handle.return_value = MagicMock()

            monitor = GPUMonitor(device_ids=[0])

            # Setup NVML mocks for metrics
            mock_util = MagicMock()
            mock_util.gpu = 85
            mock_mem_info = MagicMock()
            mock_mem_info.used = 10e9

            with patch('utils.gpu_monitor.nvml.nvmlDeviceGetUtilizationRates', return_value=mock_util), \
                 patch('utils.gpu_monitor.nvml.nvmlDeviceGetMemoryInfo', return_value=mock_mem_info), \
                 patch('utils.gpu_monitor.nvml.nvmlDeviceGetTemperature', return_value=70), \
                 patch('utils.gpu_monitor.nvml.nvmlDeviceGetPowerUsage', return_value=250000), \
                 patch('utils.gpu_monitor.nvml.nvmlDeviceGetClockInfo', return_value=1500):
                metrics = monitor.get_metrics(0)

            assert metrics.utilization == 85.0
            assert metrics.memory_used == 10.0
            assert metrics.temperature == 70.0
            assert metrics.power_usage == 250.0
            assert metrics.clock_speed == 1500.0

    def test_get_gpu_info_with_nvml(self):
        """Test get_gpu_info includes NVML driver info."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.get_device_name', return_value='Test GPU'), \
             patch('utils.gpu_monitor.torch.cuda.get_device_capability', return_value=(8, 0)), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', True), \
             patch('utils.gpu_monitor.nvml.nvmlInit'), \
             patch('utils.gpu_monitor.nvml.nvmlDeviceGetHandleByIndex') as mock_handle:
            mock_props.return_value = MagicMock(total_memory=16e9)
            mock_handle.return_value = MagicMock()

            monitor = GPUMonitor(device_ids=[0])

            with patch('utils.gpu_monitor.nvml.nvmlSystemGetDriverVersion', return_value='525.60.11'):
                info = monitor.get_gpu_info(0)

            assert info['name'] == 'Test GPU'
            assert info['driver_version'] == '525.60.11'

    def test_get_gpu_info_nvml_exception(self):
        """Test get_gpu_info handles NVML exceptions gracefully."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.get_device_name', return_value='Test GPU'), \
             patch('utils.gpu_monitor.torch.cuda.get_device_capability', return_value=(8, 0)), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', True), \
             patch('utils.gpu_monitor.nvml.nvmlInit'), \
             patch('utils.gpu_monitor.nvml.nvmlDeviceGetHandleByIndex') as mock_handle:
            mock_props.return_value = MagicMock(total_memory=16e9)
            mock_handle.return_value = MagicMock()

            monitor = GPUMonitor(device_ids=[0])

            with patch('utils.gpu_monitor.nvml.nvmlSystemGetDriverVersion', side_effect=Exception("Error")):
                info = monitor.get_gpu_info(0)

            assert info['name'] == 'Test GPU'
            assert 'driver_version' not in info

    def test_nvml_init_failure(self, capsys):
        """Test GPUMonitor handles NVML init failure gracefully."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.NVML_AVAILABLE', True), \
             patch('utils.gpu_monitor.nvml.nvmlInit', side_effect=Exception("Init failed")):
            monitor = GPUMonitor(device_ids=[0])

            captured = capsys.readouterr()
            assert "Failed to initialize NVML" in captured.out
            assert monitor.nvml_initialized is False

    def test_nvml_handle_failure(self, capsys):
        """Test GPUMonitor handles NVML handle failure gracefully."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.NVML_AVAILABLE', True), \
             patch('utils.gpu_monitor.nvml.nvmlInit'), \
             patch('utils.gpu_monitor.nvml.nvmlDeviceGetHandleByIndex', side_effect=Exception("Handle error")):
            monitor = GPUMonitor(device_ids=[0])

            captured = capsys.readouterr()
            assert "Failed to get handle" in captured.out


@pytest.mark.unit
class TestCheckGPUAvailabilityExtended:
    """Extended tests for check_gpu_availability function."""

    def test_check_with_cuda_available(self):
        """Test check_gpu_availability with CUDA available."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=2), \
             patch('utils.gpu_monitor.torch.cuda.get_device_name') as mock_name, \
             patch('utils.gpu_monitor.torch.version.cuda', '11.8'), \
             patch('utils.gpu_monitor.torch.backends.cudnn.version', return_value=8600):
            mock_name.side_effect = ['GPU 0', 'GPU 1']

            info = check_gpu_availability()

            assert info['cuda_available'] is True
            assert info['num_gpus'] == 2
            assert len(info['gpu_names']) == 2


@pytest.mark.unit
class TestWaitForGPUMemoryExtended:
    """Extended tests for wait_for_gpu_memory function."""

    def test_wait_memory_available_immediately(self):
        """Test wait_for_gpu_memory when memory available immediately."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=4e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            mock_props.return_value = MagicMock(total_memory=16e9)

            result = wait_for_gpu_memory(
                required_gb=8.0,
                device_id=0,
                timeout=1,
                check_interval=0.1
            )
            assert result is True

    def test_wait_memory_timeout(self):
        """Test wait_for_gpu_memory times out when not enough memory."""
        with patch('utils.gpu_monitor.torch.cuda.is_available', return_value=True), \
             patch('utils.gpu_monitor.torch.cuda.device_count', return_value=1), \
             patch('utils.gpu_monitor.torch.cuda.memory_allocated', return_value=15e9), \
             patch('utils.gpu_monitor.torch.cuda.get_device_properties') as mock_props, \
             patch('utils.gpu_monitor.NVML_AVAILABLE', False):
            mock_props.return_value = MagicMock(total_memory=16e9)

            result = wait_for_gpu_memory(
                required_gb=10.0,
                device_id=0,
                timeout=0.2,
                check_interval=0.1
            )
            assert result is False
