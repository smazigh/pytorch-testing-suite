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
