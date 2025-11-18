"""
GPU monitoring utilities for PyTorch Testing Framework.
Monitors GPU utilization, memory, temperature, and other metrics.
"""

import torch
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import subprocess
import re

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class GPUMetrics:
    """Container for GPU metrics."""
    utilization: float  # 0-100%
    memory_used: float  # GB
    memory_total: float  # GB
    memory_allocated: float  # GB (PyTorch allocated)
    temperature: float  # Celsius
    power_usage: float  # Watts
    clock_speed: float  # MHz


class GPUMonitor:
    """Monitor GPU metrics during training."""

    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize GPU monitor.

        Args:
            device_ids: List of GPU device IDs to monitor. If None, monitors all available GPUs.
        """
        self.device_ids = device_ids
        self.nvml_initialized = False

        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, GPU monitoring disabled")
            return

        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                print(f"WARNING: Failed to initialize NVML: {e}")
                print("Falling back to PyTorch-only monitoring")

        # Determine device IDs to monitor
        if self.device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))

        # Get GPU handles
        self.gpu_handles = {}
        if self.nvml_initialized:
            for device_id in self.device_ids:
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
                    self.gpu_handles[device_id] = handle
                except Exception as e:
                    print(f"WARNING: Failed to get handle for GPU {device_id}: {e}")

    def __del__(self):
        """Cleanup NVML."""
        if self.nvml_initialized:
            try:
                nvml.nvmlShutdown()
            except:
                pass

    def get_gpu_info(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get GPU information.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary of GPU information
        """
        if not torch.cuda.is_available():
            return {}

        info = {
            'name': torch.cuda.get_device_name(device_id),
            'capability': torch.cuda.get_device_capability(device_id),
            'memory_total_gb': torch.cuda.get_device_properties(device_id).total_memory / 1e9,
        }

        if self.nvml_initialized and device_id in self.gpu_handles:
            try:
                handle = self.gpu_handles[device_id]
                info['driver_version'] = nvml.nvmlSystemGetDriverVersion()
                info['cuda_version'] = nvml.nvmlSystemGetCudaDriverVersion()
            except Exception as e:
                pass

        return info

    def get_metrics(self, device_id: int = 0) -> Optional[GPUMetrics]:
        """
        Get current GPU metrics.

        Args:
            device_id: GPU device ID

        Returns:
            GPUMetrics object or None if unavailable
        """
        if not torch.cuda.is_available():
            return None

        # PyTorch metrics (always available)
        memory_allocated = torch.cuda.memory_allocated(device_id) / 1e9
        memory_total = torch.cuda.get_device_properties(device_id).total_memory / 1e9

        # Default values
        utilization = 0.0
        memory_used = memory_allocated
        temperature = 0.0
        power_usage = 0.0
        clock_speed = 0.0

        # NVML metrics (if available)
        if self.nvml_initialized and device_id in self.gpu_handles:
            try:
                handle = self.gpu_handles[device_id]

                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = float(util.gpu)

                # Memory
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = mem_info.used / 1e9

                # Temperature
                temperature = float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))

                # Power
                power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W

                # Clock speed
                clock_speed = float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))

            except Exception as e:
                # Silently fall back to PyTorch metrics
                pass

        return GPUMetrics(
            utilization=utilization,
            memory_used=memory_used,
            memory_total=memory_total,
            memory_allocated=memory_allocated,
            temperature=temperature,
            power_usage=power_usage,
            clock_speed=clock_speed
        )

    def get_all_metrics(self) -> Dict[int, GPUMetrics]:
        """
        Get metrics for all monitored GPUs.

        Returns:
            Dictionary mapping device_id -> GPUMetrics
        """
        metrics = {}
        for device_id in self.device_ids:
            gpu_metrics = self.get_metrics(device_id)
            if gpu_metrics is not None:
                metrics[device_id] = gpu_metrics
        return metrics

    def print_metrics(self, device_id: int = 0) -> None:
        """
        Print GPU metrics in a formatted way.

        Args:
            device_id: GPU device ID
        """
        metrics = self.get_metrics(device_id)
        if metrics is None:
            print(f"GPU {device_id}: No metrics available")
            return

        print(f"\nGPU {device_id} Metrics:")
        print(f"  Utilization: {metrics.utilization:.1f}%")
        print(f"  Memory Used: {metrics.memory_used:.2f} GB / {metrics.memory_total:.2f} GB "
              f"({100*metrics.memory_used/metrics.memory_total:.1f}%)")
        print(f"  Memory Allocated (PyTorch): {metrics.memory_allocated:.2f} GB")
        if metrics.temperature > 0:
            print(f"  Temperature: {metrics.temperature:.1f}°C")
        if metrics.power_usage > 0:
            print(f"  Power Usage: {metrics.power_usage:.1f} W")
        if metrics.clock_speed > 0:
            print(f"  Clock Speed: {metrics.clock_speed:.0f} MHz")

    def print_all_metrics(self) -> None:
        """Print metrics for all monitored GPUs."""
        print("\n" + "=" * 80)
        print("GPU Metrics")
        print("=" * 80)
        for device_id in self.device_ids:
            self.print_metrics(device_id)
        print("=" * 80 + "\n")

    def log_metrics_summary(self) -> str:
        """
        Get a one-line summary of GPU metrics.

        Returns:
            Formatted string with GPU metrics
        """
        summaries = []
        for device_id in self.device_ids:
            metrics = self.get_metrics(device_id)
            if metrics is not None:
                summary = (f"GPU{device_id}: {metrics.utilization:.0f}% | "
                          f"{metrics.memory_used:.1f}/{metrics.memory_total:.1f}GB | "
                          f"{metrics.temperature:.0f}°C")
                summaries.append(summary)

        return " | ".join(summaries) if summaries else "No GPU metrics"

    def reset_peak_memory_stats(self, device_id: Optional[int] = None) -> None:
        """
        Reset peak memory stats for PyTorch.

        Args:
            device_id: GPU device ID. If None, resets for all devices.
        """
        if device_id is not None:
            torch.cuda.reset_peak_memory_stats(device_id)
        else:
            for dev_id in self.device_ids:
                torch.cuda.reset_peak_memory_stats(dev_id)

    def get_memory_summary(self, device_id: int = 0) -> str:
        """
        Get PyTorch memory summary.

        Args:
            device_id: GPU device ID

        Returns:
            Memory summary string
        """
        if not torch.cuda.is_available():
            return "CUDA not available"

        return torch.cuda.memory_summary(device_id)

    def synchronize(self, device_id: Optional[int] = None) -> None:
        """
        Synchronize CUDA device(s).

        Args:
            device_id: GPU device ID. If None, synchronizes all devices.
        """
        if device_id is not None:
            torch.cuda.synchronize(device_id)
        else:
            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        """Empty PyTorch CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return system information.

    Returns:
        Dictionary with GPU availability info
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info['gpu_names'].append(torch.cuda.get_device_name(i))

    return info


def wait_for_gpu_memory(
    required_gb: float,
    device_id: int = 0,
    timeout: int = 300,
    check_interval: int = 5
) -> bool:
    """
    Wait for sufficient GPU memory to become available.

    Args:
        required_gb: Required free memory in GB
        device_id: GPU device ID
        timeout: Maximum time to wait in seconds
        check_interval: Check interval in seconds

    Returns:
        True if memory available, False if timeout
    """
    if not torch.cuda.is_available():
        return False

    monitor = GPUMonitor([device_id])
    start_time = time.time()

    while time.time() - start_time < timeout:
        metrics = monitor.get_metrics(device_id)
        if metrics is not None:
            free_memory = metrics.memory_total - metrics.memory_used
            if free_memory >= required_gb:
                return True

        time.sleep(check_interval)

    return False


if __name__ == "__main__":
    # Example usage
    print("Checking GPU availability...")
    gpu_info = check_gpu_availability()
    print(f"CUDA available: {gpu_info['cuda_available']}")
    print(f"Number of GPUs: {gpu_info['num_gpus']}")
    print(f"GPU names: {gpu_info['gpu_names']}")

    if gpu_info['cuda_available']:
        print("\nInitializing GPU monitor...")
        monitor = GPUMonitor()

        print("\nGPU Information:")
        for device_id in monitor.device_ids:
            info = monitor.get_gpu_info(device_id)
            print(f"\nGPU {device_id}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

        print("\nCurrent GPU Metrics:")
        monitor.print_all_metrics()

        print("\nOne-line summary:")
        print(monitor.log_metrics_summary())

        # Allocate some memory to test
        print("\nAllocating 1GB tensor...")
        x = torch.randn(256, 1024, 1024, device='cuda')
        print("After allocation:")
        monitor.print_all_metrics()

        # Clean up
        del x
        torch.cuda.empty_cache()
