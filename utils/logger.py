"""
Enhanced logging system for PyTorch Testing Framework.
Provides verbose logging with progress tracking and formatting.
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import json


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format the message
        record.levelname = f"{log_color}{record.levelname}{reset}"
        return super().format(record)


class PerformanceLogger:
    """Logger for tracking performance metrics and progress."""

    def __init__(
        self,
        name: str = "pytorch_test",
        log_dir: Optional[str] = None,
        level: str = "INFO",
        rank: int = 0
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_dir: Directory to save log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            rank: Process rank for distributed training (only rank 0 logs to file)
        """
        self.name = name
        self.rank = rank
        self.start_time = time.time()
        self.iteration_times = []
        self.metrics_history = []

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers

        # Console handler (all ranks)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (only rank 0)
        if rank == 0 and log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # JSON metrics file
            self.metrics_file = log_dir / f"{name}_metrics_{timestamp}.json"
        else:
            self.metrics_file = None

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    def log_header(self, title: str) -> None:
        """Log a section header."""
        separator = "=" * 80
        self.info(f"\n{separator}")
        self.info(f"  {title}")
        self.info(f"{separator}\n")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters."""
        self.log_header("Configuration")
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"  {sub_key}: {sub_value}")
            else:
                self.info(f"{key}: {value}")
        self.info("")

    def log_metrics(
        self,
        epoch: int,
        iteration: int,
        metrics: Dict[str, float],
        prefix: str = ""
    ) -> None:
        """
        Log training/evaluation metrics.

        Args:
            epoch: Current epoch
            iteration: Current iteration
            metrics: Dictionary of metric name -> value
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        elapsed = time.time() - self.start_time
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        if prefix:
            message = f"[{prefix}] Epoch {epoch} | Iter {iteration} | {metric_str} | Elapsed: {elapsed:.1f}s"
        else:
            message = f"Epoch {epoch} | Iter {iteration} | {metric_str} | Elapsed: {elapsed:.1f}s"

        self.info(message)

        # Store metrics
        metric_record = {
            'timestamp': time.time(),
            'epoch': epoch,
            'iteration': iteration,
            'elapsed': elapsed,
            'prefix': prefix,
            **metrics
        }
        self.metrics_history.append(metric_record)

    def log_progress(
        self,
        current: int,
        total: int,
        metrics: Optional[Dict[str, float]] = None,
        prefix: str = ""
    ) -> None:
        """
        Log progress with optional metrics.

        Args:
            current: Current progress value
            total: Total value
            metrics: Optional metrics to display
            prefix: Prefix for the progress bar
        """
        percentage = 100.0 * current / total if total > 0 else 0
        metric_str = ""
        if metrics:
            metric_str = " | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        message = f"{prefix} [{current}/{total}] {percentage:.1f}%{metric_str}"
        self.info(message)

    def create_progress_bar(
        self,
        total: int,
        desc: str = "Progress",
        disable: bool = False
    ) -> tqdm:
        """
        Create a tqdm progress bar.

        Args:
            total: Total iterations
            desc: Description
            disable: Disable progress bar (useful for non-rank-0 processes)

        Returns:
            tqdm progress bar
        """
        return tqdm(
            total=total,
            desc=desc,
            disable=disable or (self.rank != 0),
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

    def log_system_info(self, info: Dict[str, Any]) -> None:
        """Log system information."""
        self.log_header("System Information")
        for key, value in info.items():
            self.info(f"{key}: {value}")
        self.info("")

    def log_gpu_info(self, gpu_info: Dict[str, Any]) -> None:
        """Log GPU information."""
        self.log_header("GPU Information")
        for gpu_id, info in gpu_info.items():
            self.info(f"GPU {gpu_id}:")
            for key, value in info.items():
                self.info(f"  {key}: {value}")
        self.info("")

    def start_iteration(self) -> None:
        """Mark the start of an iteration."""
        self.iteration_start = time.time()

    def end_iteration(self) -> float:
        """
        Mark the end of an iteration.

        Returns:
            Iteration time in seconds
        """
        iteration_time = time.time() - self.iteration_start
        self.iteration_times.append(iteration_time)
        return iteration_time

    def get_avg_iteration_time(self, last_n: int = 100) -> float:
        """
        Get average iteration time.

        Args:
            last_n: Average over last N iterations

        Returns:
            Average iteration time in seconds
        """
        if not self.iteration_times:
            return 0.0
        times = self.iteration_times[-last_n:]
        return sum(times) / len(times)

    def get_throughput(self, batch_size: int, last_n: int = 100) -> float:
        """
        Calculate throughput (samples/sec).

        Args:
            batch_size: Batch size
            last_n: Calculate over last N iterations

        Returns:
            Throughput in samples/sec
        """
        avg_time = self.get_avg_iteration_time(last_n)
        if avg_time == 0:
            return 0.0
        return batch_size / avg_time

    def save_metrics(self) -> None:
        """Save metrics history to JSON file."""
        if self.rank == 0 and self.metrics_file is not None:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            self.info(f"Metrics saved to {self.metrics_file}")

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log final summary."""
        self.log_header("Summary")
        total_time = time.time() - self.start_time

        self.info(f"Total elapsed time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

        if self.iteration_times:
            avg_iter = self.get_avg_iteration_time()
            self.info(f"Average iteration time: {avg_iter*1000:.2f}ms")

        for key, value in summary.items():
            if isinstance(value, float):
                self.info(f"{key}: {value:.4f}")
            else:
                self.info(f"{key}: {value}")

        self.info("")
        self.save_metrics()


def get_logger(
    name: str = "pytorch_test",
    log_dir: Optional[str] = None,
    level: str = "INFO",
    rank: int = 0
) -> PerformanceLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        log_dir: Directory to save logs
        level: Logging level
        rank: Process rank

    Returns:
        PerformanceLogger instance
    """
    return PerformanceLogger(name=name, log_dir=log_dir, level=level, rank=rank)


if __name__ == "__main__":
    # Example usage
    logger = get_logger("test", log_dir="./logs")

    logger.log_header("Test Run")
    logger.info("Starting test...")

    # Simulate training loop
    for epoch in range(3):
        pbar = logger.create_progress_bar(10, desc=f"Epoch {epoch}")
        for i in range(10):
            logger.start_iteration()
            time.sleep(0.1)  # Simulate work
            iter_time = logger.end_iteration()

            metrics = {'loss': 0.5 - i*0.01, 'accuracy': 0.7 + i*0.02}
            logger.log_metrics(epoch, i, metrics, prefix='train')

            pbar.update(1)
        pbar.close()

    logger.log_summary({'final_loss': 0.4, 'final_accuracy': 0.9})
