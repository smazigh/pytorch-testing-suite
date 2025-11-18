"""
Unit tests for logger module.
"""

import time
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.logger import (
    ColoredFormatter,
    PerformanceLogger,
    get_logger,
)


@pytest.mark.unit
class TestColoredFormatter:
    """Test ColoredFormatter class."""

    def test_formatter_creation(self):
        """Test creating ColoredFormatter."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        assert formatter is not None

    def test_format_info_message(self):
        """Test formatting INFO level message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert 'Test message' in formatted
        # Check that color codes are applied
        assert '\033[32m' in formatted  # Green for INFO

    def test_format_error_message(self):
        """Test formatting ERROR level message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='',
            lineno=0,
            msg='Error message',
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert 'Error message' in formatted
        assert '\033[31m' in formatted  # Red for ERROR

    def test_format_warning_message(self):
        """Test formatting WARNING level message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        record = logging.LogRecord(
            name='test',
            level=logging.WARNING,
            pathname='',
            lineno=0,
            msg='Warning message',
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert 'Warning message' in formatted
        assert '\033[33m' in formatted  # Yellow for WARNING

    def test_format_debug_message(self):
        """Test formatting DEBUG level message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        record = logging.LogRecord(
            name='test',
            level=logging.DEBUG,
            pathname='',
            lineno=0,
            msg='Debug message',
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert 'Debug message' in formatted
        assert '\033[36m' in formatted  # Cyan for DEBUG

    def test_format_critical_message(self):
        """Test formatting CRITICAL level message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        record = logging.LogRecord(
            name='test',
            level=logging.CRITICAL,
            pathname='',
            lineno=0,
            msg='Critical message',
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert 'Critical message' in formatted
        assert '\033[35m' in formatted  # Magenta for CRITICAL


@pytest.mark.unit
class TestPerformanceLogger:
    """Test PerformanceLogger class."""

    def test_logger_creation(self):
        """Test creating PerformanceLogger."""
        logger = PerformanceLogger(name='test_logger')

        assert logger.name == 'test_logger'
        assert logger.rank == 0
        assert logger.iteration_times == []
        assert logger.metrics_history == []

    def test_logger_with_log_dir(self, temp_dir):
        """Test logger with log directory."""
        logger = PerformanceLogger(
            name='test_logger',
            log_dir=str(temp_dir),
            rank=0
        )

        assert logger.metrics_file is not None
        assert temp_dir in logger.metrics_file.parents

    def test_logger_non_zero_rank(self, temp_dir):
        """Test logger with non-zero rank doesn't create file."""
        logger = PerformanceLogger(
            name='test_logger',
            log_dir=str(temp_dir),
            rank=1
        )

        assert logger.metrics_file is None

    def test_info_logging(self, caplog):
        """Test info logging."""
        logger = PerformanceLogger(name='test_info')
        logger.logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO, logger='test_info'):
            logger.info('Test info message')

        assert 'Test info message' in caplog.text

    def test_debug_logging(self, caplog):
        """Test debug logging."""
        logger = PerformanceLogger(name='test_debug', level='DEBUG')

        with caplog.at_level(logging.DEBUG, logger='test_debug'):
            logger.debug('Test debug message')

        assert 'Test debug message' in caplog.text

    def test_warning_logging(self, caplog):
        """Test warning logging."""
        logger = PerformanceLogger(name='test_warning')

        with caplog.at_level(logging.WARNING, logger='test_warning'):
            logger.warning('Test warning message')

        assert 'Test warning message' in caplog.text

    def test_error_logging(self, caplog):
        """Test error logging."""
        logger = PerformanceLogger(name='test_error')

        with caplog.at_level(logging.ERROR, logger='test_error'):
            logger.error('Test error message')

        assert 'Test error message' in caplog.text

    def test_critical_logging(self, caplog):
        """Test critical logging."""
        logger = PerformanceLogger(name='test_critical')

        with caplog.at_level(logging.CRITICAL, logger='test_critical'):
            logger.critical('Test critical message')

        assert 'Test critical message' in caplog.text

    def test_log_header(self, caplog):
        """Test log_header method."""
        logger = PerformanceLogger(name='test_header')

        with caplog.at_level(logging.INFO, logger='test_header'):
            logger.log_header('Test Header')

        assert 'Test Header' in caplog.text
        assert '=' in caplog.text

    def test_log_config(self, caplog):
        """Test log_config method."""
        logger = PerformanceLogger(name='test_config')

        config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'nested': {
                'param1': 'value1',
                'param2': 'value2'
            }
        }

        with caplog.at_level(logging.INFO, logger='test_config'):
            logger.log_config(config)

        assert 'batch_size' in caplog.text
        assert '32' in caplog.text
        assert 'nested' in caplog.text
        assert 'param1' in caplog.text

    def test_log_metrics(self, caplog):
        """Test log_metrics method."""
        logger = PerformanceLogger(name='test_metrics')

        metrics = {'loss': 0.5, 'accuracy': 0.95}

        with caplog.at_level(logging.INFO, logger='test_metrics'):
            logger.log_metrics(epoch=1, iteration=100, metrics=metrics, prefix='train')

        assert 'Epoch 1' in caplog.text
        assert 'Iter 100' in caplog.text
        assert 'loss' in caplog.text
        assert '0.5' in caplog.text
        assert 'train' in caplog.text

        # Check metrics history
        assert len(logger.metrics_history) == 1
        assert logger.metrics_history[0]['epoch'] == 1
        assert logger.metrics_history[0]['loss'] == 0.5

    def test_log_metrics_no_prefix(self, caplog):
        """Test log_metrics without prefix."""
        logger = PerformanceLogger(name='test_metrics_no_prefix')

        metrics = {'loss': 0.3}

        with caplog.at_level(logging.INFO, logger='test_metrics_no_prefix'):
            logger.log_metrics(epoch=2, iteration=50, metrics=metrics)

        assert 'Epoch 2' in caplog.text
        assert 'Iter 50' in caplog.text

    def test_log_progress(self, caplog):
        """Test log_progress method."""
        logger = PerformanceLogger(name='test_progress')

        with caplog.at_level(logging.INFO, logger='test_progress'):
            logger.log_progress(current=50, total=100, prefix='Training')

        assert '50/100' in caplog.text
        assert '50.0%' in caplog.text
        assert 'Training' in caplog.text

    def test_log_progress_with_metrics(self, caplog):
        """Test log_progress with metrics."""
        logger = PerformanceLogger(name='test_progress_metrics')

        metrics = {'loss': 0.4}

        with caplog.at_level(logging.INFO, logger='test_progress_metrics'):
            logger.log_progress(current=75, total=100, metrics=metrics)

        assert '75/100' in caplog.text
        assert 'loss' in caplog.text

    def test_iteration_timing(self):
        """Test iteration timing."""
        logger = PerformanceLogger(name='test_timing')

        logger.start_iteration()
        time.sleep(0.01)
        iter_time = logger.end_iteration()

        assert iter_time > 0
        assert len(logger.iteration_times) == 1
        assert logger.iteration_times[0] == iter_time

    def test_get_avg_iteration_time(self):
        """Test get_avg_iteration_time method."""
        logger = PerformanceLogger(name='test_avg_time')

        # Add known iteration times
        logger.iteration_times = [0.1, 0.2, 0.3]

        avg = logger.get_avg_iteration_time()
        assert avg == pytest.approx(0.2)

    def test_get_avg_iteration_time_empty(self):
        """Test get_avg_iteration_time with no iterations."""
        logger = PerformanceLogger(name='test_avg_empty')

        avg = logger.get_avg_iteration_time()
        assert avg == 0.0

    def test_get_avg_iteration_time_last_n(self):
        """Test get_avg_iteration_time with last_n parameter."""
        logger = PerformanceLogger(name='test_avg_last_n')

        # Add many iteration times
        logger.iteration_times = [0.1] * 50 + [0.2] * 50

        avg_last_10 = logger.get_avg_iteration_time(last_n=10)
        assert avg_last_10 == pytest.approx(0.2)

    def test_get_throughput(self):
        """Test get_throughput method."""
        logger = PerformanceLogger(name='test_throughput')

        # 0.1 seconds per iteration, batch size 32
        logger.iteration_times = [0.1, 0.1, 0.1]

        throughput = logger.get_throughput(batch_size=32)
        assert throughput == pytest.approx(320.0)  # 32 / 0.1 = 320

    def test_get_throughput_zero_time(self):
        """Test get_throughput with zero average time."""
        logger = PerformanceLogger(name='test_throughput_zero')

        throughput = logger.get_throughput(batch_size=32)
        assert throughput == 0.0

    def test_log_system_info(self, caplog):
        """Test log_system_info method."""
        logger = PerformanceLogger(name='test_system_info')

        system_info = {
            'platform': 'Linux',
            'python_version': '3.8.0',
            'torch_version': '2.0.0'
        }

        with caplog.at_level(logging.INFO, logger='test_system_info'):
            logger.log_system_info(system_info)

        assert 'platform' in caplog.text
        assert 'Linux' in caplog.text

    def test_log_gpu_info(self, caplog):
        """Test log_gpu_info method."""
        logger = PerformanceLogger(name='test_gpu_info')

        gpu_info = {
            0: {'name': 'Test GPU', 'memory': '16GB'},
            1: {'name': 'Test GPU 2', 'memory': '16GB'}
        }

        with caplog.at_level(logging.INFO, logger='test_gpu_info'):
            logger.log_gpu_info(gpu_info)

        assert 'GPU 0' in caplog.text
        assert 'Test GPU' in caplog.text

    def test_save_metrics(self, temp_dir):
        """Test save_metrics method."""
        logger = PerformanceLogger(
            name='test_save',
            log_dir=str(temp_dir),
            rank=0
        )

        # Add some metrics
        logger.log_metrics(epoch=1, iteration=1, metrics={'loss': 0.5})

        logger.save_metrics()

        # Check file was created
        assert logger.metrics_file.exists()

    def test_save_metrics_non_zero_rank(self, temp_dir):
        """Test save_metrics with non-zero rank."""
        logger = PerformanceLogger(
            name='test_save_rank1',
            log_dir=str(temp_dir),
            rank=1
        )

        # This should not crash even with no metrics_file
        logger.save_metrics()

    def test_log_summary(self, temp_dir, caplog):
        """Test log_summary method."""
        logger = PerformanceLogger(
            name='test_summary',
            log_dir=str(temp_dir),
            rank=0
        )

        # Add some iteration times
        logger.iteration_times = [0.1, 0.1, 0.1]

        summary = {
            'final_loss': 0.25,
            'final_accuracy': 0.95,
            'total_epochs': 10
        }

        with caplog.at_level(logging.INFO, logger='test_summary'):
            logger.log_summary(summary)

        assert 'Summary' in caplog.text
        assert 'final_loss' in caplog.text
        assert '0.25' in caplog.text

    def test_create_progress_bar(self):
        """Test create_progress_bar method."""
        logger = PerformanceLogger(name='test_pbar', rank=0)

        pbar = logger.create_progress_bar(total=100, desc='Test')
        assert pbar is not None
        assert pbar.total == 100
        pbar.close()

    def test_create_progress_bar_disabled(self):
        """Test create_progress_bar when disabled."""
        logger = PerformanceLogger(name='test_pbar_disabled', rank=1)

        pbar = logger.create_progress_bar(total=100, desc='Test')
        assert pbar is not None
        assert pbar.disable is True
        pbar.close()


@pytest.mark.unit
class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_default(self):
        """Test get_logger with defaults."""
        logger = get_logger()

        assert isinstance(logger, PerformanceLogger)
        assert logger.name == 'pytorch_test'
        assert logger.rank == 0

    def test_get_logger_custom_name(self):
        """Test get_logger with custom name."""
        logger = get_logger(name='custom_logger')

        assert logger.name == 'custom_logger'

    def test_get_logger_with_log_dir(self, temp_dir):
        """Test get_logger with log directory."""
        logger = get_logger(
            name='test_with_dir',
            log_dir=str(temp_dir)
        )

        assert logger.metrics_file is not None

    def test_get_logger_custom_level(self):
        """Test get_logger with custom level."""
        logger = get_logger(name='test_level', level='DEBUG')

        assert logger.logger.level == logging.DEBUG

    def test_get_logger_custom_rank(self):
        """Test get_logger with custom rank."""
        logger = get_logger(name='test_rank', rank=2)

        assert logger.rank == 2


@pytest.mark.unit
class TestLoggerEdgeCases:
    """Test edge cases for logger."""

    def test_log_progress_zero_total(self, caplog):
        """Test log_progress with zero total."""
        logger = PerformanceLogger(name='test_zero_total')

        with caplog.at_level(logging.INFO, logger='test_zero_total'):
            logger.log_progress(current=0, total=0)

        # Should not crash and show 0%
        assert '0.0%' in caplog.text

    def test_multiple_handlers(self, temp_dir):
        """Test that handlers are properly cleared on init."""
        # Create two loggers with same name
        logger1 = PerformanceLogger(name='duplicate', log_dir=str(temp_dir))
        logger2 = PerformanceLogger(name='duplicate', log_dir=str(temp_dir))

        # Second logger should have cleared handlers
        assert len(logger2.logger.handlers) <= 2  # Console + maybe file

    def test_log_metrics_various_types(self):
        """Test log_metrics with different value types in summary."""
        logger = PerformanceLogger(name='test_types')

        summary = {
            'float_val': 0.123456,
            'int_val': 42,
            'str_val': 'test'
        }

        # Should not raise any exception
        logger.log_summary(summary)
