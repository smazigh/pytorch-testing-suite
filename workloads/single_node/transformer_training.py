#!/usr/bin/env python3
"""
Transformer Training Workload - Single Node
Trains transformer models on synthetic sequence data for testing and burn-in.
"""

import os
import sys
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    load_config,
    get_logger,
    PerformanceBenchmark,
    GPUMonitor,
    create_synthetic_dataloader
)


class TransformerModel(nn.Module):
    """Simple transformer model for language modeling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        """
        Initialize transformer model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Forward pass.

        Args:
            src: Input tensor [batch_size, seq_length]

        Returns:
            Output tensor [batch_size, seq_length, vocab_size]
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src)
        output = self.fc_out(output)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output with positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTrainer:
    """Trainer for transformer models."""

    def __init__(self, config_path: str = None):
        """
        Initialize transformer trainer.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self.logger = get_logger(
            name="transformer_training",
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
            name="transformer_training",
            output_dir=self.config.get('general.output_dir', './results')
        )

        # Model configuration
        self.vocab_size = self.config.get('workloads.transformer.vocab_size', 30000)
        self.max_seq_length = self.config.get('workloads.transformer.max_seq_length', 512)
        self.d_model = self.config.get('workloads.transformer.model_dim', 768)
        self.num_layers = self.config.get('workloads.transformer.num_layers', 12)
        self.num_heads = self.config.get('workloads.transformer.num_heads', 12)
        self.dim_feedforward = self.config.get('workloads.transformer.feedforward_dim', 3072)
        self.dropout = self.config.get('workloads.transformer.dropout', 0.1)

        # Training configuration
        self.batch_size = self.config.get('training.batch_size', 32)
        self.epochs = self.config.get('training.epochs', 10)
        self.lr = self.config.get('training.learning_rate', 0.0001)

        # Mixed precision
        self.use_amp = self.config.get('gpu.mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        self.logger.log_header("Transformer Training")
        self._log_configuration()

    def _log_configuration(self):
        """Log configuration details."""
        config_dict = {
            'Device': str(self.device),
            'Batch Size': self.batch_size,
            'Epochs': self.epochs,
            'Learning Rate': self.lr,
            'Vocab Size': self.vocab_size,
            'Max Seq Length': self.max_seq_length,
            'Model Dim': self.d_model,
            'Num Layers': self.num_layers,
            'Num Heads': self.num_heads,
            'FFN Dim': self.dim_feedforward,
            'Mixed Precision': self.use_amp,
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
        Create transformer model.

        Returns:
            PyTorch model
        """
        self.logger.info("Creating transformer model...")

        model = TransformerModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_length=self.max_seq_length
        )

        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        return model

    def create_dataloader(self):
        """Create data loader with synthetic data."""
        self.logger.info("Creating synthetic sequence dataloader...")

        dataloader = create_synthetic_dataloader(
            dataset_type="sequence",
            batch_size=self.batch_size,
            num_samples=self.config.get('workloads.transformer.dataset_size', 100000),
            num_workers=self.config.get('training.num_workers', 4),
            pin_memory=self.config.get('training.pin_memory', True),
            seq_length=self.max_seq_length,
            vocab_size=self.vocab_size,
            seed=self.config.get('general.seed', 42)
        )

        self.logger.info(f"DataLoader created with {len(dataloader)} batches")
        return dataloader

    def train_epoch(self, model, dataloader, optimizer, epoch):
        """
        Train for one epoch.

        Args:
            model: Model to train
            dataloader: Training dataloader
            optimizer: Optimizer
            epoch: Current epoch number
        """
        model.train()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_tokens = 0

        # Progress bar
        pbar = self.logger.create_progress_bar(
            total=len(dataloader),
            desc=f"Epoch {epoch}/{self.epochs}"
        )

        self.benchmark.start_epoch()

        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            self.benchmark.start_iteration()

            input_seq = input_seq.to(self.device, non_blocking=True)
            target_seq = target_seq.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = model(input_seq)
                    # Reshape for cross entropy loss
                    loss = criterion(
                        outputs.reshape(-1, self.vocab_size),
                        target_seq.reshape(-1)
                    )

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(input_seq)
                loss = criterion(
                    outputs.reshape(-1, self.vocab_size),
                    target_seq.reshape(-1)
                )
                loss.backward()
                optimizer.step()

            # Update metrics
            batch_tokens = input_seq.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            iter_time = self.benchmark.end_iteration()
            self.benchmark.record_loss(loss.item())

            # Record GPU metrics
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                gpu_metrics = self.gpu_monitor.get_metrics(0)
                if gpu_metrics:
                    self.benchmark.record_gpu_metrics(
                        utilization=gpu_metrics.utilization,
                        memory_used=gpu_metrics.memory_used,
                        memory_allocated=gpu_metrics.memory_allocated,
                        temperature=gpu_metrics.temperature
                    )

            # Log progress
            if batch_idx % self.config.get('logging.console_log_interval', 10) == 0:
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability
                throughput = self.benchmark.get_throughput(self.batch_size)
                tokens_per_sec = throughput * self.max_seq_length if throughput > 0 else 0

                metrics = {
                    'loss': avg_loss,
                    'ppl': perplexity,
                    'tok/s': tokens_per_sec,
                }

                # Add GPU metrics
                if torch.cuda.is_available():
                    gpu_summary = self.gpu_monitor.log_metrics_summary()
                    pbar.set_postfix_str(f"{metrics} | {gpu_summary}")
                else:
                    pbar.set_postfix(metrics)

            pbar.update(1)

        pbar.close()

        epoch_time = self.benchmark.end_epoch()
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))

        self.logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s | "
            f"Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}"
        )

        return avg_loss, perplexity

    def train(self):
        """Run training."""
        self.logger.log_header("Starting Training")

        # Configure benchmark
        self.benchmark.configure(
            batch_size=self.batch_size,
            sequence_length=self.max_seq_length,
            model_flops=0
        )

        # Create model
        model = self.create_model()

        # Create dataloader
        dataloader = self.create_dataloader()

        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            betas=tuple(self.config.get('optimizer.betas', [0.9, 0.999])),
            weight_decay=self.config.get('training.weight_decay', 0.01)
        )

        # Training loop
        for epoch in range(1, self.epochs + 1):
            loss, perplexity = self.train_epoch(model, dataloader, optimizer, epoch)

        # Print summary
        self.logger.log_header("Training Complete")
        self.benchmark.print_summary()
        self.benchmark.save_results()

        # Final GPU metrics
        if torch.cuda.is_available():
            self.gpu_monitor.print_all_metrics()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Transformer Training Workload')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Run training
    trainer = TransformerTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
