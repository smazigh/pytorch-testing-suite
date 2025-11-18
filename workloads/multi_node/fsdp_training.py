#!/usr/bin/env python3
"""
Fully Sharded Data Parallel (FSDP) Training Workload - Multi Node
Trains large models using PyTorch FSDP for memory-efficient distributed training.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as models

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    load_config,
    get_logger,
    PerformanceBenchmark,
    GPUMonitor,
    SyntheticImageDataset
)


def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


class FSDPTrainer:
    """Trainer for FSDP training."""

    def __init__(self, config_path: str = None):
        """Initialize FSDP trainer."""
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.config = load_config(config_path)

        self.logger = get_logger(
            name="fsdp_training",
            log_dir=self.config.get('general.output_dir', './results'),
            level=self.config.get('general.log_level', 'INFO'),
            rank=self.rank
        )

        self.device = torch.device(f'cuda:{self.local_rank}')
        self.gpu_monitor = GPUMonitor(device_ids=[self.local_rank])

        if self.rank == 0:
            self.benchmark = PerformanceBenchmark(
                name="fsdp_training",
                output_dir=self.config.get('general.output_dir', './results')
            )

        self.model_name = self.config.get('workloads.fsdp.model', 'resnet50')
        self.batch_size = self.config.get('training.batch_size', 32)
        self.epochs = self.config.get('training.epochs', 5)
        self.lr = self.config.get('training.learning_rate', 0.001)

        if self.rank == 0:
            self.logger.log_header(f"FSDP Training - {self.model_name}")

    def create_model(self) -> FSDP:
        """Create and wrap model with FSDP."""
        if self.rank == 0:
            self.logger.info(f"Creating FSDP model: {self.model_name}")

        if self.model_name == 'resnet50':
            model = models.resnet50(num_classes=1000)
        elif self.model_name == 'resnet101':
            model = models.resnet101(num_classes=1000)
        else:
            model = models.resnet50(num_classes=1000)

        # FSDP wrapping
        auto_wrap_policy = size_based_auto_wrap_policy
        model = FSDP(
            model.to(self.device),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank
        )

        if self.rank == 0:
            self.logger.info("Model wrapped with FSDP")

        return model

    def create_dataloader(self):
        """Create distributed dataloader."""
        dataset = SyntheticImageDataset(
            num_samples=50000,
            image_size=224,
            num_classes=1000
        )

        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        return dataloader, sampler

    def train(self):
        """Run FSDP training."""
        if self.rank == 0:
            self.logger.log_header("Starting FSDP Training")

        model = self.create_model()
        dataloader, sampler = self.create_dataloader()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.epochs + 1):
            model.train()
            sampler.set_epoch(epoch)

            if self.rank == 0:
                pbar = self.logger.create_progress_bar(len(dataloader), f"Epoch {epoch}")

            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if self.rank == 0 and batch_idx % 10 == 0:
                    pbar.update(10)

            if self.rank == 0:
                pbar.close()
                self.logger.info(f"Epoch {epoch} completed")

        if self.rank == 0:
            self.logger.log_header("FSDP Training Complete")

        cleanup_distributed()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='FSDP Training Workload')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    args = parser.parse_args()

    trainer = FSDPTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
