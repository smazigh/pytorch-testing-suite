# Standalone VM Deployment

Deploy PyTorch testing workloads on standalone VMs or physical machines.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with CUDA 11.8+ (for GPU workloads)
- NCCL 2.15+ (for multi-GPU workloads)

## Installation

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install framework dependencies
pip install -r requirements/base.txt

# For distributed training
pip install -r requirements/distributed.txt
```

## Single GPU Workloads

### Running Workloads

Use the `run_single.sh` script for single GPU workloads:

```bash
# Basic usage
./deployment/standalone/run_single.sh --workload WORKLOAD_NAME

# Specify GPU device
./deployment/standalone/run_single.sh --workload cnn_training --gpu 0

# Custom configuration
./deployment/standalone/run_single.sh --workload cnn_training --config my_config.yaml
```

### Available Workloads

| Workload | Command | Duration | Description |
|----------|---------|----------|-------------|
| CNN Training | `cnn_training` | ~10-30 min | Train ResNet/VGG on synthetic images |
| Transformer | `transformer_training` | ~20-40 min | Train transformer on synthetic text |
| Mixed Precision | `mixed_precision` | ~10-20 min | Compare FP32 vs AMP performance |
| GPU Burn-in | `gpu_burnin` | ~30 min | Stress test GPU at max utilization |
| PPO Training | `ppo_training` | ~15-30 min | Reinforcement learning workload |

### Examples

```bash
# Train ResNet50
./deployment/standalone/run_single.sh --workload cnn_training

# Run 1-hour burn-in test
./deployment/standalone/run_single.sh --workload gpu_burnin

# Train transformer
./deployment/standalone/run_single.sh --workload transformer_training
```

## Multi-GPU Workloads

### Prerequisites

- Multiple NVIDIA GPUs on the same node
- Working NCCL installation
- Network connectivity between GPUs (NVLink/PCIe)

### Running Multi-GPU Workloads

Use the `run_multi.sh` script for distributed training:

```bash
# Basic usage
./deployment/standalone/run_multi.sh --workload WORKLOAD_NAME --num-gpus N

# 4 GPU training
./deployment/standalone/run_multi.sh --workload ddp_training --num-gpus 4

# Custom configuration
./deployment/standalone/run_multi.sh --workload ddp_training --num-gpus 8 --config my_config.yaml
```

### Available Multi-GPU Workloads

| Workload | Command | GPUs | Description |
|----------|---------|------|-------------|
| DDP Training | `ddp_training` | 2-8 | Distributed Data Parallel training |
| FSDP Training | `fsdp_training` | 2-8 | Fully Sharded Data Parallel |

### Examples

```bash
# 2-GPU DDP training
./deployment/standalone/run_multi.sh --workload ddp_training --num-gpus 2

# 8-GPU FSDP training
./deployment/standalone/run_multi.sh --workload fsdp_training --num-gpus 8
```

## Configuration

Edit `config/config.yaml` to customize workload parameters:

```yaml
# GPU settings
gpu:
  device: cuda
  device_ids: [0, 1, 2, 3]  # Which GPUs to use
  mixed_precision: true

# Training settings
training:
  batch_size: 64
  epochs: 10
  learning_rate: 0.001

# Workload-specific
workloads:
  cnn:
    model: resnet50

  gpu_burnin:
    duration_minutes: 30
```

## Monitoring

### Real-time Monitoring

Monitor GPU usage during training:

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Output Files

Results are saved to `results/` directory:

- `<workload>_summary.json` - Performance summary
- `<workload>_metrics.csv` - Detailed per-iteration metrics
- `<workload>_TIMESTAMP.log` - Complete logs

## Troubleshooting

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 32  # Reduce from 64
   ```

2. Enable gradient checkpointing (for large models):
   ```yaml
   workloads:
     ddp:
       gradient_checkpointing: true
   ```

3. Use gradient accumulation:
   ```yaml
   training:
     gradient_accumulation_steps: 2
   ```

### Slow Performance

**Problem**: Low throughput or GPU utilization

**Solutions**:
1. Increase DataLoader workers:
   ```yaml
   training:
     num_workers: 8  # Increase from 4
   ```

2. Enable pin_memory:
   ```yaml
   training:
     pin_memory: true
   ```

3. Use mixed precision:
   ```yaml
   gpu:
     mixed_precision: true
   ```

### Multi-GPU Issues

**Problem**: Distributed training hangs or fails

**Solutions**:
1. Check NCCL installation:
   ```bash
   python -c "import torch; print(torch.cuda.nccl.version())"
   ```

2. Enable NCCL debug:
   ```bash
   export NCCL_DEBUG=INFO
   ./deployment/standalone/run_multi.sh --workload ddp_training --num-gpus 2
   ```

3. Verify GPU connectivity:
   ```bash
   nvidia-smi topo -m
   ```

## Performance Tips

1. **Use Mixed Precision**: Enable AMP for 2-3x speedup on modern GPUs
   ```yaml
   gpu:
     mixed_precision: true
   ```

2. **Optimize Batch Size**: Use largest batch size that fits in memory
   - Start with batch_size=32, increase until OOM
   - Use batch_size that's a multiple of 8 for optimal GPU utilization

3. **Enable cuDNN Autotuner**: Let cuDNN find fastest algorithms
   ```yaml
   gpu:
     cudnn_benchmark: true
   ```

4. **Use Channels Last**: For CNN workloads on Ampere+ GPUs
   ```yaml
   performance:
     channels_last: true
   ```

## Running Custom Workloads

To run a custom Python script:

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Run script
python3 my_custom_script.py --config config/config.yaml
```

For multi-GPU custom scripts:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  my_custom_script.py --config config/config.yaml
```

## System Requirements

### Minimum

- 1x NVIDIA GPU (8GB+ VRAM)
- 16GB system RAM
- 4 CPU cores
- 50GB free disk space

### Recommended

- 4x NVIDIA A100/H100 GPUs
- 128GB system RAM
- 32 CPU cores
- 500GB NVMe SSD

### Tested Configurations

| GPU | VRAM | Batch Size | Workload | Performance |
|-----|------|------------|----------|-------------|
| RTX 3090 | 24GB | 64 | ResNet50 | 350 img/s |
| A100 | 40GB | 128 | ResNet50 | 650 img/s |
| A100 | 80GB | 256 | ResNet50 | 700 img/s |
| H100 | 80GB | 256 | ResNet50 | 1200 img/s |

## Next Steps

- Review [Main README](../../README.md) for overview
- Check [configuration reference](../../config/config.yaml)
- Try different workloads to validate your setup
- Customize configurations for your use case
