# SLURM Deployment

Deploy PyTorch testing workloads on HPC clusters managed by SLURM.

## Prerequisites

- SLURM 21.08+
- CUDA 11.8+
- NVIDIA GPU nodes
- Python 3.8+ (module or conda environment)
- Shared filesystem (NFS, Lustre, GPFS) across nodes

## Setup

### 1. Clone Repository

```bash
# On login node or shared filesystem
cd /shared/workspace
git clone <repository-url>
cd pytorch-test-suite
```

### 2. Setup Python Environment

Option A: Using module system
```bash
module load python/3.10
module load cuda/12.1
module load cudnn/8.9
module load nccl/2.18

pip install -r requirements/base.txt
pip install -r requirements/distributed.txt
```

Option B: Using Conda
```bash
module load cuda/12.1

conda create -n pytorch-test python=3.10
conda activate pytorch-test

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/base.txt
pip install -r requirements/distributed.txt
```

### 3. Verify Setup

```bash
# Test PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Check NCCL
python3 -c "import torch; print(torch.cuda.nccl.version())"

# Test GPU access (on compute node)
srun --gres=gpu:1 nvidia-smi
```

## Single-Node Jobs

### Basic Usage

```bash
# Submit single-node job
sbatch deployment/slurm/single-node.sbatch

# With specific workload
sbatch deployment/slurm/single-node.sbatch cnn_training

# With custom config
sbatch deployment/slurm/single-node.sbatch cnn_training config/my_config.yaml
```

### Batch Script Options

Edit `single-node.sbatch` to customize:

```bash
#SBATCH --job-name=pytorch-single-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --mem=32G                 # Memory
#SBATCH --time=04:00:00           # Wall time
#SBATCH --partition=gpu           # GPU partition
#SBATCH --account=your-account    # Accounting
```

### Available Workloads

| Workload | Command | GPUs | Duration |
|----------|---------|------|----------|
| CNN Training | `cnn_training` | 1 | ~20 min |
| Transformer | `transformer_training` | 1 | ~30 min |
| Mixed Precision | `mixed_precision` | 1 | ~15 min |
| GPU Burn-in | `gpu_burnin` | 1 | ~30 min |
| PPO Training | `ppo_training` | 1 | ~20 min |

### Examples

```bash
# Train ResNet50
sbatch deployment/slurm/single-node.sbatch cnn_training

# 1-hour burn-in test
sbatch deployment/slurm/single-node.sbatch gpu_burnin

# Transformer training
sbatch deployment/slurm/single-node.sbatch transformer_training
```

## Multi-Node Jobs

### Basic Usage

```bash
# Submit multi-node job
sbatch deployment/slurm/multi-node.sbatch

# Specify number of nodes
sbatch --nodes=4 deployment/slurm/multi-node.sbatch ddp_training

# With custom config
sbatch --nodes=8 deployment/slurm/multi-node.sbatch ddp_training config/my_config.yaml
```

### Batch Script Configuration

Edit `multi-node.sbatch`:

```bash
#SBATCH --job-name=pytorch-multi-node
#SBATCH --nodes=4                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Tasks per node (1 for single-GPU per node)
#SBATCH --cpus-per-task=8         # CPUs per task
#SBATCH --gres=gpu:1              # GPUs per node
#SBATCH --mem=32G
#SBATCH --time=08:00:00
```

### Multi-Node Workloads

| Workload | Command | Nodes | Description |
|----------|---------|-------|-------------|
| DDP Training | `ddp_training` | 2-16 | Distributed Data Parallel |
| FSDP Training | `fsdp_training` | 2-16 | Fully Sharded Data Parallel |

### Examples

```bash
# 2-node DDP training
sbatch --nodes=2 deployment/slurm/multi-node.sbatch ddp_training

# 8-node DDP training
sbatch --nodes=8 deployment/slurm/multi-node.sbatch ddp_training

# 4-node FSDP training
sbatch --nodes=4 deployment/slurm/multi-node.sbatch fsdp_training
```

## Job Management

### Submit Jobs

```bash
# Submit job
sbatch script.sbatch

# Submit with dependencies
sbatch --dependency=afterok:12345 script.sbatch

# Submit job array
sbatch --array=1-10 script.sbatch
```

### Monitor Jobs

```bash
# View queue
squeue -u $USER

# Detailed job info
scontrol show job JOBID

# Job efficiency
seff JOBID

# Cancel job
scancel JOBID
```

### View Outputs

```bash
# View output log
tail -f logs/pytorch-single-JOBID.out

# View error log
tail -f logs/pytorch-single-JOBID.err

# Copy results
cp -r results/ /permanent/storage/
```

## Advanced Configuration

### GPU Types

Request specific GPU types:

```bash
#SBATCH --gres=gpu:a100:2         # 2x A100 GPUs
#SBATCH --gres=gpu:h100:4         # 4x H100 GPUs
#SBATCH --constraint="gpu_mem:80gb"  # GPUs with 80GB memory
```

### Multiple GPUs per Node

For multi-GPU single-node:

```bash
#SBATCH --gres=gpu:4              # 4 GPUs on 1 node
#SBATCH --ntasks-per-node=4       # 4 tasks (1 per GPU)
```

Then use:
```bash
srun python3 workloads/multi_node/ddp_training.py
```

### Interactive Jobs

For testing and debugging:

```bash
# Request interactive session
srun --nodes=1 --gres=gpu:1 --time=01:00:00 --pty bash

# Inside interactive session
cd pytorch-test-suite
python3 workloads/single_node/cnn_training.py
```

## Environment Configuration

### Module System

Add to your batch script:

```bash
# Load required modules
module load python/3.10
module load cuda/12.1
module load cudnn/8.9
module load nccl/2.18

# Activate conda environment
source activate pytorch-test
```

### Environment Variables

Set in batch script:

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL settings (for multi-node)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5

# PyTorch settings
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
```

## Network Configuration

### InfiniBand

For optimal multi-node performance:

```bash
# Enable IB
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1

# Tune IB parameters
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
```

### Ethernet

If using Ethernet:

```bash
# Disable IB, use sockets
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_NTHREADS=4
```

## Performance Optimization

### File System

Use local SSD for temporary data:

```bash
#SBATCH --tmp=100GB

# In script
export TMPDIR=/tmp
export PYTORCH_CACHE_DIR=$TMPDIR/pytorch
```

### Process Pinning

For NUMA systems:

```bash
# CPU affinity
srun --cpu-bind=cores python3 script.py

# GPU affinity (automatic)
srun --gpu-bind=closest python3 script.py
```

### Profiling

Enable PyTorch profiler:

```yaml
# In config.yaml
monitoring:
  profile_performance: true
```

## Monitoring

### Job Metrics

```bash
# During job
sstat -j JOBID --format=JobID,MaxRSS,MaxVMSize,NTasks

# After job
sacct -j JOBID --format=JobID,JobName,Elapsed,State,MaxRSS,MaxVMSize
```

### GPU Monitoring

```bash
# On compute node
srun --jobid=JOBID nvidia-smi dmon -s pucvmet

# Or in batch script
nvidia-smi dmon -s pucvmet -o DT > gpu_metrics.log &
```

## Troubleshooting

### Job Fails Immediately

**Check**:
```bash
# View error log
cat logs/pytorch-*-JOBID.err

# Check job details
scontrol show job JOBID
```

**Common issues**:
- Module not found → Check module names
- CUDA not available → Verify GPU allocation
- Out of memory → Reduce batch size or request more memory

### Multi-Node Hangs

**Problem**: Distributed training hangs at initialization

**Debug**:
```bash
# Enable NCCL debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check network connectivity
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  bash -c 'hostname; ifconfig'
```

**Solutions**:
- Verify NCCL installation
- Check firewall/network settings
- Test with simple NCCL test:
  ```bash
  srun nccl-test/all_reduce_perf -b 8 -e 128M -f 2 -g 1
  ```

### Slow Performance

**Check**:
```bash
# CPU usage
srun --jobid=JOBID top -b -n 1

# GPU usage
srun --jobid=JOBID nvidia-smi

# I/O
srun --jobid=JOBID iotop -b -n 1
```

**Solutions**:
- Use local SSD for temporary files
- Increase DataLoader workers
- Check GPU utilization (should be >90%)
- Enable mixed precision

## Best Practices

1. **Test First**: Run short interactive jobs before submitting long batch jobs

2. **Use Job Arrays**: For parameter sweeps
   ```bash
   #SBATCH --array=1-10

   # Use SLURM_ARRAY_TASK_ID in script
   BATCH_SIZE=$((32 * SLURM_ARRAY_TASK_ID))
   ```

3. **Checkpointing**: Save checkpoints for long jobs
   ```yaml
   checkpoint:
     enabled: true
     save_interval: 10
   ```

4. **Resource Efficiency**: Don't request more than needed
   - Check actual usage with `seff JOBID`
   - Adjust future requests accordingly

5. **Logging**: Always save logs and results to permanent storage

## Job Templates

### Parameter Sweep

```bash
#!/bin/bash
#SBATCH --array=1-10
#SBATCH --job-name=param-sweep

# Vary batch size
BATCH_SIZE=$((32 * SLURM_ARRAY_TASK_ID))

# Update config
export PYTORCH_TEST_TRAINING_BATCH_SIZE=$BATCH_SIZE

python3 workloads/single_node/cnn_training.py
```

### Chain Jobs

```bash
# Job 1: Training
JOB1=$(sbatch --parsable train.sbatch)

# Job 2: Evaluation (after training)
sbatch --dependency=afterok:$JOB1 eval.sbatch
```

## Example Workflows

### Infrastructure Validation

```bash
# 1. Single GPU burn-in
sbatch deployment/slurm/single-node.sbatch gpu_burnin

# 2. Multi-GPU test
sbatch --nodes=2 deployment/slurm/multi-node.sbatch ddp_training

# 3. Full cluster burn-in
sbatch --nodes=8 deployment/slurm/multi-node.sbatch ddp_training
```

### Performance Benchmarking

```bash
# Scaling study
for nodes in 1 2 4 8; do
  sbatch --nodes=$nodes \
    --job-name=scale-$nodes \
    deployment/slurm/multi-node.sbatch ddp_training
done
```

## Next Steps

- Review [Main README](../../README.md) for overview
- Check [standalone deployment](../standalone/README.md) for VM testing
- Customize batch scripts for your cluster
- Contact your HPC admin for cluster-specific settings
