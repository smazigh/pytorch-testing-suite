#!/bin/bash
#SBATCH --job-name=pytorch-multi-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pytorch-multi-%j.out
#SBATCH --error=logs/pytorch-multi-%j.err

# PyTorch Testing Framework - Multi-Node SLURM Script
# Usage: sbatch multi-node.sbatch [WORKLOAD] [CONFIG]
#
# Example:
#   sbatch --nodes=4 multi-node.sbatch ddp_training config/config.yaml

echo "=========================================="
echo "PyTorch Testing Framework - Multi-Node"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "=========================================="
echo ""

# Parse arguments
WORKLOAD="${1:-ddp_training}"
CONFIG="${2:-config/config.yaml}"

echo "Workload: $WORKLOAD"
echo "Config: $CONFIG"
echo ""

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/12.1
# module load cudnn/8.9
# module load nccl/2.18

# Activate virtual environment (if using)
# source venv/bin/activate

# Create logs directory
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# Get master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo ""

# Determine workload script
case $WORKLOAD in
  ddp_training)
    SCRIPT="workloads/multi_node/ddp_training.py"
    ;;
  fsdp_training)
    SCRIPT="workloads/multi_node/fsdp_training.py"
    ;;
  *)
    echo "ERROR: Unknown workload: $WORKLOAD"
    exit 1
    ;;
esac

# Print system info on master node
if [ "$SLURM_PROCID" -eq 0 ]; then
  echo "System Information:"
  echo "  Python: $(python3 --version)"
  echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
  echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
  echo "  NCCL: $(python3 -c 'import torch; print(torch.cuda.nccl.version())')"
  echo ""
fi

# Calculate global rank
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Starting process: RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE"

# Run workload using srun
srun python3 "$SCRIPT" --config "$CONFIG"

echo ""
echo "=========================================="
echo "Job completed!"
echo "=========================================="
