#!/bin/bash
#SBATCH --job-name=pytorch-single-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pytorch-single-%j.out
#SBATCH --error=logs/pytorch-single-%j.err

# PyTorch Testing Framework - Single Node SLURM Script
# Usage: sbatch single-node.sbatch [WORKLOAD] [CONFIG]
#
# Example:
#   sbatch single-node.sbatch cnn_training config/config.yaml

echo "=========================================="
echo "PyTorch Testing Framework - Single Node"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="
echo ""

# Parse arguments
WORKLOAD="${1:-cnn_training}"
CONFIG="${2:-config/config.yaml}"

echo "Workload: $WORKLOAD"
echo "Config: $CONFIG"
echo ""

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/12.1
# module load cudnn/8.9

# Activate virtual environment (if using)
# source venv/bin/activate

# Create logs directory
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Determine workload script
case $WORKLOAD in
  cnn_training)
    SCRIPT="workloads/single_node/cnn_training.py"
    ;;
  transformer_training)
    SCRIPT="workloads/single_node/transformer_training.py"
    ;;
  mixed_precision)
    SCRIPT="workloads/single_node/mixed_precision.py"
    ;;
  gpu_burnin)
    SCRIPT="workloads/single_node/gpu_burnin.py"
    ;;
  ppo_training)
    SCRIPT="workloads/reinforcement_learning/ppo_training.py"
    ;;
  *)
    echo "ERROR: Unknown workload: $WORKLOAD"
    exit 1
    ;;
esac

# Print system info
echo "System Information:"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "  GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Run workload
echo "Starting workload..."
python3 "$SCRIPT" --config "$CONFIG"

echo ""
echo "=========================================="
echo "Job completed!"
echo "=========================================="
