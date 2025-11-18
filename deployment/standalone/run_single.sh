#!/bin/bash
# Standalone VM - Single GPU/Node Runner
# Runs PyTorch workloads on a single GPU or CPU

set -e

# Default values
WORKLOAD="cnn_training"
CONFIG="config/config.yaml"
GPU_ID=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --workload)
      WORKLOAD="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --workload WORKLOAD    Workload to run (default: cnn_training)"
      echo "                         Options: cnn_training, transformer_training,"
      echo "                                  mixed_precision, gpu_burnin"
      echo "  --config CONFIG        Path to config file (default: config/config.yaml)"
      echo "  --gpu GPU_ID          GPU device ID (default: 0)"
      echo "  --help                Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --workload cnn_training --gpu 0"
      echo "  $0 --workload gpu_burnin --config my_config.yaml"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/../.."

echo "========================================="
echo "PyTorch Testing Framework - Single Node"
echo "========================================="
echo "Workload: $WORKLOAD"
echo "Config: $CONFIG"
echo "GPU ID: $GPU_ID"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if config exists
if [ ! -f "$REPO_ROOT/$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Set GPU device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Determine workload path
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
    echo "Available workloads: cnn_training, transformer_training, mixed_precision, gpu_burnin, ppo_training"
    exit 1
    ;;
esac

# Run workload
cd "$REPO_ROOT"
echo "Running: python3 $SCRIPT --config $CONFIG"
echo ""

python3 "$SCRIPT" --config "$CONFIG"

echo ""
echo "========================================="
echo "Workload completed successfully!"
echo "========================================="
