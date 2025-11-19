#!/bin/bash
# Standalone VM - Single GPU/Node Runner
# Runs PyTorch workloads on a single GPU or CPU

set -e

# Default values
WORKLOAD="cnn_training"
CONFIG="config/config.yaml"
GPU_ID=0
NUM_GPUS=""
ALL_GPUS=""
DURATION=""

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
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --all-gpus)
      ALL_GPUS="true"
      shift
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --workload WORKLOAD    Workload to run (default: cnn_training)"
      echo "                         Options: cnn_training, transformer_training,"
      echo "                                  mixed_precision, gpu_burnin, ppo_training"
      echo "  --config CONFIG        Path to config file (default: config/config.yaml)"
      echo "  --gpu GPU_ID           GPU device ID (default: 0)"
      echo "  --num-gpus N           Number of GPUs (cnn_training, gpu_burnin)"
      echo "  --all-gpus             Use all available GPUs (cnn_training, gpu_burnin)"
      echo "  --duration MINS        Duration in minutes (gpu_burnin only)"
      echo "  --help                 Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --workload cnn_training --gpu 0"
      echo "  $0 --workload cnn_training --num-gpus 4"
      echo "  $0 --workload cnn_training --all-gpus"
      echo "  $0 --workload gpu_burnin --num-gpus 4 --duration 30"
      echo "  $0 --workload gpu_burnin --all-gpus"
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
if [ "$WORKLOAD" = "gpu_burnin" ] && { [ -n "$NUM_GPUS" ] || [ "$ALL_GPUS" = "true" ]; }; then
    if [ "$ALL_GPUS" = "true" ]; then
        echo "GPUs: All available"
    else
        echo "GPUs: $NUM_GPUS"
    fi
else
    echo "GPU ID: $GPU_ID"
fi
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

# Set GPU device (only for single GPU mode, not for multi-GPU workloads)
if { [ "$WORKLOAD" = "gpu_burnin" ] || [ "$WORKLOAD" = "cnn_training" ]; } && { [ -n "$NUM_GPUS" ] || [ "$ALL_GPUS" = "true" ]; }; then
    # Multi-GPU mode: don't restrict CUDA_VISIBLE_DEVICES
    echo "Multi-GPU mode: using all visible GPUs"
else
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

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

# Build command with extra arguments
EXTRA_ARGS=""
if [ "$WORKLOAD" = "gpu_burnin" ] || [ "$WORKLOAD" = "cnn_training" ]; then
  if [ -n "$NUM_GPUS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --num-gpus $NUM_GPUS"
  fi
  if [ "$ALL_GPUS" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --all-gpus"
  fi
fi
# Duration only applies to gpu_burnin
if [ "$WORKLOAD" = "gpu_burnin" ] && [ -n "$DURATION" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --duration $DURATION"
fi

echo "Running: python3 $SCRIPT --config $CONFIG $EXTRA_ARGS"
echo ""

python3 "$SCRIPT" --config "$CONFIG" $EXTRA_ARGS

echo ""
echo "========================================="
echo "Workload completed successfully!"
echo "========================================="
