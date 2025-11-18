#!/bin/bash
# Standalone VM - Multi GPU/Node Runner
# Runs distributed PyTorch workloads using torchrun

set -e

# Default values
WORKLOAD="ddp_training"
CONFIG="config/config.yaml"
NUM_GPUS=2

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
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --workload WORKLOAD    Workload to run (default: ddp_training)"
      echo "                         Options: ddp_training, fsdp_training"
      echo "  --config CONFIG        Path to config file (default: config/config.yaml)"
      echo "  --num-gpus N          Number of GPUs to use (default: 2)"
      echo "  --help                Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --workload ddp_training --num-gpus 4"
      echo "  $0 --workload fsdp_training --config my_config.yaml"
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

echo "=========================================="
echo "PyTorch Testing Framework - Multi GPU"
echo "=========================================="
echo "Workload: $WORKLOAD"
echo "Config: $CONFIG"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="
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

# Determine workload path
case $WORKLOAD in
  ddp_training)
    SCRIPT="workloads/multi_node/ddp_training.py"
    ;;
  fsdp_training)
    SCRIPT="workloads/multi_node/fsdp_training.py"
    ;;
  *)
    echo "ERROR: Unknown workload: $WORKLOAD"
    echo "Available workloads: ddp_training, fsdp_training"
    exit 1
    ;;
esac

# Run workload with torchrun
cd "$REPO_ROOT"
echo "Running: torchrun --nproc_per_node=$NUM_GPUS $SCRIPT --config $CONFIG"
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    "$SCRIPT" \
    --config "$CONFIG"

echo ""
echo "=========================================="
echo "Workload completed successfully!"
echo "=========================================="
