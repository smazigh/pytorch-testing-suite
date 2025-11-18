#!/bin/bash
# PyTorch Testing Framework - Test Runner
# Quick script to run various test suites

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PyTorch Testing Framework - Test Runner"
echo "=========================================="
echo ""

# Default mode
MODE="${1:-all}"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Installing test dependencies..."
    pip install -r requirements/test.txt
fi

case $MODE in
    all)
        print_info "Running all tests..."
        pytest -v --cov=utils --cov=workloads --cov-report=term-missing
        ;;

    unit)
        print_info "Running unit tests..."
        pytest tests/unit/ -v -m unit --cov=utils --cov-report=term
        ;;

    integration)
        print_info "Running integration tests..."
        pytest tests/integration/ -v -m integration --timeout=600
        ;;

    smoke)
        print_info "Running smoke tests (quick validation)..."
        pytest -v -m smoke --timeout=300
        ;;

    fast)
        print_info "Running fast tests (excluding slow tests)..."
        pytest -v -m "not slow" --timeout=300
        ;;

    coverage)
        print_info "Running tests with full coverage report..."
        pytest -v --cov=utils --cov=workloads \
               --cov-report=term-missing \
               --cov-report=html \
               --cov-report=xml
        print_info "HTML coverage report: htmlcov/index.html"
        ;;

    imports)
        print_info "Testing all imports..."
        python -c "from utils import *" && print_info "✓ Utils import successful"
        python -c "from workloads.single_node.cnn_training import CNNTrainer" && print_info "✓ CNN workload import successful"
        python -c "from workloads.single_node.transformer_training import TransformerTrainer" && print_info "✓ Transformer workload import successful"
        python -c "from workloads.single_node.gpu_burnin import GPUBurnIn" && print_info "✓ GPU burn-in import successful"
        python -c "from workloads.multi_node.ddp_training import DDPTrainer" && print_info "✓ DDP workload import successful"
        python -c "from workloads.reinforcement_learning.ppo_training import PPOTrainer" && print_info "✓ PPO workload import successful"
        print_info "All imports successful!"
        ;;

    quick)
        print_info "Running quick validation (smoke tests + imports)..."
        pytest -v -m smoke --timeout=120
        python -c "from utils import *"
        print_info "Quick validation passed!"
        ;;

    ci)
        print_info "Running CI test suite..."
        print_info "Step 1/3: Unit tests"
        pytest tests/unit/ -v -m unit --cov=utils
        print_info "Step 2/3: Smoke tests"
        pytest -v -m smoke --timeout=300
        print_info "Step 3/3: Import tests"
        python -c "from utils import *"
        python -c "from workloads.single_node.cnn_training import CNNTrainer"
        print_info "CI test suite passed!"
        ;;

    parallel)
        print_info "Running tests in parallel..."
        pytest -v -n auto --cov=utils --cov=workloads
        ;;

    help)
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Available modes:"
        echo "  all         - Run all tests (default)"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  smoke       - Run smoke tests (quick validation)"
        echo "  fast        - Run fast tests (exclude slow tests)"
        echo "  coverage    - Run tests with full coverage report"
        echo "  imports     - Test that all modules can be imported"
        echo "  quick       - Quick validation (smoke + imports)"
        echo "  ci          - CI test suite"
        echo "  parallel    - Run tests in parallel"
        echo "  help        - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Run all tests"
        echo "  $0 unit         # Run unit tests"
        echo "  $0 quick        # Quick validation"
        echo "  $0 coverage     # Generate coverage report"
        exit 0
        ;;

    *)
        print_error "Unknown mode: $MODE"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="
