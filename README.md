# PyTorch Testing Framework

A comprehensive testing and benchmarking framework for PyTorch workloads, designed for Solution Engineers and Architects to validate GPU infrastructure across standalone VMs, Kubernetes clusters, and SLURM environments.

## Features

- **Zero External Dependencies**: All workloads use synthetic data - no datasets required
- **Multi-Platform Support**: Run on standalone VMs, Kubernetes, or SLURM
- **Comprehensive Workloads**: CNN, Transformers, RL, GPU burn-in, and distributed training
- **Production-Ready Logging**: Verbose progress tracking and performance benchmarking
- **Configurable**: All parameters in easy-to-edit YAML configuration
- **GPU Monitoring**: Real-time GPU utilization, memory, and temperature tracking

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pytorch-test-suite

# Install dependencies
pip install -r requirements/base.txt

# For distributed training
pip install -r requirements/distributed.txt

# For reinforcement learning
pip install -r requirements/reinforcement-learning.txt
```

**Note**: PyTorch 2.0+ includes built-in AMP and distributed support. See [INSTALL.md](INSTALL.md) for detailed installation instructions, GPU setup, and optional dependencies.

### Run Your First Workload

```bash
# Single GPU CNN training
./deployment/standalone/run_single.sh --workload cnn_training

# GPU burn-in test (30 minutes by default)
./deployment/standalone/run_single.sh --workload gpu_burnin

# Multi-GPU distributed training
./deployment/standalone/run_multi.sh --workload ddp_training --num-gpus 4
```

## Architecture

```
pytorch-test-suite/
├── config/                    # Configuration files
│   └── config.yaml           # Main configuration
├── workloads/                # Training workloads
│   ├── single_node/          # Single GPU/node workloads
│   │   ├── cnn_training.py
│   │   ├── transformer_training.py
│   │   ├── mixed_precision.py
│   │   └── gpu_burnin.py
│   ├── multi_node/           # Multi GPU/node workloads
│   │   ├── ddp_training.py
│   │   └── fsdp_training.py
│   └── reinforcement_learning/
│       └── ppo_training.py
├── utils/                    # Core utilities
│   ├── config_loader.py
│   ├── logger.py
│   ├── benchmark.py
│   ├── gpu_monitor.py
│   └── data_generators.py
├── deployment/              # Deployment configurations
│   ├── standalone/          # VM scripts
│   ├── kubernetes/          # K8s manifests
│   └── slurm/              # SLURM batch scripts
└── docs/                   # Documentation
```

## Available Workloads

### Single-Node Workloads

| Workload | Description | Use Case |
|----------|-------------|----------|
| `cnn_training` | Train ResNet/VGG models | Image classification benchmarking |
| `transformer_training` | Train transformer models | NLP/sequence modeling benchmarking |
| `mixed_precision` | Compare FP32 vs AMP | Mixed precision validation |
| `gpu_burnin` | Max GPU utilization | Stress testing and validation |
| `ppo_training` | Reinforcement learning | RL workload testing |

### Multi-Node Workloads

| Workload | Description | Use Case |
|----------|-------------|----------|
| `ddp_training` | Distributed Data Parallel | Multi-GPU data parallelism |
| `fsdp_training` | Fully Sharded Data Parallel | Large model training |

## Deployment Options

### 1. Standalone VM

Perfect for single machines or VMs with one or more GPUs.

**Single GPU:**
```bash
./deployment/standalone/run_single.sh --workload cnn_training --gpu 0
```

**Multi GPU (same node):**
```bash
./deployment/standalone/run_multi.sh --workload ddp_training --num-gpus 4
```

See [deployment/standalone/README.md](deployment/standalone/README.md) for details.

### 2. Kubernetes

Designed for containerized workloads in K8s clusters.

**Single Node:**
```bash
kubectl apply -f deployment/kubernetes/single-node-job.yaml
```

**Multi Node (requires PyTorch operator):**
```bash
kubectl apply -f deployment/kubernetes/multi-node-job.yaml
```

See [deployment/kubernetes/README.md](deployment/kubernetes/README.md) for setup.

### 3. SLURM

For HPC clusters managed by SLURM.

**Single Node:**
```bash
sbatch deployment/slurm/single-node.sbatch cnn_training
```

**Multi Node:**
```bash
sbatch --nodes=4 deployment/slurm/multi-node.sbatch ddp_training
```

See [deployment/slurm/README.md](deployment/slurm/README.md) for details.

## Configuration

All workload parameters are configurable via `config/config.yaml`:

```yaml
# GPU settings
gpu:
  device: cuda
  mixed_precision: true

# Training settings
training:
  batch_size: 64
  epochs: 10
  learning_rate: 0.001

# Workload-specific settings
workloads:
  cnn:
    model: resnet50
    image_size: 224

  gpu_burnin:
    duration_minutes: 30
    stress_level: 100
```

You can override settings via environment variables:
```bash
export PYTORCH_TEST_TRAINING_BATCH_SIZE=128
export PYTORCH_TEST_GPU_MIXED_PRECISION=false
```

See `config/config.yaml` for full configuration reference.

## Output and Benchmarking

Each workload generates:

1. **Console Output**: Real-time progress with metrics
   - Loss, accuracy, throughput
   - GPU utilization, memory, temperature
   - Estimated time remaining

2. **JSON Summary**: `results/<workload>_summary.json`
   - Average/peak throughput
   - GPU statistics
   - Training metrics

3. **CSV Metrics**: `results/<workload>_metrics.csv`
   - Per-iteration timing and metrics
   - For detailed analysis

Example output:
```
================================================================================
  CNN Training - resnet50
================================================================================

Configuration:
  Model: resnet50
  Device: cuda
  Batch Size: 64
  Epochs: 10
  Learning Rate: 0.001
  Mixed Precision: True

[Epoch 1/10] ████████████████████ 100% | 781/781 [02:15<00:00, 5.76it/s]
  Loss: 0.4523 | Accuracy: 85.32% | Throughput: 368 samples/s
  GPU0: 95% | 10.2/24.0GB | 72°C

Summary:
  Total time: 1352.3s (22.5 minutes)
  Avg iteration time: 173ms
  Avg throughput: 370 samples/sec
  Final accuracy: 89.7%
  Peak GPU memory: 10.8 GB
```

## Common Use Cases

### 1. Infrastructure Validation
Test new GPU hardware or cluster setup:
```bash
# Single GPU burn-in
./deployment/standalone/run_single.sh --workload gpu_burnin

# Multi-GPU burn-in (all GPUs)
./deployment/standalone/run_single.sh --workload gpu_burnin --all-gpus

# Specific number of GPUs
./deployment/standalone/run_single.sh --workload gpu_burnin --num-gpus 4 --duration 30

# Verify multi-GPU communication
./deployment/standalone/run_multi.sh --workload ddp_training --num-gpus 8
```

### 2. Performance Benchmarking
Compare different configurations:
```bash
# Test mixed precision speedup
./deployment/standalone/run_single.sh --workload mixed_precision

# Benchmark distributed scaling
for n in 1 2 4 8; do
  ./deployment/standalone/run_multi.sh --workload ddp_training --num-gpus $n
done
```

### 3. Customer Demonstrations
Show PyTorch capabilities:
```bash
# Quick CNN demo (single GPU)
./deployment/standalone/run_single.sh --workload cnn_training

# Multi-GPU CNN training
./deployment/standalone/run_single.sh --workload cnn_training --num-gpus 4
./deployment/standalone/run_single.sh --workload cnn_training --all-gpus

# Transformer training
./deployment/standalone/run_single.sh --workload transformer_training
```

### 4. Reinforcement Learning with PPO
Test RL workloads:
```bash
# Run PPO training
python workloads/reinforcement_learning/ppo_training.py

# With custom config
python workloads/reinforcement_learning/ppo_training.py --config config/config.yaml
```

## CNN Training Details

The CNN training workload supports multiple architectures and multi-GPU training using DataParallel.

### Multi-GPU Support

Use DataParallel to distribute training across multiple GPUs:

```bash
# Single GPU (default)
./deployment/standalone/run_single.sh --workload cnn_training

# Specific number of GPUs
./deployment/standalone/run_single.sh --workload cnn_training --num-gpus 4

# All available GPUs
./deployment/standalone/run_single.sh --workload cnn_training --all-gpus
```

### Supported Models

- `resnet18`, `resnet50`, `resnet101`
- `vgg16`
- `efficientnet_b0`

### Configuration

```yaml
workloads:
  cnn:
    model: resnet50
    image_size: 224
    num_classes: 1000
    dataset_size: 50000
```

### Output Example

```
Configuration:
  Model: resnet50
  Device: cuda:0
  Num GPUs: 4
  GPU IDs: 0, 1, 2, 3
  Batch Size: 64 (effective: 256)
  ...

Using DataParallel on 4 GPUs: [0, 1, 2, 3]
```

## PPO Training Details

The PPO (Proximal Policy Optimization) workload implements a complete reinforcement learning training pipeline using a synthetic environment that requires no external dependencies.

### Architecture

**Actor-Critic Network:**
- Shared feature extraction: Two fully-connected layers (256 units, ReLU)
- Actor head: Gaussian (continuous) or Categorical (discrete) policy
- Critic head: State value estimation

**Training Pipeline:**
1. Collect trajectories (2048 steps by default)
2. Compute returns using Generalized Advantage Estimation (GAE)
3. Run multiple PPO epochs with clipped surrogate objective
4. Log metrics and repeat

### Configuration

Configure PPO in `config/config.yaml`:

```yaml
training:
  epochs: 100           # Total training episodes
  learning_rate: 0.0003 # Learning rate for Adam optimizer

workloads:
  reinforcement_learning:
    algorithm: ppo
    environment: synthetic
    env_type: continuous  # continuous or discrete

    ppo:
      clip_epsilon: 0.2      # PPO clipping parameter
      value_coef: 0.5        # Critic loss coefficient
      entropy_coef: 0.01     # Entropy bonus coefficient
      gae_lambda: 0.95       # GAE lambda for advantage estimation
      num_steps: 2048        # Environment steps per episode
      num_epochs: 10         # PPO optimization epochs per episode
      mini_batch_size: 64    # Mini-batch size for updates
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_epsilon` | 0.2 | Clipping range for policy ratio |
| `gae_lambda` | 0.95 | GAE lambda for bias-variance tradeoff |
| `num_steps` | 2048 | Trajectory length per episode |
| `num_epochs` | 10 | Optimization passes per trajectory |
| `mini_batch_size` | 64 | Samples per gradient update |

### Environment

The PPO trainer uses a `SyntheticReinforcementEnvironment` with:
- **State dimension**: 4
- **Action dimension**: 2
- **Episode length**: 200 steps
- **Action space**: Continuous (default) or discrete

### Output

PPO training generates:
- Real-time logging every 10 episodes (reward, loss)
- Performance benchmarks (throughput, timing)
- Results saved to `results/ppo_training_summary.json`

Example output:
```
================================================================================
  PPO Training
================================================================================

Configuration:
  Device: cuda
  Total Episodes: 100
  Steps per Episode: 2048
  PPO Epochs: 10
  Mini Batch Size: 64
  Clip Epsilon: 0.2
  Learning Rate: 0.0003

Episode 10/100 | Avg Reward: 0.45 | Loss: 0.1234
Episode 20/100 | Avg Reward: 0.68 | Loss: 0.0987
...
```

### Running PPO Training

**Direct execution:**
```bash
python workloads/reinforcement_learning/ppo_training.py
```

**With custom configuration:**
```bash
python workloads/reinforcement_learning/ppo_training.py --config path/to/config.yaml
```

**Override settings via environment variables:**
```bash
export PYTORCH_TEST_TRAINING_EPOCHS=200
export PYTORCH_TEST_TRAINING_LEARNING_RATE=0.0001
python workloads/reinforcement_learning/ppo_training.py
```

## GPU Burn-in Details

The GPU burn-in workload maximizes GPU utilization for stress testing and validation, with support for multi-GPU parallel stress testing.

### Features

- **Multi-GPU Support**: Stress test multiple GPUs simultaneously
- **Configurable Operations**: matmul, conv2d, attention
- **Real-time Monitoring**: In-place status updates showing all GPU metrics
- **Graceful Shutdown**: Ctrl+C stops all workers cleanly

### Running GPU Burn-in

**Single GPU:**
```bash
./deployment/standalone/run_single.sh --workload gpu_burnin
```

**All available GPUs:**
```bash
./deployment/standalone/run_single.sh --workload gpu_burnin --all-gpus
```

**Specific number of GPUs with custom duration:**
```bash
./deployment/standalone/run_single.sh --workload gpu_burnin --num-gpus 4 --duration 60
```

**Direct execution:**
```bash
python workloads/single_node/gpu_burnin.py --num-gpus 4 --duration 30
python workloads/single_node/gpu_burnin.py --all-gpus
```

### Configuration

Configure burn-in in `config/config.yaml`:

```yaml
burnin:
  duration_minutes: 30
  target_utilization: 95
  memory_fraction: 0.9

workloads:
  gpu_burnin:
    matrix_size: 8192
    operations: [matmul, conv2d, attention]
    stress_level: 100
```

### Console Output

The multi-GPU mode displays a concise, in-place status line:

```
[5.2m/30.0m] G0:conv2d:1250it:8.2GB:72C | G1:conv2d:1248it:8.1GB:71C | G2:conv2d:1251it:8.3GB:73C
```

Format: `[elapsed/total] GPU_ID:operation:iterations:memory:temperature`

### Final Summary

```
================================================================================
  Multi-GPU Burn-in Complete
================================================================================

Total GPUs: 4
Total iterations: 25000
Elapsed time: 30.0 minutes
Avg iterations/GPU: 6250

Per-GPU Summary:
  GPU 0: 6248 iterations
  GPU 1: 6251 iterations
  GPU 2: 6250 iterations
  GPU 3: 6251 iterations
```

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `batch_size` in `config/config.yaml`
- Enable `gradient_checkpointing` for large models

**Slow DataLoader:**
- Increase `num_workers` in config
- Enable `pin_memory` for faster GPU transfer

**Distributed Training Issues:**
- Check NCCL environment variables
- Verify network connectivity between nodes
- Enable `NCCL_DEBUG=INFO` for diagnostics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU workloads)
- 8GB+ GPU memory recommended

Optional:
- NCCL 2.15+ (for multi-GPU)
- Kubernetes 1.24+ (for K8s deployment)
- SLURM 21.08+ (for HPC deployment)

## Testing

The framework includes comprehensive tests and CI/CD pipelines.

### Running Tests

```bash
# Install test dependencies
pip install -r requirements/test.txt

# Run all tests
pytest

# Run quick validation
./run_tests.sh quick

# Run unit tests only
./run_tests.sh unit

# Run with coverage
./run_tests.sh coverage
```

### Test Categories

- **Unit Tests**: Test utilities (config, logger, benchmark, data generators)
- **Integration Tests**: Test workloads end-to-end
- **Smoke Tests**: Quick validation tests
- **CI/CD**: Automated testing via GitHub Actions

See [tests/README.md](tests/README.md) for detailed testing documentation.

### CI/CD Pipeline

Automated tests run on every push and pull request:
- Code quality checks (Black, flake8, isort)
- Unit tests on Python 3.8, 3.9, 3.10, 3.11
- Integration tests
- Import validation
- Deployment script validation

## Documentation

Platform-specific deployment guides:
- [Standalone VM](deployment/standalone/README.md)
- [Kubernetes](deployment/kubernetes/README.md)
- [SLURM](deployment/slurm/README.md)
- [Testing Guide](tests/README.md)

## License

This project is provided as-is for testing and benchmarking purposes.

## Contributing

This framework is designed to be extensible. To add new workloads:

1. Create script in appropriate `workloads/` directory
2. Follow existing workload structure
3. Use utilities from `utils/` package
4. Add configuration section to `config/config.yaml`
5. Update documentation
