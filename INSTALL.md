# Installation Guide

## Quick Install

### Minimal Installation (CPU)

```bash
pip install -r requirements/base.txt
```

This installs core dependencies for running workloads on CPU.

### GPU Support

For GPU support, install PyTorch with CUDA:

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements/base.txt
```

### Distributed Training

```bash
pip install -r requirements/distributed.txt
```

### Reinforcement Learning

```bash
pip install -r requirements/reinforcement-learning.txt
```

### Testing

```bash
pip install -r requirements/test.txt
```

## Optional Dependencies

### NVIDIA Apex (Advanced Mixed Precision)

NVIDIA Apex is **optional** and must be installed from source. PyTorch 2.0+ has built-in AMP support (`torch.cuda.amp`) which works for most use cases.

To install Apex (only if needed):

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

### FairScale (Advanced Distributed Features)

FairScale is **optional** for advanced distributed training features:

```bash
pip install fairscale
```

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip

# Install CUDA (if using GPU)
# See: https://developer.nvidia.com/cuda-downloads

# Install PyTorch testing framework
pip install -r requirements/base.txt
```

### RHEL/CentOS

```bash
# Install system dependencies
sudo yum install -y python3-devel

# Install PyTorch testing framework
pip install -r requirements/base.txt
```

### macOS

```bash
# PyTorch with CPU only (no CUDA on macOS)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements/base.txt
```

### Windows

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements/base.txt
```

## Conda Installation

Alternatively, use conda:

```bash
# Create environment
conda create -n pytorch-test python=3.10

# Activate environment
conda activate pytorch-test

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements/base.txt
```

## Verification

Verify your installation:

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU count
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Run quick tests
./run_tests.sh quick
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements/base.txt
```

### CUDA Version Mismatch

Check your CUDA version:

```bash
nvcc --version
```

Install matching PyTorch version:
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`

### py3nvml Installation Issues

If `py3nvml` fails to install:

```bash
# It's optional, comment it out in requirements/base.txt
# GPU monitoring will be limited but workloads will still run
```

### Matplotlib Backend Issues

If matplotlib has display issues:

```bash
# Set backend for headless environments
export MPLBACKEND=Agg
```

## Development Installation

For development with all optional dependencies:

```bash
# Install in editable mode
pip install -e .

# Install all requirements
pip install -r requirements/base.txt
pip install -r requirements/distributed.txt
pip install -r requirements/reinforcement-learning.txt
pip install -r requirements/test.txt

# Install code quality tools
pip install black isort flake8 pylint mypy
```

## Docker Installation

Use the provided Dockerfile (if available) or create your own:

```bash
docker build -t pytorch-test-suite .
docker run --gpus all -it pytorch-test-suite
```

## Minimal Requirements

Absolute minimum for testing:

- Python 3.8+
- PyTorch 2.0+
- NumPy
- PyYAML
- tqdm

All other dependencies are optional for enhanced functionality.

## Next Steps

After installation:

1. Run quick validation: `./run_tests.sh quick`
2. Try a workload: `./deployment/standalone/run_single.sh --workload cnn_training`
3. Check documentation: `README.md`
