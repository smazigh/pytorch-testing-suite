# Kubernetes Deployment

Deploy PyTorch testing workloads on Kubernetes clusters.

## Prerequisites

- Kubernetes 1.24+
- NVIDIA GPU Operator or device plugin installed
- kubectl configured to access your cluster
- Container registry access (Docker Hub, GCR, ECR, etc.)
- PyTorch Training Operator (for multi-node jobs)

## Quick Start

### 1. Install Prerequisites

#### Install NVIDIA GPU Operator

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator
```

#### Install PyTorch Training Operator (for multi-node)

```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
```

### 2. Build Container Image (Optional)

If you want to use a custom image:

```bash
# Build image
docker build -t your-registry/pytorch-test-suite:latest .

# Push to registry
docker push your-registry/pytorch-test-suite:latest
```

Or use the official PyTorch image (default in manifests):
- `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

### 3. Create ConfigMap with Code

```bash
# Create ConfigMap from repository
kubectl create configmap pytorch-test-suite-code \
  --from-file=config/ \
  --from-file=workloads/ \
  --from-file=utils/ \
  --from-file=requirements/
```

### 4. Run Workloads

```bash
# Single-node job
kubectl apply -f deployment/kubernetes/single-node-job.yaml

# Multi-node job (requires PyTorch operator)
kubectl apply -f deployment/kubernetes/multi-node-job.yaml
```

## Single-Node Jobs

### Basic Job

The `single-node-job.yaml` runs a workload on a single GPU:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-single-node
spec:
  template:
    spec:
      containers:
      - name: pytorch-workload
        image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
        command: ["/bin/bash", "-c"]
        args:
          - |
            cd /workspace
            pip install -r requirements/base.txt
            python3 workloads/single_node/cnn_training.py
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Running Different Workloads

Edit the `args` section to run different workloads:

```yaml
# CNN Training
args:
  - |
    python3 workloads/single_node/cnn_training.py --config config/config.yaml

# GPU Burn-in
args:
  - |
    python3 workloads/single_node/gpu_burnin.py --config config/config.yaml

# Transformer Training
args:
  - |
    python3 workloads/single_node/transformer_training.py --config config/config.yaml
```

### Deploy and Monitor

```bash
# Deploy job
kubectl apply -f single-node-job.yaml

# Check status
kubectl get jobs
kubectl get pods

# View logs
kubectl logs -f pytorch-single-node-xxxxx

# Get results (if using persistent volume)
kubectl cp pytorch-single-node-xxxxx:/workspace/results ./results
```

## Multi-Node Jobs

### Prerequisites

- PyTorch Training Operator installed
- Multiple GPU nodes in cluster
- Shared storage (NFS, EFS) or object storage for results

### PyTorchJob Manifest

The `multi-node-job.yaml` uses the PyTorchJob CRD:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-multi-node-ddp
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch-master
            image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
            resources:
              limits:
                nvidia.com/gpu: 1

    Worker:
      replicas: 3
      template:
        spec:
          containers:
          - name: pytorch-worker
            image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
            resources:
              limits:
                nvidia.com/gpu: 1
```

### Deploy Multi-Node Job

```bash
# Create PVC for results (optional)
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pytorch-results-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
EOF

# Deploy multi-node job
kubectl apply -f multi-node-job.yaml

# Monitor
kubectl get pytorchjobs
kubectl get pods -l app=pytorch-test-suite

# View master logs
kubectl logs pytorch-multi-node-ddp-master-0

# View worker logs
kubectl logs pytorch-multi-node-ddp-worker-0
```

## GPU Burn-in DaemonSet

Run burn-in tests on all GPU nodes simultaneously:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-burnin
spec:
  selector:
    matchLabels:
      app: gpu-burnin
  template:
    metadata:
      labels:
        app: gpu-burnin
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"
      containers:
      - name: burnin
        image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
        command: ["/bin/bash", "-c"]
        args:
          - |
            cd /workspace
            pip install -r requirements/base.txt
            python3 workloads/single_node/gpu_burnin.py
        resources:
          limits:
            nvidia.com/gpu: 1
```

```bash
# Deploy
kubectl apply -f gpu-burnin-daemonset.yaml

# Check status on all nodes
kubectl get pods -l app=gpu-burnin -o wide

# View logs from specific node
kubectl logs gpu-burnin-xxxxx
```

## Configuration Management

### Using ConfigMaps

Store configuration in ConfigMaps:

```bash
# Create config ConfigMap
kubectl create configmap pytorch-config \
  --from-file=config/config.yaml

# Reference in pod
volumes:
- name: config
  configMap:
    name: pytorch-config
volumeMounts:
- name: config
  mountPath: /workspace/config
```

### Using Secrets

For sensitive configuration:

```bash
# Create secret
kubectl create secret generic pytorch-secrets \
  --from-literal=api-key=your-key

# Reference in pod
env:
- name: API_KEY
  valueFrom:
    secretKeyRef:
      name: pytorch-secrets
      key: api-key
```

## Storage Options

### EmptyDir (Temporary)

For test runs where results aren't needed:

```yaml
volumes:
- name: results
  emptyDir: {}
```

### PersistentVolume (Persistent)

For saving results:

```yaml
# Create PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pytorch-results
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi

# Use in pod
volumes:
- name: results
  persistentVolumeClaim:
    claimName: pytorch-results
```

### Object Storage (S3, GCS)

Mount object storage:

```yaml
# Using s3fs or similar
volumes:
- name: s3-storage
  csi:
    driver: s3.csi.aws.com
    volumeAttributes:
      bucketName: my-results-bucket
```

## Resource Management

### GPU Requests and Limits

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
    memory: "16Gi"
    cpu: "4"
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi"
    cpu: "8"
```

### Node Affinity

Target specific GPU types:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: gpu.nvidia.com/class
          operator: In
          values:
          - A100
          - H100
```

### Tolerations

For tainted GPU nodes:

```yaml
tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

## Monitoring and Debugging

### View Logs

```bash
# Tail logs
kubectl logs -f pod-name

# Previous container logs
kubectl logs pod-name --previous

# All containers in pod
kubectl logs pod-name --all-containers
```

### Debug Pod

```bash
# Get shell in running pod
kubectl exec -it pod-name -- /bin/bash

# Check GPU
kubectl exec pod-name -- nvidia-smi

# View resources
kubectl describe pod pod-name
```

### GPU Metrics

If using GPU operator with DCGM:

```bash
# View GPU metrics
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes | jq .

# Or use Prometheus queries
kubectl port-forward -n gpu-operator service/nvidia-dcgm-exporter 9400:9400
```

## Troubleshooting

### Pod Stuck in Pending

**Symptoms**: Pod not starting

**Check**:
```bash
kubectl describe pod pod-name
```

**Common causes**:
- Insufficient GPU resources
- Node selector mismatch
- PVC not available
- Image pull errors

**Solutions**:
- Check GPU availability: `kubectl get nodes -o json | jq '.items[].status.allocatable'`
- Verify node labels match selectors
- Check PVC status: `kubectl get pvc`
- Verify image name and registry access

### CUDA/NCCL Errors

**Problem**: CUDA initialization failed

**Solutions**:
1. Verify GPU operator is running:
   ```bash
   kubectl get pods -n gpu-operator
   ```

2. Check device plugin:
   ```bash
   kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
   ```

3. Recreate pod:
   ```bash
   kubectl delete pod pod-name
   ```

### OOM Killed

**Problem**: Pod killed due to memory

**Solutions**:
1. Increase memory limits
2. Reduce batch size in config
3. Enable gradient checkpointing

## Best Practices

1. **Use Init Containers**: Install dependencies before main container
   ```yaml
   initContainers:
   - name: install-deps
     image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
     command: ["pip", "install", "-r", "/workspace/requirements/base.txt"]
   ```

2. **Resource Limits**: Always set limits to prevent resource exhaustion

3. **Health Checks**: Add liveness/readiness probes for long-running jobs

4. **Logging**: Use structured logging and log aggregation

5. **Version Control**: Tag images and configurations

## Next Steps

- Review [Main README](../../README.md) for overview
- Check [SLURM deployment](../slurm/README.md) for HPC clusters
- Customize manifests for your cluster
- Set up monitoring and alerting
