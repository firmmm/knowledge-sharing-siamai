# AUTOMODEL GUIDE

## Set up environment for automodel
The container can be run with the following docker command:
```bash
docker run --gpus all --network=host -it --rm -v <local-Automodel-path>:/opt/Automodel --shm-size=32g nvcr.io/nvidia/nemo-automodel:25.11.00 /bin/bash
```
Within the container, cd into /opt/Automodel/ and update the pyproject.toml and uv.lock scripts by chanting the following:
```bash
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
```
Finally, perform the uv sync ritual to align the container with the updated repository:
```bash
uv sync --locked --extra all --all-group
```
Ref : [automod_set_env](https://github.com/NVIDIA-NeMo/Automodel?tab=contributing-ov-file#1-developing-with-automodel-container)

## Create config.yaml and run training
Create config.yaml at [config.yaml](training_framework/nemo_automod/config.yaml)
and run the training with:
```bash
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py -c /path/to/your/config.yaml
```
[run_automod.sh](training_framework/nemo_automod/run_automod.sh)

## Configuration Explanation

### Training Control
```yaml
step_scheduler:
  global_batch_size: 32        # Total batch size across all GPUs
  local_batch_size: 1          # Batch size per GPU
  ckpt_every_steps: 50         # Save checkpoint every 50 gradient steps
  val_every_steps: 50          # Run validation every 50 gradient steps
  num_epochs: 5                # Total number of training epochs
  # max_steps: 100             # Alternative to epochs - max training steps
```
The `StepScheduler` manages gradient accumulation and training loop timing.

### Distributed Training Environment
```yaml
dist_env:
  backend: nccl               # Use NCCL for GPU communication
  timeout_minutes: 1          # Timeout for distributed setup
```

### Random Number Generation
```yaml
rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111                  # Random seed for reproducibility
  ranked: true                # Different seeds per GPU rank
```

### Model Configuration
```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-32B  # Hugging Face model ID
  torch_dtype: bf16          # Use bfloat16 precision
  attn_implementation: "sdpa"  # Use Scaled Dot-Product Attention
```

### Checkpointing
```yaml
checkpoint:
  enabled: true
  checkpoint_dir: /home/firm/training/ckpts/automod-qwen32b-csld
  model_save_format: safetensors  # Save in safetensors format for inference
  save_consolidated: true         # Create HF-compatible consolidated model
```

### Distributed Training Manager
```yaml
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none              # Data parallel size (auto-detect)
  tp_size: 1                 # Tensor parallel size
  cp_size: 1                 # Context parallel size
  sequence_parallel: true    # Enable sequence parallelism
```

### Parameter-Efficient Fine-Tuning (LoRA)
```yaml
peft:  
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: True     # Apply LoRA to all linear layers
  dim: 8                     # LoRA rank (low-rank dimension)
  alpha: 32                  # LoRA scaling factor
  use_triton: True           # Use optimized Triton kernels
```
The `PeftConfig` class defines these LoRA parameters [2](#0-1) .

### Loss Function
```yaml
loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy
```

### Dataset Configuration
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: /home/firm/training/data/fintune-data/automodel/squad_data/split-data
  split: train
```
The `make_squad_dataset` function loads and preprocesses SQuAD-style QA data [3](#0-2) .

### Sequence Packing
```yaml
packed_sequence:
  packed_sequence_size: 0    # Disabled (0 = no packing)
```

### Data Loading
```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: true
```

### Validation Configuration
```yaml
validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: /home/firm/training/data/fintune-data/automodel/squad_data/split-data
  split: validation

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
```

### Test Dataset
```yaml
test_dataset:  
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset  
  dataset_name: /home/firm/training/data/fintune-data/automodel/squad_data/split-data
  split: test
```

### Optimizer Settings
```yaml
optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]        # Adam beta parameters
  eps: 1e-8                  # Numerical stability epsilon
  lr: 5.0e-6                 # Learning rate
  weight_decay: 0            # L2 regularization
  # min_lr: 1.0e-5           # Minimum learning rate (optional)
```

### Experiment Tracking
```yaml
wandb:
  project: qwen3_32b_squad_automod  # Weights & Biases project name
```
