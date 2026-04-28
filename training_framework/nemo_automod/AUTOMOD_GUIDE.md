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

This configuration file sets up distributed fine-tuning of a Qwen3-32B model using NeMo AutoModel with LoRA PEFT. Here's a comprehensive explanation of each section:

## Training Control

**step_scheduler**: Controls the training loop timing and batch sizes
- `global_batch_size: 64` - Total batch size across all GPUs
- `local_batch_size: 1` - Batch size per GPU (64 GPUs × 1 = 64 global)
- `ckpt_every_steps: 100` - Save checkpoint every 100 training steps
- `val_every_steps: 100` - Run validation every 100 steps
- `num_epochs: 3` - Train for 3 complete epochs

## Distributed Environment

**dist_env**: Sets up distributed training backend
- `backend: nccl` - Use NVIDIA's NCCL for GPU communication
- `timeout_minutes: 120` - Timeout for distributed operations

## Random Number Generation

**rng**: Configures reproducible randomness
- `_target_: nemo_automodel.components.training.rng.StatefulRNG` - Stateful RNG implementation
- `seed: 1111` - Fixed seed for reproducibility
- `ranked: true` - Different seeds per GPU rank

## Model Configuration

**model**: Base model setup
- `_target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained` - Model loader
- `pretrained_model_name_or_path: Qwen/Qwen3-32B` - HuggingFace model identifier
- `torch_dtype: bf16` - Use bfloat16 precision
- `use_liger_kernel: true` - Enable optimized Liger kernels
- Commented quantization section shows optional 4-bit quantization setup

## Checkpointing

**checkpoint**: Model saving configuration
- `enabled: true` - Enable checkpointing
- `checkpoint_dir: /home/firm/training/ckpts/run2_3` - Save location
- `model_save_format: safetensors` - Use safetensors format (safer than pickle)
- `save_consolidated: false` - Keep distributed shards (faster for training resumption)
- `is_async: true` - Non-blocking checkpoint saves

## Distributed Training Strategy

**distributed**: FSDP2 configuration for large-scale training
- `_target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager` - FSDP2 implementation
- `dp_size: 8` - Data parallel size (8 GPUs)
- `tp_size: 1` - Tensor parallel size (disabled)
- `cp_size: 1` - Context parallel size (disabled)
- `sequence_parallel: true` - Enable sequence parallelism
- `activation_checkpointing: true` - Trade compute for memory savings

## Parameter-Efficient Fine-Tuning

**peft**: LoRA configuration for memory-efficient training
- `_target_: nemo_automodel.components._peft.lora.PeftConfig` - LoRA config class
- `match_all_linear: True` - Apply LoRA to all linear layers
- `dim: 64` - Low-rank adapter dimension
- `alpha: 32` - LoRA scaling factor (alpha/dim = 0.5)
- `use_triton: True` - Use optimized Triton kernels

## Loss Function

**loss_fn**: Training objective
- `_target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy` - Masked cross-entropy for QA tasks

## Dataset Configuration

**dataset**: Training data setup
- `_target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset` - SQuAD dataset loader
- `dataset_name: /real-all-data/finetune-data/squad-data/spilt` - Local dataset path
- `split: train` - Use training split
- `seq_length: 512` - Maximum sequence length

**packed_sequence**: Memory optimization
- `packed_sequence_size: 32768` - Pack multiple sequences into batches

## Data Loading

**dataloader**: Training data configuration
- `_target_: torchdata.stateful_dataloader.StatefulDataLoader` - Stateful dataloader for resumption
- `collate_fn: nemo_automodel.components.datasets.utils.default_collater` - Default collation
- `shuffle: true` - Shuffle training data

**validation_dataset** and **validation_dataloader**: Same setup but for validation split

## Optimization

**optimizer**: Training optimizer
- `_target_: torch.optim.AdamW` - AdamW optimizer
- `betas: [0.9, 0.95]` - Momentum parameters
- `eps: 1e-8` - Numerical stability
- `lr: 5.0e-5` - Learning rate
- `weight_decay: 0.1` - L2 regularization

**clip_grad_norm**: Gradient stability
- `max_norm: 1.0` - Clip gradients to prevent explosion

**lr_scheduler**: Learning rate schedule
- `lr_warmup_steps: 100` - Linear warmup for 100 steps
- `lr_decay_steps: 12000` - Total decay steps
- `lr_decay_style: cosine` - Cosine annealing schedule

## Experiment Tracking

**wandb**: Weights & Biases logging
- `project: qwen3_32b_ex_automod` - Project name for experiment tracking
