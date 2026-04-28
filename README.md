# Knowledge Sharing - Siam AI

Internal team knowledge repository for LLM inference engines and training frameworks.

## 📚 Repository Structure

```
knowledge-sharing-siamai/
├── inference_engine/       # LLM Serving & Inference
│   ├── sglang/            # SGLang inference engine
│   └── vllm/              # vLLM inference engine
└── training_framework/     # Model Training & Fine-tuning
    ├── nemo_automod/      # NeMo AutoModel fine-tuning
    └── nemo_rl/           # NeMo RL (Reinforcement Learning)
```

## 🚀 Inference Engines

### SGLang (`inference_engine/sglang/`)

**Guide:** [`SGLANG_GUIDE.md`](inference_engine/sglang/SGLANG_GUIDE.md)

SGLang is optimized for serving models with advanced scheduling and memory management.

**Quick Start:**
```bash
# Install
pip install sglang

# Serve model
python3 -m sglang.launch_server \
    --model-path "google/gemma-4-26B-A4B-it" \
    --host 0.0.0.0 \
    --port 30000
```

**Docker (Recommended):**
```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path "google/gemma-4-26B-A4B-it" \
        --host 0.0.0.0 \
        --port 30000
```

**Key Features:**
- Tensor parallelism (`--tp`)
- Data parallelism (`--dp-size`)
- Chunked prefill for long prompts
- KV cache optimization (`--mem-fraction-static`)
- Prometheus metrics support

**Example Script:** [`run_sglang.sh`](inference_engine/sglang/run_sglang.sh)

---

### vLLM (`inference_engine/vllm/`)

**Guide:** [`VLLM_GUIDE.md`](inference_engine/vllm/VLLM_GUIDE.md)

vLLM provides high-throughput serving with OpenAI-compatible API.

**Quick Start:**
```bash
# Install
pip install vllm

# Serve model
vllm serve "google/gemma-4-26B-A4B-it"
```

**Docker (Recommended):**
```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model google/gemma-4-26B-A4B-it
```

**Key Features:**
- OpenAI-compatible API
- Continuous batching
- Prefix caching for repeated prompts
- Long context support (up to 128K tokens)
- Multi-GPU support (tensor/data parallelism)

**Example Script:** [`run_vllm.sh`](inference_engine/vllm/run_vllm.sh)

---

## 🎯 Training Frameworks

### NeMo AutoModel (`training_framework/nemo_automod/`)

**Guide:** [`AUTOMOD_GUIDE.md`](training_framework/nemo_automod/AUTOMOD_GUIDE.md)

NeMo AutoModel for supervised fine-tuning (SFT) with LoRA.

**Setup:**
```bash
docker run --gpus all --network=host -it --rm \
    -v <local-path>:/opt/Automodel \
    --shm-size=32g \
    nvcr.io/nvidia/nemo-automodel:25.11.00 /bin/bash

# Inside container
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
uv sync --locked --extra all --all-group
```

**Training:**
```bash
uv run torchrun --nproc-per-node=8 \
    examples/llm_finetune/finetune.py \
    -c /path/to/config.yaml
```

**Configuration:** [`config.yaml`](training_framework/nemo_automod/config.yaml)

**Key Features:**
- LoRA PEFT (Parameter-Efficient Fine-Tuning)
- FSDP2 distributed training
- bfloat16 precision
- SQuAD dataset support
- Weights & Biases integration
- Checkpointing with safetensors format

**Example Script:** [`run_automod.sh`](training_framework/nemo_automod/run_automod.sh)

---

### NeMo RL (`training_framework/nemo_rl/`)

**Guide:** [`RL_GUIDE.md`](training_framework/nemo_rl/RL_GUIDE.md)

NeMo RL for reinforcement learning with GRPO (Generalized Reward-Weighted Policy Optimization).

**Setup:**
```bash
# Clone with submodules
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl
git submodule update --init --recursive

# Create environment
uv venv

# Install dependencies
sudo apt-get update && sudo apt-get install libibverbs-dev
```

**Training:**
```bash
export PYTHONPATH="/path/to/config:$PYTHONPATH"
export NRL_FORCE_REBUILD_VENVS=true
cd /path/to/RL

uv run python - <<'EOF'
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY, PY_EXECUTABLES
)
ACTOR_ENVIRONMENT_REGISTRY["thai_reward_env.DualRewardEnvironmentActor"] = (
    PY_EXECUTABLES.SYSTEM
)

import thai_reward_env
import thai_data_processor
import runpy, sys

sys.argv = [
    "run_grpo.py",
    "--config", "/path/to/config.yaml",
    "cluster.gpus_per_node=8",
]
runpy.run_path("examples/run_grpo.py", run_name="__main__")
EOF
```

**Configuration Files:**
- **Config:** [`config.yaml`](training_framework/nemo_rl/config.yaml)
- **Data Processor:** [`thai_data_processor.py`](training_framework/nemo_rl/thai_data_processor.py)
- **Reward Environment:** [`thai_reward_env.py`](training_framework/nemo_rl/thai_reward_env.py)

**Key Features:**
- GRPO algorithm for RLHF
- Dual reward system:
  - **Thai correctness:** Typhoon2.5-Qwen3-4B (generative judge)
  - **Format quality:** Skywork-Reward-V2 (Bradley-Terry RM)
- Custom Thai language data processing
- vLLM integration for generation
- Megatron-LM distributed training
- LoRA adapters for policy model
- Weights & Biases logging

**Example Script:** [`run_rl.sh`](training_framework/nemo_rl/run_rl.sh)

---

## 🔧 Common Configuration Patterns

### GPU Configuration
- **Tensor Parallelism:** Split model across GPUs (`--tp`, `tensor_parallel_size`)
- **Data Parallelism:** Replicate model on multiple GPUs (`--dp-size`, `data_parallel_size`)
- **Memory Management:** Shared memory (`--shm-size`), KV cache fraction (`--mem-fraction-static`)

### Model Precision
- **bfloat16:** Recommended for most modern GPUs (A100, H100)
- **float16:** Legacy support

### Checkpointing
- **safetensors:** Safe format for inference/deployment
- **torch_save:** For training resumption
- **Consolidated:** Merge shards for deployment (`save_consolidated: true`)

---

## 📊 Monitoring & Logging

All frameworks support **Weights & Biases** for experiment tracking:

```yaml
wandb:
  project: "your-project-name"
  name: "experiment-name"
  enabled: true
```

vLLM and SGLang also support **Prometheus metrics** for production monitoring.

---

## 🛠️ Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `--mem-fraction-static` (SGLang) or `--gpu-memory-utilization` (vLLM)
- Enable activation checkpointing
- Reduce batch size

**2. Slow Performance**
- Enable chunked prefill for long prompts
- Enable prefix caching for repeated prompts
- Increase tensor parallelism

**3. Distributed Training Issues**
- Increase `timeout_minutes` in distributed config
- Check NCCL backend configuration
- Verify GPU visibility with `CUDA_VISIBLE_DEVICES`

---

## 📖 References

- [SGLang Documentation](https://sgl-project.github.io/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL)

---

## 👥 Team

Internal knowledge sharing repository - Siam AI Team
