# 🧙 Arcane Codex - Siam AI Grimoire

> *Internal knowledge repository for the Order of Siam AI - Where mere mortals command great language models through ancient incantations*

---

## 📜 Table of Contents

- [🏰 The Sanctuary](#-the-sanctuary)
- [⚡ Invocation Engines](#-invocation-engines)
  - [SGLang - The Swift Summoning](#sglang---the-swift-summoning)
  - [vLLM - The OpenAI Incantation](#vllm---the-openai-incantation)
- [🔮 Training Rituals](#-training-rituals)
  - [NeMo AutoModel - The Fine-Tuning Ceremony](#nemo-automodel---the-fine-tuning-ceremony)
  - [NeMo RL - The Reinforcement Enchantment](#nemo-rl---the-reinforcement-enchantment)
- [🧪 Alchemical Configurations](#-alchemical-configurations)
- [👁️ Scrying & Divination](#-scrying--divination)
- [⚠️ Curses & Counter-Curses](#-curses--counter-curses)
- [📚 Ancient Tomes](#-ancient-tomes)

---

## 🏰 The Sanctuary

```
knowledge-sharing-siamai/
├── inference_engine/       # Chambers of Invocation
│   ├── sglang/            # SGLang - Swift Summoning Arts
│   └── vllm/              # vLLM - OpenAI Compatibility Spells
└── training_framework/     # Ritual Circles of Training
    ├── nemo_automod/      # AutoModel - Fine-Tuning Enchantments
    └── nemo_rl/           # RL - Reinforcement Magic
```

---

## ⚡ Invocation Engines

### SGLang - The Swift Summoning

**📖 Scroll:** [`SGLANG_GUIDE.md`](inference_engine/sglang/SGLANG_GUIDE.md)

SGLang channels the fastest spirits for model manifestation with advanced scheduling and memory alchemy.

#### 🪄 Quick Invocation

```bash
# Bind the spirit to your realm
pip install sglang

# Summon the model (ancient incantation)
python3 -m sglang.launch_server \
    --model-path "google/gemma-4-26B-A4B-it" \
    --host 0.0.0.0 \
    --port 30000
```

#### 🐋 Docker Ritual (Recommended by the High Council)

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

#### ✨ Arcane Powers

| Rune | Purpose |
|------|---------|
| `--tp` | Tensor parallelism - splits the spirit across multiple crystals |
| `--dp-size` | Data parallelism - creates parallel ritual circles |
| `--chunked-prefill-size` | Breaks long scrolls into manageable fragments |
| `--mem-fraction-static` | Allocates mana pool for KV cache (0.6 = 60%) |
| `--enable-metrics` | Opens scrying portal for Prometheus |

**📜 Example Ritual:** [`run_sglang.sh`](inference_engine/sglang/run_sglang.sh)

---

### vLLM - The OpenAI Incantation

**📖 Scroll:** [`VLLM_GUIDE.md`](inference_engine/vllm/VLLM_GUIDE.md)

vLLM conjures high-throughput manifestations with OpenAI-compatible API enchantments.

#### 🪄 Quick Invocation

```bash
# Bind the spirit
pip install vllm

# Summon the model
vllm serve "google/gemma-4-26B-A4B-it"
```

#### 🐋 Docker Ritual (Recommended by the High Council)

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model google/gemma-4-26B-A4B-it
```

#### ✨ Arcane Powers

| Rune | Purpose |
|------|---------|
| `--tensor-parallel-size` | Distributes spirit essence across GPU crystals |
| `--data-parallel-size` | Creates multiple manifestation circles |
| `--enable-prefix-caching` | Remembers repeated incantations |
| `--max-model-len` | Maximum scroll length (131072 = 128K tokens) |
| `--rope-scaling` | Extends the wizard's memory beyond natural limits |
| `--api-key` | Protective ward against unauthorized summoners |

**📜 Example Ritual:** [`run_vllm.sh`](inference_engine/vllm/run_vllm.sh)

---

## 🔮 Training Rituals

### NeMo AutoModel - The Fine-Tuning Ceremony

**📖 Scroll:** [`AUTOMOD_GUIDE.md`](training_framework/nemo_automod/AUTOMOD_GUIDE.md)

The sacred ritual for bending pre-trained models to thy will using LoRA enchantments.

#### 🕯️ Preparation Ritual

```bash
# Summon the sacred vessel
docker run --gpus all --network=host -it --rm \
    -v <local-path>:/opt/Automodel \
    --shm-size=32g \
    nvcr.io/nvidia/nemo-automodel:25.11.00 /bin/bash

# Inside the vessel, speak these words:
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
uv sync --locked --extra all --all-group
```

#### ⚡ The Grand Ceremony

```bash
uv run torchrun --nproc-per-node=8 \
    examples/llm_finetune/finetune.py \
    -c /path/to/config.yaml
```

#### 📜 Sacred Configuration

**Config Scroll:** [`config.yaml`](training_framework/nemo_automod/config.yaml)

#### ✨ Mystical Properties

- **LoRA PEFT** - Parameter-Efficient Fine-Tuning enchantment
- **FSDP2** - Fully Sharded Data Parallel ritual circle
- **bfloat16** - Ethereal precision for faster computations
- **SQuAD Dataset** - Ancient texts of question and answer
- **Wands & Biases** - Crystal ball for tracking experimental outcomes
- **Safetensors** - Safe preservation of trained spirits

**📜 Example Ritual:** [`run_automod.sh`](training_framework/nemo_automod/run_automod.sh)

---

### NeMo RL - The Reinforcement Enchantment

**📖 Scroll:** [`RL_GUIDE.md`](training_framework/nemo_rl/RL_GUIDE.md)

The most advanced sorcery - teaching models through reward and punishment using GRPO (Generalized Reward-Weighted Policy Optimization).

#### 🕯️ Preparation Ritual

```bash
# Summon the forbidden tome with its hidden sub-spirits
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl
git submodule update --init --recursive

# Create the mana vessel
uv venv

# Install protective wards (required for deep_ep spirit)
sudo apt-get update && sudo apt-get install libibverbs-dev
```

#### ⚡ The Grand Ceremony

```bash
export PYTHONPATH="/path/to/config:$PYTHONPATH"
export NRL_FORCE_REBUILD_VENVS=true
cd /path/to/RL

uv run python - <<'EOF'
# Register the Thai reward spirit
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY, PY_EXECUTABLES
)
ACTOR_ENVIRONMENT_REGISTRY["thai_reward_env.DualRewardEnvironmentActor"] = (
    PY_EXECUTABLES.SYSTEM
)

# Awaken the ancient processors
import thai_reward_env
import thai_data_processor
import runpy, sys

# Speak the invocation words
sys.argv = [
    "run_grpo.py",
    "--config", "/path/to/config.yaml",
    "cluster.gpus_per_node=8",
]
runpy.run_path("examples/run_grpo.py", run_name="__main__")
EOF
```

#### 📜 Sacred Artifacts

| Artifact | Purpose |
|----------|---------|
| [`config.yaml`](training_framework/nemo_rl/config.yaml) | The Grand Grimoire of GRPO configuration |
| [`thai_data_processor.py`](training_framework/nemo_rl/thai_data_processor.py) | Translates Thai scrolls into model-tongue |
| [`thai_reward_env.py`](training_framework/nemo_rl/thai_reward_env.py) | The Dual-Judge Chamber |

#### ✨ The Dual-Judge System

The Thai Reinforcement Ritual employs **two mystical arbiters**:

1. **Typhoon2.5-Qwen3-4B** - The Correctness Oracle
   - Judges whether answers are *true* or *false*
   - Speaks only in Thai tongue
   - Returns: `rewards[:, 0]` (1.0 = correct, 0.0 = incorrect)

2. **Skywork-Reward-V2-Qwen3-0.6B** - The Format Guardian
   - Judges the *quality of presentation*
   - Uses Bradley-Terry divination
   - Returns: `rewards[:, 1]` (0.0 to 1.0 scale)

#### ✨ Arcane Properties

- **GRPO Algorithm** - Generalized Reward-Weighted Policy Optimization
- **Dual Reward Dimensions** - Two independent judgment axes
- **Thai Language Processing** - Native understanding of Siamese texts
- **vLLM Generation** - Swift spirit manifestation
- **Megatron-LM** - Distributed training across multiple realms
- **LoRA Adapters** - Lightweight policy modifications
- **Wands & Biases** - Crystal ball for experiment scrying

**📜 Example Ritual:** [`run_rl.sh`](training_framework/nemo_rl/run_rl.sh)

---

## 🧪 Alchemical Configurations

### GPU Crystal Arrangements

| Configuration | Incantation | Effect |
|--------------|-------------|--------|
| **Tensor Parallelism** | `--tp N` / `tensor_parallel_size: N` | Splits one model spirit across N crystals |
| **Data Parallelism** | `--dp-size N` / `data_parallel_size: N` | Creates N identical ritual circles |
| **Shared Memory** | `--shm-size 200g` | Expands the mana pool for inter-crystal communication |
| **KV Cache** | `--mem-fraction-static 0.6` | Allocates 60% of crystal memory for context |

### Precision Enchantments

| Precision | Rune | Best Used On |
|-----------|------|--------------|
| **bfloat16** | `--dtype bfloat16` / `torch_dtype: bf16` | A100, H100, modern crystals |
| **float16** | `--dtype float16` | Legacy GPU crystals |

### Preservation Scrolls

| Format | Purpose |
|--------|---------|
| **safetensors** | Safe for deployment and inference (no cursed pickle) |
| **torch_save** | For resuming interrupted rituals |
| **Consolidated** | Merge all shards for final deployment (`save_consolidated: true`) |

---

## 👁️ Scrying & Divination

### Wands & Biases Crystal Ball

All frameworks commune with the **Wands & Biases** oracle:

```yaml
wandb:
  project: "your-magical-project"
  name: "experiment-of-great-importance"
  enabled: true
```

### Prometheus Scrying Portal

vLLM and SGLang open portals for **Prometheus** monitoring:

```bash
--enable-metrics  # SGLang
--enable-metrics  # vLLM
```

---

## ⚠️ Curses & Counter-Curses

### Common Afflictions

#### 💀 Out of Mana (OOM Curse)

**Symptoms:** Model spirit consumes too much energy and collapses

**Counter-Curse:**
```bash
# Reduce memory allocation
--mem-fraction-static 0.5      # SGLang
--gpu-memory-utilization 0.8   # vLLM

# Enable activation checkpointing (trades speed for memory)
activation_checkpointing: true

# Reduce batch size
global_batch_size: 32  # was 64
```

#### 🐌 Slow Manifestation

**Symptoms:** Model responds slower than a sleeping dragon

**Counter-Curse:**
```bash
# Enable chunked prefill for long scrolls
--chunked-prefill-size 8192

# Enable prefix caching for repeated incantations
--enable-prefix-caching

# Increase tensor parallelism
--tp 4  # spread across more crystals
```

#### ⚡ Distributed Ritual Failures

**Symptoms:** Multi-GPU ceremony fails to commence

**Counter-Curse:**
```yaml
# Extend the ritual timeout
dist_env:
  timeout_minutes: 120  # was 30

# Verify NCCL backend
backend: nccl

# Check crystal visibility
CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## 📚 Ancient Tomes

- [📜 SGLang Grimoire](https://sgl-project.github.io/)
- [📜 vLLM Tome of Knowledge](https://docs.vllm.ai/)
- [📜 NeMo AutoModel Scrolls](https://github.com/NVIDIA-NeMo/Automodel)
- [📜 NeMo RL Forbidden Tomes](https://github.com/NVIDIA-NeMo/RL)

---

## 🏅 Order of Siam AI

*This grimoire is maintained by the wizards of the Siam AI Order. May your models be swift, your rewards be dense, and your GPUs never OOM.*

**⚡ Remember:** With great model size comes great responsibility.

---

*Last updated: Year of the Dragon, 2026*
