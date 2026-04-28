# RL GUIDE
## Set up environment for RL
```bash
# 1. Clone NeMo RL with its hidden submodules
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
#or git clone https://github.com/NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl

# 2. If the submodules were left in the void (cloned without --recursive):
git submodule update --init --recursive

# 3. Create the managed virtual sanctum
uv venv

# 4. Install the uv familiar if not present
# See: https://docs.astral.sh/uv/getting-started/installation/

# 5. (Bare metal only) Scry for cuDNN headers if missing
dpkg -l | grep cudnn.*cuda
# Follow NVIDIA cuDNN download instructions for your distro

# 6. (Optional) Scribe libibverbs-dev for deep_ep dependency
sudo apt-get update && sudo apt-get install libibverbs-dev
```
## Create config, processor and reward file
- Config file at [config.yaml](https://github.com/firmmm/knowledge-sharing-siamai/blob/main/training_framework/nemo_rl/config.yaml)
- processor at [thai_data_processor.py](https://github.com/firmmm/knowledge-sharing-siamai/blob/main/training_framework/nemo_rl/thai_data_processor.py)
- reward at [thai_reward_env.py](https://github.com/firmmm/knowledge-sharing-siamai/blob/main/training_framework/nemo_rl/thai_reward_env.py)


and run file with:
```bash
#!/bin/bash
export PYTHONPATH="/home/firm/training/conf/nemorl/gdpo:$PYTHONPATH"
export NRL_FORCE_REBUILD_VENVS=true
cd /home/firm/training/src/RL

uv run python - <<'EOF'
# ── register the python environment for the Ray actor ─────────────────────
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
ACTOR_ENVIRONMENT_REGISTRY["thai_reward_env.DualRewardEnvironmentActor"] = (
    PY_EXECUTABLES.SYSTEM
)

import thai_reward_env
import thai_data_processor

import runpy, sys
sys.argv = [
    "run_grpo.py",
    "--config", "/home/firm/training/conf/nemorl/gdpo/qwen32b-fix.yaml",
    "cluster.gpus_per_node=8",
]
runpy.run_path("examples/run_grpo.py", run_name="__main__")
EOF
```
at [run_rl.sh](https://github.com/firmmm/knowledge-sharing-siamai/blob/main/training_framework/nemo_rl/run_rl.sh)
