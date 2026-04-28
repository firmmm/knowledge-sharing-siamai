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
