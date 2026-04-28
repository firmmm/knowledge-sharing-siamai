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
- Config file at [config.yaml](training_framework/nemo_rl/config.yaml)
- processor at [thai_data_processor.py](training_framework/nemo_rl/thai_data_processor.py)
- reward at [thai_reward_env.py](training_framework/nemo_rl/thai_reward_env.py)
