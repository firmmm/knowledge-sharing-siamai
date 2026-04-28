# SGLANG GUIDE
## Serve using sglang library (quick experiment)

```bash

#install vllm via pip
pip install sglang

#basic command for serve model
python3 -m sglang.launch_server \
    --model-path "google/gemma-4-26B-A4B-it" \
    --host 0.0.0.0 \
    --port 30000
```

## Serve via docker(recommend)
### basic command with docker

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path "google/gemma-4-26B-A4B-it" \
        --host 0.0.0.0 \
        --port 30000
```
for some specific command example : [here](inference_engine/sglang/run_sglang.sh)
