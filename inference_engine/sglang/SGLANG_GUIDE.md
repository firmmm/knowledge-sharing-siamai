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

## Docker Command Arguments Explanation
```bash
docker run -d --gpus all \  # Run detached with all GPUs
    --restart=always \        # Auto-restart container
    --shm-size 200g \         # 200GB shared memory for multi-GPU communication
    -p 8003:30000 \           # Map host port 8003 to container port 30000
    -v /raid/hf_model:/root/.cache/huggingface \  # Mount model cache
    --env-file /raid/firm/script/.env \  # Load environment variables
    -e "CUDA_VISIBLE_DEVICES=1,2,3,4" \  # Use GPUs 1-4
    --ipc=host \              # Use host IPC for performance
    --name gemma-3-27b-it-sglang-firm \  # Container name
    lmsysorg/sglang:latest \  # SGLang Docker image
    python3 -m sglang.launch_server \
        --model-path google/gemma-3-27b-it \  # Model to load  
    	--host 0.0.0.0 \                      # Bind to all interfaces  
    	--port 30000 \                         # Server port  
    	--tp 2 \                               # Tensor parallelism across 2 GPUs [2](#1-1)   
    	--dp-size 2 \                          # Data parallelism groups [3](#1-2)   
    	--mem-fraction-static 0.6 \            # 60% GPU memory for KV cache [4](#1-3)   
    	--chunked-prefill-size 8192 \          # Chunk size for long prompts [5](#1-4)   
    	--max-running-requests 500 \           # Max concurrent requests  
    	--enable-metrics \                     # Enable Prometheus metrics [6](#1-5)   
    	--max-prefill-tokens 32768 \           # Max prefill tokens  
    	--schedule-conservativeness 0.3        # Scheduling conservativeness

```

Ref : [sglang_serve_docs](https://sgl-project.github.io/advanced_features/server_arguments.html)
