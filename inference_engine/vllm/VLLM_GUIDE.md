# VLLM GUIDE
## Serve using vllm library (quick experiment)

```bash
#install vllm via pip
pip install vllm

#basic command for serve model
vllm serve "google/gemma-4-26B-A4B-it"
```

## Serve via docker (recommend)
### basic command with docker
```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model google/gemma-4-26B-A4B-it
```

for some specific command example : [here](https://github.com/firmmm/knowledge-sharing-siamai/blob/main/inference_engine/vllm/run_vllm.sh)

## Docker Command Arguments Explanation

```bash
docker run -d \                          # Run in background
  --name qwen-3-32B-vllm-prod-firm \     # Container name
  --restart always \                     # Auto restart if stopped
  --shm-size=200g \                      # Large shared memory (helps performance)
  --ipc=host \                           # Faster communication inside container
  --gpus all \                           # Use GPUs
  -p 8004:8000 \                         # Access API at port 8004
  -e CUDA_VISIBLE_DEVICES=0,1,2 \        # Use GPU 0,1,2
  -e HF_HOME=/hf_model \                 # Model cache path
  -v /raid/hf_model:/hf_model \          # Save models on host disk
  vllm/vllm-openai:v0.10.2 \             # vLLM image
  --model Qwen/Qwen3-32B \               # Load Qwen 32B model
  --host 0.0.0.0 \                       # Allow external access
  --port 8000 \                          # API port inside container
  --data-parallel-size 3 \               # Use 3 GPUs (copy model on each)
  --tensor-parallel-size 1 \             # No model splitting across GPUs
  --gpu-memory-utilization 0.9 \         # Use 90% GPU memory
  --enable-chunked-prefill \             # Better for long prompts
  --enable-prefix-caching \              # Faster repeated prompts
  --max-seq-len-to-capture 65536 \       # Optimize long inputs
  --max-num-batched-tokens 8192 \        # Batch size (speed vs latency)
  --max-num-seqs 1024 \                  # Max concurrent requests
  --max-model-len 131072 \               # Max context = 128K tokens
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \ # Extend context length
  --trust-remote-code \                  # Allow custom model code
  --dtype bfloat16 \                     # Faster + less memory
  --api-key siamai-dev-qwen              # API access key
```

## Example of google/gemma-4-26B-A4B-it
```bash
docker run -d \
  --name gemma-4-26b-vllm \
  --restart always \
  --gpus all \
  -p 8000:8000 \
  -e HF_HOME=/hf_model \
  -v /raid/hf_model:/hf_model \
  vllm/vllm-openai:latest \
  --model google/gemma-4-26B-A4B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \ #Enables automatic tool selection when the model deems it appropriate
  --reasoning-parser gemma4 \ #Uses Gemma4-specific parser for reasoning outputs
  --tool-call-parser gemma4 \ #Uses Gemma4-specific parser for tool calls in custom format
  --async-scheduling #Enables asynchronous scheduling for improved throughput
```
Ref : [vllm_serve_docs](https://docs.vllm.ai/en/v0.8.4/serving/openai_compatible_server.html#openai-compatible-server)
