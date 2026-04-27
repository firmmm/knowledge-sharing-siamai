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
    --model mistralai/Mistral-7B-v0.1
```

### for some specific command : [here]()
