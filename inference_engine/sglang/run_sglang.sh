docker run -d --gpus all \
  --restart=always \
  --shm-size 200g \
  -p 8003:30000 \
  -v /raid/hf_model:/root/.cache/huggingface \
  --env-file /raid/firm/script/.env \
  -e "CUDA_VISIBLE_DEVICES=1,2,3,4" \
  --ipc=host \
  --name gemma-3-27b-it-sglang-firm \
  lmsysorg/sglang:latest \
  bash -c 'python3 -m sglang.launch_server \
  --model-path google/gemma-3-27b-it \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 2 \
  --dp-size 2 \
  --mem-fraction-static 0.6 \
  --chunked-prefill-size 8192 \
  --max-running-requests 500 \
  --enable-metrics \
  --max-prefill-tokens 32768 \
  --schedule-conservativeness 0.3
