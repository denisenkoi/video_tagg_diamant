#!/bin/bash
# Qwen3-VL-32B-Instruct-AWQ Model Launch Script
# Vision Language Model for video frame analysis

MODEL_NAME="Qwen3-VL-32B-Instruct-AWQ"
SERVED_NAME="Qwen/Qwen3-32B-AWQ"
HOST="0.0.0.0"
PORT="11434"

echo "🚀 Starting Qwen3-VL-32B-Instruct-AWQ model..."
echo "   Model: $MODEL_NAME"
echo "   Served as: $SERVED_NAME"
echo "   Host: $HOST:$PORT"
echo "   Time: $(date)"

# Activate conda environment
echo "📦 Activating conda environment 'vllm'..."
eval "$(/home/vano/miniconda3/condabin/conda shell.bash hook)"
conda activate vllm

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'vllm'"
    exit 1
fi

echo "✅ Conda environment activated"

# Ensure nvidia-smi and CUDA libs are available (WSL location)
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="/usr/lib/wsl/lib:/usr/bin:/usr/local/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Launch sglang server with Qwen3-VL parameters
echo "🔥 Launching sglang server..."
python -m sglang.launch_server \
    --model-path QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
    --host $HOST \
    --port $PORT \
    --mem-fraction-static 0.75 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e5m2 \
    --max-running-requests 2 \
    --context-length 8000 \
    --served-model-name "$SERVED_NAME"
