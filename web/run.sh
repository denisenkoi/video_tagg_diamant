#!/bin/bash
# Run Video Tagging Web Application

cd "$(dirname "$0")"

echo "Starting Video Tagging Web..."
echo "Open http://localhost:8000 in browser"
echo ""

/home/vano/anaconda3/envs/vllm/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
