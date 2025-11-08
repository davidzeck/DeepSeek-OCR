#!/bin/bash

# Startup script for DeepSeek-OCR API Server
# Usage: ./start_api.sh [port] [host]

PORT=${1:-8000}
HOST=${2:-0.0.0.0}

echo "Starting DeepSeek-OCR API Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo ""

# Check if GPU is available
python3 -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# Start the server
python3 api_server.py --host $HOST --port $PORT

