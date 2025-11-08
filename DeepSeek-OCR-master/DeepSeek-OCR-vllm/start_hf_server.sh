#!/bin/bash

# Start HuggingFace-based API Server (macOS compatible)

echo "🚀 Starting DeepSeek-OCR API Server (HuggingFace)"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: python3 -m venv venv && source venv/bin/activate"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "import transformers; import torch; import fastapi" 2>/dev/null || {
    echo "❌ Missing dependencies!"
    echo "   Run: pip install transformers torch fastapi uvicorn"
    exit 1
}

echo "✅ Dependencies OK"
echo ""

# Check device
python3 -c "import torch; print('Device info:'); print('  MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A'); print('  CUDA:', torch.cuda.is_available()); print('  CPU: Always available')" 2>/dev/null

echo ""
echo "📡 Starting server on http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

# Start server
python3 api_server_hf.py

