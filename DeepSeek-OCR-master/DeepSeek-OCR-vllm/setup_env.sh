#!/bin/bash

# Setup script for DeepSeek-OCR API Server
# This will check and install dependencies

echo "🔍 Checking Python environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Found: $PYTHON_VERSION"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment found"
    source venv/bin/activate
else
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created"
fi

echo ""
echo "📥 Installing dependencies..."
echo "This may take a few minutes..."

# Upgrade pip first
pip install --upgrade pip

# Install base requirements
if [ -f "../requirements.txt" ]; then
    echo "Installing base requirements..."
    pip install -r ../requirements.txt
fi

# Install API requirements
if [ -f "api_requirements.txt" ]; then
    echo "Installing API requirements..."
    pip install -r api_requirements.txt
fi

# Check for torch (this is the main dependency)
echo ""
echo "🔍 Checking for PyTorch..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')" 2>/dev/null || {
    echo "⚠️  PyTorch not found. You may need to install it separately."
    echo "   For CUDA 11.8: pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118"
    echo "   For CPU only: pip install torch torchvision torchaudio"
}

# Check for vLLM
echo ""
echo "🔍 Checking for vLLM..."
python3 -c "import vllm; print(f'✅ vLLM installed')" 2>/dev/null || {
    echo "⚠️  vLLM not found. You need to install vLLM separately."
    echo "   See README.md for installation instructions"
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run the API server with:"
echo "  python3 api_server.py"

