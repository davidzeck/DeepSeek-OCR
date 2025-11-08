#!/bin/bash

# vLLM Installation Helper
# Note: vLLM typically requires CUDA/GPU support

echo "🔍 vLLM Installation Helper"
echo ""

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup_env.sh first."
    exit 1
fi

echo ""
echo "⚠️  IMPORTANT: vLLM typically requires:"
echo "   - CUDA-capable GPU (NVIDIA)"
echo "   - CUDA 11.8 or 12.1"
echo "   - Linux (macOS support is limited)"
echo ""

# Check if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️  You're on macOS. vLLM has limited macOS support."
    echo "   Options:"
    echo "   1) Try installing vLLM anyway (may not work)"
    echo "   2) Use CPU-only mode (very slow)"
    echo "   3) Use remote GPU server"
    echo ""
    read -p "Continue anyway? [y/N]: " continue_choice
    if [[ ! $continue_choice =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

echo ""
echo "Select installation method:"
echo "1) Install vLLM from PyPI (latest, recommended)"
echo "2) Install vLLM from nightly build"
echo "3) Install specific version (0.8.5)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Installing vLLM from PyPI..."
        pip install vllm
        ;;
    2)
        echo "Installing vLLM from nightly build..."
        pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
        ;;
    3)
        echo "Installing vLLM 0.8.5..."
        echo "Note: You may need to download the wheel file manually"
        echo "See: https://github.com/vllm-project/vllm/releases/tag/v0.8.5"
        pip install vllm==0.8.5 || {
            echo "⚠️  Direct installation failed. You may need to:"
            echo "   1. Download the wheel file for your platform"
            echo "   2. Install with: pip install vllm-0.8.5*.whl"
        }
        ;;
    *)
        echo "Installing vLLM from PyPI (default)..."
        pip install vllm
        ;;
esac

echo ""
echo "Verifying installation..."
python3 -c "import vllm; print('✅ vLLM installed successfully')" 2>/dev/null || {
    echo "❌ vLLM installation failed or not compatible with your system"
    echo ""
    echo "Alternative options:"
    echo "1. Use the HuggingFace version (DeepSeek-OCR-hf) instead"
    echo "2. Run on a Linux server with GPU"
    echo "3. Use Docker with GPU support"
}

