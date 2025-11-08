#!/bin/bash

# Intel Mac Setup Script for DeepSeek-OCR
# Forces CPU mode and installs compatible dependencies

set -e  # Exit on error

echo "🔧 Intel Mac Setup for DeepSeek-OCR"
echo "===================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Step 1: Clean environment
echo ""
echo "🧹 Step 1: Cleaning environment..."
echo "   - Removing HuggingFace cache..."
rm -rf ~/.cache/huggingface/hub 2>/dev/null || true

echo "   - Clearing pip cache..."
pip cache purge 2>/dev/null || true

echo "   - Uninstalling old packages..."
pip uninstall -y transformers torch torchvision torchaudio accelerate safetensors 2>/dev/null || true

# Step 2: Install compatible versions
echo ""
echo "📥 Step 2: Installing compatible packages..."
echo "   - PyTorch CPU-only (2.1.0)..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

echo "   - Transformers 4.36.0 (stable, no FlashAttention2)..."
pip install transformers==4.36.0

echo "   - Accelerate and Safetensors..."
pip install accelerate==0.25.0 safetensors==0.4.1

echo "   - Other dependencies..."
pip install addict matplotlib einops easydict pillow "numpy<2.0"
pip install fastapi uvicorn python-multipart pydantic

# Step 3: Verify installation
echo ""
echo "✅ Step 3: Verifying installation..."
python3 -c "
import torch
import transformers
print('✅ PyTorch:', torch.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
print('✅ Device will be: cpu (forced)')
"

# Step 4: Test imports
echo ""
echo "🧪 Step 4: Testing imports..."
python3 -c "
from transformers import AutoModel, AutoTokenizer
print('✅ Transformers imports OK')
print('✅ Ready to load DeepSeek-OCR model')
" || {
    echo "❌ Import test failed!"
    exit 1
}

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run: source venv/bin/activate"
echo "  2. Run: python3 api_server_hf.py"
echo ""
echo "The server will:"
echo "  - Force CPU mode (no MPS/CUDA)"
echo "  - Use default attention (no FlashAttention2)"
echo "  - Load model with float32 precision"
echo ""

