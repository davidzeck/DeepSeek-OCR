#!/bin/bash

# PyTorch Installation Helper
# This script helps install the correct PyTorch version

echo "🔍 PyTorch Installation Helper"
echo ""

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

echo ""
echo "Select your PyTorch installation:"
echo "1) PyTorch 2.5.1 with CUDA 11.8 (GPU - Recommended for DeepSeek-OCR)"
echo "2) PyTorch 2.5.1 with CUDA 12.1 (GPU - Newer CUDA)"
echo "3) PyTorch 2.5.1 CPU only (No GPU)"
echo "4) Latest PyTorch (Auto-detect)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "Installing PyTorch 2.5.1 with CUDA 11.8..."
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "Installing PyTorch 2.5.1 with CUDA 12.1..."
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "Installing PyTorch 2.5.1 CPU only..."
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
        ;;
    4)
        echo "Installing latest PyTorch..."
        pip install torch torchvision torchaudio
        ;;
    *)
        echo "Invalid choice. Installing PyTorch 2.5.1 with CUDA 11.8 (default)..."
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
        ;;
esac

echo ""
echo "✅ PyTorch installation complete!"
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "CUDA check skipped"

