# Intel Mac Fix Guide - DeepSeek-OCR

## Problem Analysis

**Issue**: `ImportError: cannot import name 'LlamaFlashAttention2'` occurs because:
1. The code detects MPS (Apple Silicon) even on Intel Macs
2. FlashAttention2 is not compatible with Intel Macs
3. Dependency versions may be mismatched

**Solution**: Force CPU mode, use compatible transformer versions, and disable FlashAttention.

---

## Step-by-Step Fix

### Step 1: Clean Environment and Cache

**Why**: Remove corrupted or incompatible cached models and packages.

```bash
# Activate venv
cd /Users/macbook/Deepseek-Ocr/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm
source venv/bin/activate

# Clear HuggingFace cache (removes downloaded models)
rm -rf ~/.cache/huggingface/hub

# Clear pip cache
pip cache purge

# Uninstall problematic packages
pip uninstall -y transformers torch torchvision torchaudio accelerate safetensors
```

**What this does**: 
- Removes cached model files that might have FlashAttention dependencies
- Clears pip cache to force fresh downloads
- Removes packages to reinstall with correct versions

---

### Step 2: Install Compatible Versions

**Why**: Specific versions are needed for Intel Mac CPU compatibility without FlashAttention.

```bash
# Install PyTorch CPU-only (no CUDA/MPS dependencies)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install compatible transformers (avoid FlashAttention2 issues)
pip install transformers==4.36.0

# Install other required packages
pip install accelerate==0.25.0 safetensors==0.4.1

# Install remaining dependencies
pip install addict matplotlib einops easydict easydict pillow numpy
pip install fastapi uvicorn python-multipart pydantic
```

**What this does**:
- PyTorch 2.1.0 CPU-only: No MPS/CUDA, pure CPU
- Transformers 4.36.0: Stable version before FlashAttention2 became default
- Accelerate 0.25.0: Compatible with transformers 4.36.0
- Safetensors 0.4.1: Stable version for model loading

---

### Step 3: Fix api_server_hf.py

**Why**: Force CPU mode and disable FlashAttention detection.

We need to modify the `initialize_model()` function to:
1. Force device to "cpu"
2. Disable FlashAttention2
3. Use float32 for CPU

---

### Step 4: Verify Installation

```bash
# Check versions
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"

python3 -c "import transformers; print('Transformers:', transformers.__version__)"

# Test imports
python3 -c "from transformers import AutoModel, AutoTokenizer; print('✅ Imports OK')"
```

---

### Step 5: Run Server

```bash
python3 api_server_hf.py
```

---

## Expected Output

You should see:
```
Initializing DeepSeek-OCR model (HuggingFace)...
Loading tokenizer...
Loading model (this may take a while)...
Using CPU
Model initialized in X seconds
Device: cpu
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

**No errors about**:
- ❌ MPS
- ❌ FlashAttention2
- ❌ LlamaFlashAttention2

---

## Troubleshooting

### If you still see MPS errors:
```bash
# Explicitly set environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=0
python3 api_server_hf.py
```

### If transformers version conflicts:
```bash
pip install --force-reinstall transformers==4.36.0
```

### If model download fails:
```bash
# Set HuggingFace token if needed (optional)
export HF_TOKEN=your_token_here
```

---

## Why This Works

1. **CPU-only PyTorch**: Eliminates MPS/CUDA detection
2. **Older Transformers**: Avoids FlashAttention2 dependency
3. **Forced CPU in code**: Explicitly sets device="cpu"
4. **Float32**: CPU-friendly precision
5. **Clean cache**: Removes incompatible cached files

---

## Performance Notes

- **CPU mode is slower** than GPU but will work
- First inference: 30-60 seconds (model warmup)
- Subsequent: 10-30 seconds per image
- Consider smaller `base_size` and `image_size` for faster processing

