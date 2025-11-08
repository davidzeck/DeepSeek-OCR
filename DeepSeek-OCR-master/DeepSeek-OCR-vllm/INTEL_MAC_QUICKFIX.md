# Quick Fix for Intel Mac - Step by Step

## The Problem

- `ImportError: cannot import name 'LlamaFlashAttention2'` 
- Code incorrectly detects MPS on Intel Mac
- FlashAttention2 not compatible with Intel Mac CPU

## The Solution (5 Steps)

### Step 1: Run Setup Script

```bash
cd /Users/macbook/Deepseek-Ocr/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm
./setup_intel_mac.sh
```

**What it does:**
- Cleans HuggingFace cache (removes incompatible models)
- Uninstalls problematic packages
- Installs CPU-only PyTorch 2.1.0
- Installs Transformers 4.36.0 (no FlashAttention2)
- Installs compatible dependencies

**Why:** Fresh start with correct versions prevents conflicts.

---

### Step 2: Verify Installation

```bash
source venv/bin/activate
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**Expected output:**
```
PyTorch: 2.1.0
CUDA: False
```

**Why:** Confirms CPU-only PyTorch is installed.

---

### Step 3: Code Already Fixed

The `api_server_hf.py` has been updated to:
- ✅ Force `device = "cpu"` (line 132)
- ✅ Skip MPS detection
- ✅ Use `_attn_implementation='sdpa'` instead of FlashAttention2
- ✅ Use float32 for CPU

**What changed:**
```python
# OLD (problematic):
if torch.backends.mps.is_available():
    device = "mps"  # ❌ Causes issues on Intel Mac

# NEW (fixed):
device = "cpu"  # ✅ Always CPU for Intel Mac
print("Using CPU (forced for Intel Mac compatibility)")
```

**Why:** Explicitly forces CPU mode, avoiding MPS detection bugs.

---

### Step 4: Run Server

```bash
source venv/bin/activate
python3 api_server_hf.py
```

**Expected output:**
```
Initializing DeepSeek-OCR model (HuggingFace)...
Loading tokenizer...
Loading model (this may take a while)...
Using CPU (forced for Intel Mac compatibility)
Loading with default attention (no FlashAttention2)...
Model loaded on cpu with float32 precision
Model initialized in X seconds
Device: cpu
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

**No errors about:**
- ❌ LlamaFlashAttention2
- ❌ MPS
- ❌ FlashAttention2

---

### Step 5: Test

Open browser: http://localhost:8000

Or test API:
```bash
curl http://localhost:8000/health
```

---

## Manual Steps (if script fails)

If the automated script doesn't work, run these manually:

```bash
# 1. Activate venv
source venv/bin/activate

# 2. Clean cache
rm -rf ~/.cache/huggingface/hub
pip cache purge

# 3. Uninstall old packages
pip uninstall -y transformers torch torchvision torchaudio accelerate safetensors

# 4. Install CPU-only PyTorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 5. Install compatible transformers
pip install transformers==4.36.0

# 6. Install other dependencies
pip install accelerate==0.25.0 safetensors==0.4.1 addict matplotlib einops easydict pillow "numpy<2.0"
pip install fastapi uvicorn python-multipart pydantic

# 7. Run server
python3 api_server_hf.py
```

---

## Why Each Step Works

### 1. CPU-Only PyTorch
- **Why**: Eliminates MPS/CUDA dependencies
- **Result**: No GPU detection, pure CPU

### 2. Transformers 4.36.0
- **Why**: Stable version before FlashAttention2 became default
- **Result**: Uses SDPA (scaled dot product attention) instead

### 3. Force CPU in Code
- **Why**: Avoids MPS detection bugs on Intel Mac
- **Result**: Explicit CPU mode, no device switching

### 4. SDPA Instead of FlashAttention2
- **Why**: FlashAttention2 requires CUDA, causes import errors
- **Result**: Works on CPU, slightly slower but compatible

### 5. Float32 Precision
- **Why**: Best CPU compatibility, no precision issues
- **Result**: Stable inference on Intel CPU

---

## Performance Expectations

- **First inference**: 30-60 seconds (model warmup)
- **Subsequent**: 10-30 seconds per image
- **Memory**: ~8-12GB RAM usage
- **CPU**: Will use all available cores

**Tips for faster processing:**
- Use smaller `base_size` (512 or 640 instead of 1024)
- Use smaller `image_size` (512 instead of 640)
- Disable `crop_mode` for simple images

---

## Troubleshooting

### Still seeing MPS errors?
```bash
# Set environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=0
python3 api_server_hf.py
```

### Import errors persist?
```bash
# Force reinstall
pip install --force-reinstall transformers==4.36.0
```

### Model download fails?
- Check internet connection
- Model is ~5-10GB, be patient
- Check disk space: `df -h`

### Out of memory?
- Close other applications
- Reduce `base_size` and `image_size` in requests
- Restart server

---

## Success Checklist

✅ No `LlamaFlashAttention2` import errors  
✅ No MPS detection messages  
✅ Server starts without errors  
✅ Health endpoint responds: `{"status":"healthy"}`  
✅ Can upload images via web UI  
✅ OCR processing completes  

---

## Summary

**Root cause**:** FlashAttention2 and MPS detection incompatible with Intel Mac

**Fix**: 
1. CPU-only PyTorch
2. Older transformers (4.36.0)
3. Force CPU in code
4. Use SDPA attention

**Result**: DeepSeek-OCR runs on Intel Mac CPU! 🎉

