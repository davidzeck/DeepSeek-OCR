# macOS Setup Guide for DeepSeek-OCR

## ⚠️ Important Note

**vLLM requires CUDA/GPU support and primarily works on Linux.** On macOS:
- vLLM will **NOT work** (requires NVIDIA GPU + CUDA)
- The vLLM version of the API server **cannot run on macOS**

## Options for macOS Users

### Option 1: Use HuggingFace Version (Recommended for macOS)

The HuggingFace version works on macOS with CPU or Apple Silicon:

```bash
cd ../DeepSeek-OCR-hf
python3 run_dpsk_ocr.py
```

This uses the Transformers library which supports macOS.

### Option 2: Create API Server for HuggingFace Version

I can create a FastAPI server that uses the HuggingFace model instead of vLLM. This will work on macOS.

### Option 3: Use Remote GPU Server

1. Set up the vLLM server on a Linux machine with GPU
2. Connect from macOS to the remote API

### Option 4: Docker with GPU (if you have external GPU)

If you have an external NVIDIA GPU, you could use Docker, but this is complex on macOS.

## Current Status

✅ **Installed:**
- PyTorch 2.2.2 (CPU version)
- NumPy 1.26.4 (compatible)
- FastAPI and dependencies
- Base requirements

❌ **Missing/Incompatible:**
- vLLM (requires CUDA/GPU, Linux)

## Recommended Next Steps

**For macOS, I recommend Option 2** - creating a HuggingFace-based API server that will work on your Mac.

Would you like me to:
1. Create a HuggingFace-based API server (works on macOS)?
2. Help you set up a remote GPU server?
3. Try installing vLLM anyway (will likely fail)?

## Testing What You Have

Even without vLLM, you can test the basic setup:

```bash
source venv/bin/activate
python3 -c "import torch; print('PyTorch OK')"
python3 -c "import transformers; print('Transformers OK')"
python3 -c "import fastapi; print('FastAPI OK')"
```

