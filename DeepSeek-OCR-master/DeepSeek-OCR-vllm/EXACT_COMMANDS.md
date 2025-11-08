# Exact Commands for Intel Mac Fix

Copy and paste these commands in order:

## Complete Fix (Copy-Paste Ready)

```bash
# Navigate to project
cd /Users/macbook/Deepseek-Ocr/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Activate venv
source venv/bin/activate

# Clean HuggingFace cache
rm -rf ~/.cache/huggingface/hub

# Clean pip cache
pip cache purge

# Uninstall problematic packages
pip uninstall -y transformers torch torchvision torchaudio accelerate safetensors

# Install CPU-only PyTorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install compatible transformers
pip install transformers==4.36.0

# Install other dependencies
pip install accelerate==0.25.0 safetensors==0.4.1
pip install addict matplotlib einops easydict pillow "numpy<2.0"
pip install fastapi uvicorn python-multipart pydantic

# Verify installation
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Run server
python3 api_server_hf.py
```

## Or Use the Automated Script

```bash
cd /Users/macbook/Deepseek-Ocr/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm
./setup_intel_mac.sh
source venv/bin/activate
python3 api_server_hf.py
```

## Code Changes Made

The `api_server_hf.py` file has been updated. Key changes:

**Line 132-135**: Force CPU mode
```python
# FORCE CPU MODE for Intel Mac compatibility
device = "cpu"
print("Using CPU (forced for Intel Mac compatibility)")
```

**Line 140-146**: Use SDPA instead of FlashAttention2
```python
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    _attn_implementation='sdpa'  # No FlashAttention2
)
```

**Line 151**: Force float32 for CPU
```python
model = model.to(device).to(torch.float32)
```

## Expected Output

When you run `python3 api_server_hf.py`, you should see:

```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
Initializing DeepSeek-OCR model (HuggingFace)...
Loading tokenizer...
Loading model (this may take a while)...
Using CPU (forced for Intel Mac compatibility)
Loading with default attention (no FlashAttention2)...
Model loaded on cpu with float32 precision
Model initialized in X.XX seconds
Device: cpu
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**No errors!** ✅

## Test It

```bash
# In another terminal
curl http://localhost:8000/health

# Or open browser
open http://localhost:8000
```

