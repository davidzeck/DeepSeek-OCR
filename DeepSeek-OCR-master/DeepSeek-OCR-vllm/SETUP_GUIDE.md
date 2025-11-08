# Setup Guide for DeepSeek-OCR API

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
./setup_env.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Check for required packages

### Option 2: Manual Setup

#### Step 1: Create Virtual Environment

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install base requirements
pip install -r ../requirements.txt

# Install API requirements
pip install -r api_requirements.txt
```

#### Step 3: Install PyTorch

**For CUDA 11.8 (GPU):**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

#### Step 4: Install vLLM

vLLM needs to be installed separately. See the main README.md for instructions.

For vLLM 0.8.5 with CUDA 11.8:
```bash
# Download the wheel file from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

Or use the latest version:
```bash
pip install vllm
```

#### Step 5: Install Flash Attention

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

## Running the Server

### Activate Virtual Environment

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
source venv/bin/activate
```

### Start the API Server

```bash
python3 api_server.py
```

Or with custom options:
```bash
python3 api_server.py --host 0.0.0.0 --port 8000
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Solution:**
1. Make sure virtual environment is activated: `source venv/bin/activate`
2. Install PyTorch (see Step 3 above)
3. Verify: `python3 -c "import torch; print(torch.__version__)"`

### "ModuleNotFoundError: No module named 'vllm'"

**Solution:**
1. Install vLLM (see Step 4 above)
2. Verify: `python3 -c "import vllm"`

### "Command not found: python"

**Solution:**
- Use `python3` instead of `python` on macOS
- Or create an alias: `alias python=python3`

### GPU Not Detected

**Solution:**
1. Check CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
2. Install CUDA-enabled PyTorch if needed
3. Check GPU: `nvidia-smi`

### Port Already in Use

**Solution:**
- Use a different port: `python3 api_server.py --port 8080`
- Or kill the process using port 8000

## Environment Variables

You can set these before running:

```bash
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export VLLM_USE_V1=0          # Use vLLM v0
```

## Next Steps

Once setup is complete:

1. **Start the server**: `python3 api_server.py`
2. **Open the UI**: http://localhost:8000
3. **Check health**: http://localhost:8000/health
4. **View docs**: http://localhost:8000/docs

## Notes

- Always activate the virtual environment before running
- The model will be loaded on first request (takes time)
- GPU memory usage depends on model size and concurrency
- For production, use Gunicorn + Uvicorn workers

