# 🚀 macOS Quick Start Guide

## ✅ What's Ready

You now have a **HuggingFace-based API server** that works on macOS!

- ✅ PyTorch installed (CPU/MPS compatible)
- ✅ Transformers installed
- ✅ FastAPI and dependencies installed
- ✅ NumPy fixed (compatible version)
- ✅ API server created (`api_server_hf.py`)

## 🎯 Quick Start (3 Steps)

### Step 1: Activate Virtual Environment

```bash
cd /Users/macbook/Deepseek-Ocr/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm
source venv/bin/activate
```

### Step 2: Start the API Server

```bash
python3 api_server_hf.py
```

**First run will:**
- Download the DeepSeek-OCR model from HuggingFace (this takes time, ~several GB)
- Load the model into memory
- Start the API server on http://localhost:8000

### Step 3: Open the UI

Once you see "Model initialized", open:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📝 Usage

### Using the Web UI

1. Open http://localhost:8000 in your browser
2. Upload an image (drag & drop or click)
3. Select a prompt or enter custom
4. Click "Process OCR"
5. View results!

### Using the API

```bash
# Health check
curl http://localhost:8000/health

# OCR from file
curl -X POST "http://localhost:8000/ocr/file" \
  -F "file=@your_image.jpg" \
  -F "prompt=<image>\nFree OCR."
```

## ⚙️ Configuration

The server supports these parameters:

- `base_size`: Global view resolution (512-1280, default: 1024)
- `image_size`: Local crop resolution (512-1280, default: 640)
- `crop_mode`: Enable dynamic cropping (true/false, default: true)
- `prompt`: Custom prompt text

## 🎨 Device Support

The server automatically detects and uses:

1. **Apple Silicon (MPS)**: If you have M1/M2/M3 Mac
   - Uses Metal Performance Shaders
   - Faster than CPU

2. **CPU**: Fallback for Intel Macs
   - Works but slower
   - Still functional

3. **CUDA**: If you have NVIDIA GPU (unlikely on Mac)
   - Automatically detected if available

## 📊 Performance Tips

### For Apple Silicon (M1/M2/M3):
- Model will use MPS (Metal)
- Should be reasonably fast
- First inference may be slower (warmup)

### For Intel Mac:
- Uses CPU (slower)
- Be patient with first request
- Consider using smaller `base_size` and `image_size`

## 🔧 Troubleshooting

### "Model not loaded"
- Wait for model to download (first time only)
- Check internet connection
- Check available disk space

### "Out of memory"
- Reduce `base_size` and `image_size`
- Close other applications
- Restart the server

### Slow performance
- First request is always slower (model warmup)
- Use smaller image sizes
- Consider using Apple Silicon Mac for better performance

### Port already in use
```bash
python3 api_server_hf.py --port 8080
```

## 📦 What's Different from vLLM Version?

| Feature | vLLM Version | HuggingFace Version |
|---------|-------------|---------------------|
| Platform | Linux + CUDA | macOS + CPU/MPS |
| Speed | Very Fast | Moderate |
| GPU Required | Yes (NVIDIA) | No |
| Batch Processing | Excellent | Limited |
| Streaming | Yes | No (yet) |

## 🎯 Next Steps

1. **Test the API**: Try uploading an image
2. **Try different prompts**: Experiment with various OCR tasks
3. **Check performance**: See how fast it runs on your Mac
4. **Use the UI**: Enjoy the web interface!

## 💡 Tips

- First model download takes time (~5-10 minutes depending on connection)
- Model is cached after first download
- Use the web UI for easiest experience
- API is fully async and can handle multiple requests

Enjoy using DeepSeek-OCR on macOS! 🎉

