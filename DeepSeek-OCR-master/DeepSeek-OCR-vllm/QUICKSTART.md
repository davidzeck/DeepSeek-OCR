# Quick Start Guide - DeepSeek-OCR API

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
# Install API dependencies
pip install -r api_requirements.txt

# Ensure all DeepSeek-OCR dependencies are installed
pip install -r ../requirements.txt
```

### 2. Configure Settings

Edit `config.py` if needed:
- `MODEL_PATH`: Path to your model
- `INPUT_PATH` / `OUTPUT_PATH`: Not needed for API
- `PROMPT`: Default prompt (can be overridden per request)

### 3. Start the Server

```bash
# Simple start
python api_server.py

# Or use the startup script
./start_api.sh

# Or with custom port
python api_server.py --port 8080
```

### 4. Test the API

Open your browser:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

Or use curl:
```bash
curl http://localhost:8000/health
```

## 📝 Basic Usage Examples

### Python Client

```python
import asyncio
from api_client_example import DeepSeekOCRClient

async def main():
    async with DeepSeekOCRClient() as client:
        # OCR from file
        result = await client.ocr_from_file(
            image_path="document.jpg",
            prompt="<image>\nFree OCR."
        )
        print(result['text'])

asyncio.run(main())
```

### cURL - File Upload

```bash
curl -X POST "http://localhost:8000/ocr/file" \
  -F "file=@document.jpg" \
  -F "prompt=<image>\nFree OCR."
```

### cURL - Base64

```bash
# Encode image
IMAGE_B64=$(base64 -i document.jpg)

curl -X POST "http://localhost:8000/ocr" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$IMAGE_B64\",
    \"prompt\": \"<image>\\nFree OCR.\"
  }"
```

## 🎯 Common Prompts

- **Free OCR**: `"<image>\nFree OCR."`
- **Markdown**: `"<image>\n<|grounding|>Convert the document to markdown."`
- **Describe**: `"<image>\nDescribe this image in detail."`

## 🔧 Troubleshooting

### Port Already in Use
```bash
python api_server.py --port 8080
```

### Model Not Loading
- Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify model path in `config.py`
- Check GPU memory

### Slow Performance
- Reduce `max_tokens` parameter
- Use batch endpoint for multiple images
- Check GPU utilization

## 📚 Full Documentation

See `API_README.md` for complete documentation.

## 🚀 Production Deployment

For production, use Gunicorn:

```bash
gunicorn api_server:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
```

**Important**: Use `--workers 1` for vLLM (model loaded once per process).

## 🔒 Security (Production)

1. Add authentication (see `middleware.py`)
2. Configure CORS in `api_server.py`
3. Use HTTPS (nginx reverse proxy)
4. Add rate limiting (see `middleware.py`)

