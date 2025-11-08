# DeepSeek-OCR FastAPI Server

Production-ready async API server for DeepSeek-OCR document understanding.

## Features

- ✅ **Fully Async**: Built on FastAPI with async/await throughout
- ✅ **Multiple Input Methods**: Base64, file upload, batch processing
- ✅ **Streaming Support**: Server-Sent Events for real-time output
- ✅ **Production Ready**: Health checks, error handling, CORS support
- ✅ **Type Safe**: Pydantic models for request/response validation
- ✅ **Concurrent Processing**: Handles multiple requests efficiently
- ✅ **Auto Documentation**: Swagger UI at `/docs`

## Installation

1. Install API dependencies:
```bash
pip install -r api_requirements.txt
```

2. Ensure all DeepSeek-OCR dependencies are installed (from main requirements.txt)

## Running the Server

### Basic Usage

```bash
python api_server.py
```

### With Custom Options

```bash
python api_server.py --host 0.0.0.0 --port 8000 --workers 1
```

### Using Uvicorn Directly

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

### Production Deployment (Gunicorn + Uvicorn)

```bash
gunicorn api_server:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
```

**Note**: For vLLM, use `--workers 1` as the model should be loaded once per process.

## API Endpoints

### Health Check

```bash
GET /health
```

Returns server and model status.

### OCR from Base64

```bash
POST /ocr
Content-Type: application/json

{
  "image_base64": "base64_encoded_image",
  "prompt": "<image>\nFree OCR.",
  "temperature": 0.0,
  "max_tokens": 8192
}
```

### OCR from File Upload

```bash
POST /ocr/file
Content-Type: multipart/form-data

file: <image_file>
prompt: "<image>\nFree OCR."
temperature: 0.0
max_tokens: 8192
```

### Streaming OCR

```bash
POST /ocr/stream
Content-Type: multipart/form-data

file: <image_file>
prompt: "<image>\nFree OCR."
```

Returns Server-Sent Events (SSE) stream.

### Batch OCR

```bash
POST /ocr/batch
Content-Type: application/json

{
  "requests": [
    {
      "image_base64": "base64_image_1",
      "prompt": "<image>\nFree OCR."
    },
    {
      "image_base64": "base64_image_2",
      "prompt": "<image>\nFree OCR."
    }
  ]
}
```

Processes up to 100 images concurrently.

## Usage Examples

### Python Client

See `api_client_example.py` for complete examples.

```python
import asyncio
from api_client_example import DeepSeekOCRClient

async def main():
    async with DeepSeekOCRClient() as client:
        result = await client.ocr_from_file(
            image_path="document.jpg",
            prompt="<image>\n<|grounding|>Convert the document to markdown."
        )
        print(result['text'])

asyncio.run(main())
```

### cURL Examples

#### Health Check
```bash
curl http://localhost:8000/health
```

#### OCR from File
```bash
curl -X POST "http://localhost:8000/ocr/file" \
  -F "file=@document.jpg" \
  -F "prompt=<image>\nFree OCR."
```

#### OCR from Base64
```bash
# Encode image to base64
IMAGE_B64=$(base64 -i document.jpg)

curl -X POST "http://localhost:8000/ocr" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$IMAGE_B64\",
    \"prompt\": \"<image>\\nFree OCR.\"
  }"
```

#### Streaming OCR
```bash
curl -X POST "http://localhost:8000/ocr/stream" \
  -F "file=@document.jpg" \
  -F "prompt=<image>\nFree OCR." \
  --no-buffer
```

### JavaScript/TypeScript Client

```javascript
// Using fetch API
async function ocrFromFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('prompt', '<image>\nFree OCR.');
  
  const response = await fetch('http://localhost:8000/ocr/file', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result.text;
}

// Streaming example
async function ocrStream(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/ocr/stream', {
    method: 'POST',
    body: formData
  });
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const text = line.slice(6);
        if (text === '[DONE]') return;
        console.log(text);
      }
    }
  }
}
```

## Configuration

The API uses settings from `config.py`:

- `MODEL_PATH`: Path to model
- `PROMPT`: Default prompt
- `CROP_MODE`: Enable dynamic cropping
- `BASE_SIZE`: Global view resolution
- `IMAGE_SIZE`: Local crop resolution
- `MAX_CONCURRENCY`: Max concurrent requests

You can override these per-request using the API parameters.

## Prompts

Common prompt templates:

- **Free OCR**: `"<image>\nFree OCR."`
- **Markdown Conversion**: `"<image>\n<|grounding|>Convert the document to markdown."`
- **General Description**: `"<image>\nDescribe this image in detail."`
- **Figure Parsing**: `"<image>\nParse the figure."`
- **Object Detection**: `"<image>\nLocate <|ref|>object<|/ref|> in the image."`

## Performance Tips

1. **Batch Processing**: Use `/ocr/batch` for multiple images
2. **Concurrent Requests**: API handles multiple requests automatically
3. **Streaming**: Use `/ocr/stream` for real-time feedback
4. **GPU Memory**: Adjust `gpu_memory_utilization` in `api_server.py` if needed
5. **Max Concurrency**: Tune `MAX_CONCURRENCY` based on your GPU

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid image, missing parameters)
- `503`: Service unavailable (model not loaded)
- `500`: Internal server error

## API Documentation

Interactive API documentation available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Security Considerations

For production deployment:

1. **CORS**: Configure `allow_origins` in `api_server.py`
2. **Authentication**: Add JWT or API key authentication
3. **Rate Limiting**: Implement rate limiting middleware
4. **HTTPS**: Use reverse proxy (nginx) with SSL
5. **Input Validation**: Already handled by Pydantic models
6. **File Size Limits**: Add file size restrictions

## Monitoring

- Health endpoint: `/health`
- Logs: Check uvicorn/gunicorn logs
- Metrics: Consider adding Prometheus metrics

## Troubleshooting

### Model Not Loading
- Check GPU availability: `torch.cuda.is_available()`
- Verify model path in `config.py`
- Check GPU memory

### Slow Performance
- Reduce `max_tokens` if not needed
- Adjust `MAX_CONCURRENCY` based on GPU
- Use batch endpoint for multiple images

### Memory Issues
- Reduce `gpu_memory_utilization`
- Lower `MAX_CONCURRENCY`
- Process images in smaller batches

## License

Same as DeepSeek-OCR project.

