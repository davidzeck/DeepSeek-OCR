# 🎨 UI Quick Start Guide

## Start Everything in 3 Steps

### Step 1: Start the API Server

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python api_server.py
```

Wait for: `"Engine initialized in X seconds"`

### Step 2: Open the UI

**Option A: Direct from API Server (Recommended)**
- Open browser: http://localhost:8000
- The UI is automatically served!

**Option B: Standalone UI Server**
```bash
./start_ui.sh
```
- Open browser: http://localhost:8080

**Option C: Direct File**
- Open `ui/index.html` in your browser

### Step 3: Use the UI

1. ✅ Check connection status (should be green)
2. 📤 Upload an image (drag & drop or click)
3. 📝 Select a prompt (or use custom)
4. 🚀 Click "Process OCR" or "Stream OCR"
5. 📄 View results!

## Features

### 🎯 Quick Prompts
- **Free OCR**: Basic text extraction
- **Markdown**: Convert to markdown format
- **Describe**: Detailed description
- **Parse Figure**: Extract figure data

### ⚙️ Configuration
- **Temperature**: 0.0 (deterministic) to 2.0 (creative)
- **Max Tokens**: 1-8192 (output length)
- **Crop Mode**: Enable/disable dynamic cropping

### 📡 Processing Modes
- **Standard**: Complete processing, shows time
- **Streaming**: Real-time token generation

## Screenshots Guide

### Main Interface
- Top: Connection status indicator
- Left: Configuration and prompts
- Center: Image upload area
- Bottom: Results display

### Connection Status
- 🟢 Green = Connected & Ready
- 🔴 Red = Connection Failed

## Troubleshooting

### UI Not Loading
- Check API server is running
- Verify port 8000 is accessible
- Check browser console for errors

### Connection Failed
- Ensure API server started successfully
- Check API URL matches server (default: http://localhost:8000)
- Try `/health` endpoint: http://localhost:8000/health

### CORS Errors
- Use Option A (served from API server) - no CORS issues
- Or configure CORS in `api_server.py`

### Image Not Processing
- Check image format (JPEG, PNG)
- Verify image isn't corrupted
- Check browser console for errors

## Tips

1. **For Best Performance**:
   - Use streaming mode for long documents
   - Reduce max_tokens if you don't need full output
   - Enable crop mode for large images

2. **For Accuracy**:
   - Use appropriate prompt for your task
   - Markdown prompt for structured documents
   - Free OCR for simple text extraction

3. **For Speed**:
   - Lower max_tokens
   - Disable crop mode for small images
   - Use standard mode (faster than streaming)

## Next Steps

- Try different prompts
- Experiment with temperature settings
- Process multiple images
- Check API documentation at http://localhost:8000/docs

Enjoy using DeepSeek-OCR! 🚀

