# DeepSeek-OCR Web UI

A modern, user-friendly web interface for the DeepSeek-OCR API.

## Features

- 🎨 **Modern Design**: Beautiful gradient UI with smooth animations
- 📤 **Easy Upload**: Drag & drop or click to upload images
- ⚡ **Real-time Processing**: See results as they're generated (streaming mode)
- 📝 **Prompt Presets**: Quick access to common prompts
- ⚙️ **Configurable**: Adjust temperature, max tokens, crop mode
- 🔄 **Health Monitoring**: Real-time API connection status
- 📱 **Responsive**: Works on desktop and mobile devices

## Usage

### Option 1: Direct File Access

Simply open `index.html` in your web browser:

```bash
# On macOS
open ui/index.html

# On Linux
xdg-open ui/index.html

# On Windows
start ui/index.html
```

### Option 2: Serve with Python

For better CORS handling, serve the UI with a simple HTTP server:

```bash
cd ui
python3 -m http.server 8080
```

Then open: http://localhost:8080

### Option 3: Integrate with FastAPI

Add this to your `api_server.py` to serve the UI:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add at the end of api_server.py, before if __name__ == "__main__"
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/ui")
async def serve_ui():
    return FileResponse("ui/index.html")
```

Then access: http://localhost:8000/ui

## How to Use

1. **Start the API Server**:
   ```bash
   python api_server.py
   ```

2. **Open the UI**:
   - Direct: Open `index.html` in browser
   - Or: Navigate to http://localhost:8080 (if using Python server)

3. **Configure Settings**:
   - Set API URL (default: http://localhost:8000)
   - Adjust temperature, max tokens if needed
   - Select crop mode

4. **Upload Image**:
   - Click the upload area or drag & drop an image
   - Supported formats: JPEG, PNG

5. **Select Prompt**:
   - Use quick prompt buttons or enter custom prompt
   - Common prompts:
     - **Free OCR**: Basic OCR without layout
     - **Markdown**: Convert document to markdown
     - **Describe**: Detailed image description
     - **Parse Figure**: Extract figure information

6. **Process**:
   - Click **Process OCR** for standard processing
   - Click **Stream OCR** for real-time streaming results

7. **View Results**:
   - Results appear in the results section
   - Processing time and metadata are shown
   - Copy results as needed

## Features Explained

### Connection Status
- Green indicator: API is connected and model is loaded
- Red indicator: Connection failed or model not loaded
- Auto-checks every 30 seconds

### Prompt Presets
Click any preset button to quickly set common prompts:
- **Free OCR**: `"<image>\nFree OCR."`
- **Markdown**: `"<image>\n<|grounding|>Convert the document to markdown."`
- **Describe**: `"<image>\nDescribe this image in detail."`
- **Parse Figure**: `"<image>\nParse the figure."`

### Processing Modes

**Standard Processing**:
- Processes entire image
- Returns complete result
- Shows processing time

**Streaming Mode**:
- Real-time token generation
- See results as they're created
- Better for long documents
- Can be cancelled

### Configuration Options

- **Temperature**: Controls randomness (0.0 = deterministic, higher = more creative)
- **Max Tokens**: Maximum output length (1-8192)
- **Crop Mode**: Enable/disable dynamic cropping for large images

## Troubleshooting

### CORS Errors
If you see CORS errors when opening the file directly:
- Use Option 2 (Python HTTP server) or Option 3 (FastAPI integration)
- Or configure CORS in `api_server.py` to allow your origin

### Connection Failed
- Ensure API server is running: `python api_server.py`
- Check API URL matches server address
- Verify server is accessible (try `/health` endpoint)

### Image Not Processing
- Check image format (JPEG, PNG supported)
- Verify image size (very large images may take time)
- Check browser console for errors

### Slow Processing
- Reduce `max_tokens` if not needed
- Use streaming mode for better feedback
- Check GPU utilization on server

## Customization

### Change Colors
Edit the CSS in `index.html`:
- Gradient: `background: linear-gradient(...)`
- Primary color: `#667eea` and `#764ba2`
- Adjust colors throughout the file

### Add More Prompts
Add to the prompt presets section:
```html
<div class="prompt-btn" data-prompt="<image>\nYour custom prompt">Custom</div>
```

### Modify Layout
The UI uses CSS Grid and Flexbox for responsive layout. Adjust grid columns in the `.grid` class.

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Mobile browsers

## Security Notes

- The UI runs entirely client-side
- No data is stored locally
- Images are sent directly to API
- For production, add authentication to API

## License

Same as DeepSeek-OCR project.

