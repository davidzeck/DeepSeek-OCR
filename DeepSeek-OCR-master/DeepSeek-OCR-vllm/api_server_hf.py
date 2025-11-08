"""
FastAPI Server for DeepSeek-OCR using HuggingFace Transformers
Works on macOS (CPU or Apple Silicon) - No vLLM required
"""
import os
import time
import uuid
import base64
import io
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
import uvicorn

# HuggingFace imports
from transformers import AutoModel, AutoTokenizer

# Global model instance
model: Optional[AutoModel] = None
tokenizer: Optional[AutoTokenizer] = None

# ==================== Pydantic Models ====================

class OCRRequest(BaseModel):
    """Request model for OCR endpoint"""
    prompt: Optional[str] = Field(
        default="<image>\n<|grounding|>Convert the document to markdown.",
        description="Prompt for OCR task"
    )
    image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded image"
    )
    base_size: int = Field(
        default=1024,
        ge=512,
        le=1280,
        description="Base size for global view"
    )
    image_size: int = Field(
        default=640,
        ge=512,
        le=1280,
        description="Image size for local crops"
    )
    crop_mode: bool = Field(
        default=True,
        description="Enable dynamic cropping"
    )


class OCRResponse(BaseModel):
    """Response model for OCR endpoint"""
    request_id: str = Field(description="Unique request identifier")
    text: str = Field(description="OCR output text")
    processing_time: float = Field(description="Processing time in seconds")
    status: str = Field(default="success", description="Request status")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: float


# ==================== Utility Functions ====================

def load_image_from_base64(image_base64: str) -> Image.Image:
    """Load PIL Image from base64 string"""
    try:
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        image = ImageOps.exif_transpose(image)
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 image: {str(e)}"
        )


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load PIL Image from bytes"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )


# ==================== Model Management ====================

async def initialize_model():
    """Initialize the HuggingFace model (called at startup)"""
    global model, tokenizer
    
    if model is not None:
        return
    
    print("Initializing DeepSeek-OCR model (HuggingFace)...")
    start_time = time.time()
    
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    try:
        # Load tokenizer
        print("Loading tokenizer (using slow tokenizer for compatibility)...")
        # Force slow tokenizer to avoid fast tokenizer parsing errors on Intel Mac
        # Delete tokenizer.json if it exists to force slow tokenizer
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        if os.path.exists(model_cache):
            for root, dirs, files in os.walk(model_cache):
                if "tokenizer.json" in files:
                    tokenizer_json_path = os.path.join(root, "tokenizer.json")
                    try:
                        os.remove(tokenizer_json_path)
                        print(f"Removed {tokenizer_json_path} to force slow tokenizer")
                    except:
                        pass
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,  # Force slow tokenizer
            local_files_only=False
        )
        
        # Load model
        print("Loading model (this may take a while)...")
        
        # FORCE CPU MODE for Intel Mac compatibility
        # Disable MPS detection to avoid FlashAttention issues
        device = "cpu"
        print("Using CPU (forced for Intel Mac compatibility)")
        
        # Load model WITHOUT flash_attention_2 to avoid import errors
        # Use 'eager' attention (CPU-compatible, no FlashAttention)
        print("Loading with eager attention (CPU-compatible, no FlashAttention2)...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            # Use 'eager' attention - fully CPU compatible, no FlashAttention dependencies
            _attn_implementation='eager'  # Eager attention works on CPU without any special dependencies
        )
        
        model = model.eval()
        
        # Move to CPU with float32 (best compatibility for Intel Mac)
        model = model.to(device).to(torch.float32)
        print(f"Model loaded on {device} with float32 precision")
        
        elapsed = time.time() - start_time
        print(f"Model initialized in {elapsed:.2f} seconds")
        print(f"Device: {device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


async def shutdown_model():
    """Shutdown the model (called at shutdown)"""
    global model, tokenizer
    if model is not None:
        del model
        model = None
        tokenizer = None
        print("Model shutdown complete")


# ==================== Lifespan Context Manager ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    await initialize_model()
    yield
    # Shutdown
    await shutdown_model()


# ==================== FastAPI App ====================

app = FastAPI(
    title="DeepSeek-OCR API (HuggingFace)",
    description="Fast async API for DeepSeek-OCR using HuggingFace Transformers (macOS compatible)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API Endpoints ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - redirects to UI"""
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {
        "message": "DeepSeek-OCR API Server (HuggingFace)",
        "version": "1.0.0",
        "docs": "/docs",
        "ui": "/ui"
    }

@app.get("/ui", tags=["General"])
async def serve_ui():
    """Serve the web UI"""
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    raise HTTPException(status_code=404, detail="UI not found")

# Serve static files if ui directory exists
ui_dir = os.path.join(os.path.dirname(__file__), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui/static", StaticFiles(directory=ui_dir), name="ui-static")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device,
        timestamp=time.time()
    )


@app.post("/ocr", response_model=OCRResponse, tags=["OCR"])
async def ocr_from_base64(request: OCRRequest):
    """
    OCR endpoint using base64 encoded image (HuggingFace version)
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Load and process image
        image = load_image_from_base64(request.image_base64)
        
        # Save to temporary file (model.infer expects file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file.name, 'JPEG')
            tmp_path = tmp_file.name
        
        try:
            # Call model inference
            result = model.infer(
                tokenizer,
                prompt=request.prompt or "<image>\nFree OCR.",
                image_file=tmp_path,
                output_path=None,  # Don't save results
                base_size=request.base_size,
                image_size=request.image_size,
                crop_mode=request.crop_mode,
                save_results=False,
                test_compress=False
            )
            
            processing_time = time.time() - start_time
            
            return OCRResponse(
                request_id=request_id,
                text=result if isinstance(result, str) else str(result),
                processing_time=processing_time,
                status="success"
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/ocr/file", response_model=OCRResponse, tags=["OCR"])
async def ocr_from_file(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True
):
    """
    OCR endpoint using file upload (HuggingFace version)
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file.name, 'JPEG')
            tmp_path = tmp_file.name
        
        try:
            # Call model inference
            result = model.infer(
                tokenizer,
                prompt=prompt or "<image>\nFree OCR.",
                image_file=tmp_path,
                output_path=None,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
                test_compress=False
            )
            
            processing_time = time.time() - start_time
            
            return OCRResponse(
                request_id=request_id,
                text=result if isinstance(result, str) else str(result),
                processing_time=processing_time,
                status="success"
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server_hf:app",
        host=args.host,
        port=args.port,
        workers=1,  # Single worker for model
        reload=args.reload,
        log_level="info"
    )

