"""
FastAPI Async Server for DeepSeek-OCR
Production-ready API server with best practices
"""
import os
import time
import uuid
import base64
import io
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
import uvicorn

# vLLM imports
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

# DeepSeek-OCR imports
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import (
    MODEL_PATH, PROMPT, CROP_MODE, BASE_SIZE, IMAGE_SIZE,
    MAX_CONCURRENCY
)

# Environment setup
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

# Register model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Global engine instance
llm_engine: Optional[AsyncLLMEngine] = None
processor: Optional[DeepseekOCRProcessor] = None


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
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=8192,
        ge=1,
        le=8192,
        description="Maximum tokens to generate"
    )
    ngram_size: int = Field(
        default=30,
        ge=1,
        le=100,
        description="N-gram size for repetition prevention"
    )
    window_size: int = Field(
        default=90,
        ge=1,
        le=200,
        description="Window size for n-gram checking"
    )
    crop_mode: Optional[bool] = Field(
        None,
        description="Enable dynamic cropping (overrides config)"
    )
    base_size: Optional[int] = Field(
        None,
        description="Base size for global view (overrides config)"
    )
    image_size: Optional[int] = Field(
        None,
        description="Image size for local crops (overrides config)"
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
    gpu_available: bool
    timestamp: float


class BatchOCRRequest(BaseModel):
    """Batch OCR request model"""
    requests: List[OCRRequest] = Field(
        ...,
        max_items=100,
        description="List of OCR requests (max 100)"
    )


class BatchOCRResponse(BaseModel):
    """Batch OCR response model"""
    results: List[OCRResponse]
    total_processing_time: float


# ==================== Utility Functions ====================

def load_image_from_base64(image_base64: str) -> Image.Image:
    """Load PIL Image from base64 string"""
    try:
        # Remove data URL prefix if present
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


# ==================== Engine Management ====================

async def initialize_engine():
    """Initialize the AsyncLLMEngine (called at startup)"""
    global llm_engine, processor
    
    if llm_engine is not None:
        return
    
    print("Initializing DeepSeek-OCR engine...")
    start_time = time.time()
    
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        max_num_seqs=MAX_CONCURRENCY,
    )
    
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    processor = DeepseekOCRProcessor()
    
    elapsed = time.time() - start_time
    print(f"Engine initialized in {elapsed:.2f} seconds")


async def shutdown_engine():
    """Shutdown the engine (called at shutdown)"""
    global llm_engine
    if llm_engine is not None:
        # vLLM handles cleanup automatically
        llm_engine = None
        print("Engine shutdown complete")


# ==================== Lifespan Context Manager ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    await initialize_engine()
    yield
    # Shutdown
    await shutdown_engine()


# ==================== FastAPI App ====================

app = FastAPI(
    title="DeepSeek-OCR API",
    description="Fast async API for DeepSeek-OCR document understanding",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
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
        "message": "DeepSeek-OCR API Server",
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
    return HealthResponse(
        status="healthy" if llm_engine is not None else "unhealthy",
        model_loaded=llm_engine is not None,
        gpu_available=torch.cuda.is_available(),
        timestamp=time.time()
    )


@app.post("/ocr", response_model=OCRResponse, tags=["OCR"])
async def ocr_from_base64(request: OCRRequest):
    """
    OCR endpoint using base64 encoded image
    
    - **prompt**: Custom prompt (optional)
    - **image_base64**: Base64 encoded image (required)
    - **temperature**: Sampling temperature (0.0-2.0)
    - **max_tokens**: Maximum tokens to generate
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Load and process image
        image = load_image_from_base64(request.image_base64)
        
        # Determine crop mode
        use_crop_mode = request.crop_mode if request.crop_mode is not None else CROP_MODE
        
        # Process image
        image_features = processor.tokenize_with_images(
            images=[image],
            bos=True,
            eos=True,
            cropping=use_crop_mode
        )
        
        # Prepare prompt
        prompt = request.prompt or PROMPT
        
        # Create logits processor
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=request.ngram_size,
                window_size=request.window_size,
                whitelist_token_ids={128821, 128822}  # <td>, </td>
            )
        ]
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )
        
        # Generate
        request_data = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }
        
        final_output = ""
        async for request_output in llm_engine.generate(
            request_data, sampling_params, request_id
        ):
            if request_output.outputs:
                final_output = request_output.outputs[0].text
        
        processing_time = time.time() - start_time
        
        return OCRResponse(
            request_id=request_id,
            text=final_output,
            processing_time=processing_time,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/ocr/file", response_model=OCRResponse, tags=["OCR"])
async def ocr_from_file(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    ngram_size: int = 30,
    window_size: int = 90,
    crop_mode: Optional[bool] = None
):
    """
    OCR endpoint using file upload
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **prompt**: Custom prompt (optional)
    - **temperature**: Sampling temperature
    - **max_tokens**: Maximum tokens to generate
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)
        
        # Determine crop mode
        use_crop_mode = crop_mode if crop_mode is not None else CROP_MODE
        
        # Process image
        image_features = processor.tokenize_with_images(
            images=[image],
            bos=True,
            eos=True,
            cropping=use_crop_mode
        )
        
        # Prepare prompt
        ocr_prompt = prompt or PROMPT
        
        # Create logits processor
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids={128821, 128822}
            )
        ]
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )
        
        # Generate
        request_data = {
            "prompt": ocr_prompt,
            "multi_modal_data": {"image": image_features}
        }
        
        final_output = ""
        async for request_output in llm_engine.generate(
            request_data, sampling_params, request_id
        ):
            if request_output.outputs:
                final_output = request_output.outputs[0].text
        
        processing_time = time.time() - start_time
        
        return OCRResponse(
            request_id=request_id,
            text=final_output,
            processing_time=processing_time,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/ocr/stream", tags=["OCR"])
async def ocr_stream(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192
):
    """
    Streaming OCR endpoint (Server-Sent Events)
    
    Returns text as it's generated
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_id = str(uuid.uuid4())
    
    async def generate_stream():
        try:
            # Read and process image
            image_bytes = await file.read()
            image = load_image_from_bytes(image_bytes)
            
            # Process image
            image_features = processor.tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )
            
            # Prepare prompt
            ocr_prompt = prompt or PROMPT
            
            # Create logits processor
            logits_processors = [
                NoRepeatNGramLogitsProcessor(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822}
                )
            ]
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                logits_processors=logits_processors,
                skip_special_tokens=False,
            )
            
            # Generate
            request_data = {
                "prompt": ocr_prompt,
                "multi_modal_data": {"image": image_features}
            }
            
            printed_length = 0
            async for request_output in llm_engine.generate(
                request_data, sampling_params, request_id
            ):
                if request_output.outputs:
                    full_text = request_output.outputs[0].text
                    new_text = full_text[printed_length:]
                    if new_text:
                        yield f"data: {new_text}\n\n"
                        printed_length = len(full_text)
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )


@app.post("/ocr/batch", response_model=BatchOCRResponse, tags=["OCR"])
async def ocr_batch(request: BatchOCRRequest):
    """
    Batch OCR endpoint
    
    Process multiple images in parallel (up to 100)
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 requests per batch"
        )
    
    start_time = time.time()
    results = []
    
    # Process requests concurrently
    import asyncio
    
    async def process_single(req: OCRRequest, req_id: str):
        try:
            if not req.image_base64:
                return OCRResponse(
                    request_id=req_id,
                    text="",
                    processing_time=0.0,
                    status="error: no image provided"
                )
            
            req_start = time.time()
            image = load_image_from_base64(req.image_base64)
            
            use_crop_mode = req.crop_mode if req.crop_mode is not None else CROP_MODE
            
            image_features = processor.tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=use_crop_mode
            )
            
            prompt = req.prompt or PROMPT
            
            logits_processors = [
                NoRepeatNGramLogitsProcessor(
                    ngram_size=req.ngram_size,
                    window_size=req.window_size,
                    whitelist_token_ids={128821, 128822}
                )
            ]
            
            sampling_params = SamplingParams(
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                logits_processors=logits_processors,
                skip_special_tokens=False,
            )
            
            request_data = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features}
            }
            
            final_output = ""
            async for request_output in llm_engine.generate(
                request_data, sampling_params, req_id
            ):
                if request_output.outputs:
                    final_output = request_output.outputs[0].text
            
            processing_time = time.time() - req_start
            
            return OCRResponse(
                request_id=req_id,
                text=final_output,
                processing_time=processing_time,
                status="success"
            )
        except Exception as e:
            return OCRResponse(
                request_id=req_id,
                text="",
                processing_time=0.0,
                status=f"error: {str(e)}"
            )
    
    # Create tasks for all requests
    tasks = [
        process_single(req, str(uuid.uuid4()))
        for req in request.requests
    ]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    return BatchOCRResponse(
        results=results,
        total_processing_time=total_time
    )


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )

