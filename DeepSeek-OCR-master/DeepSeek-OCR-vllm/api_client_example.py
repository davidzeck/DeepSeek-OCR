"""
Example client code for DeepSeek-OCR API
Demonstrates how to use the async API server
"""
import asyncio
import base64
import aiohttp
from pathlib import Path
from typing import Optional


class DeepSeekOCRClient:
    """Async client for DeepSeek-OCR API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> dict:
        """Check API health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def ocr_from_base64(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> dict:
        """OCR from base64 encoded image"""
        payload = {
            "image_base64": image_base64,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with self.session.post(
            f"{self.base_url}/ocr",
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def ocr_from_file(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> dict:
        """OCR from image file"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            async with self.session.post(
                f"{self.base_url}/ocr/file",
                data=data,
                files=files
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def ocr_stream(
        self,
        image_path: str,
        prompt: Optional[str] = None
    ):
        """Stream OCR results"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {"prompt": prompt}
            
            async with self.session.post(
                f"{self.base_url}/ocr/stream",
                data=data,
                files=files
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith('data: '):
                            text = decoded[6:].strip()
                            if text == '[DONE]':
                                break
                            yield text
    
    async def ocr_batch(
        self,
        image_paths: list[str],
        prompt: Optional[str] = None
    ) -> dict:
        """Batch OCR processing"""
        requests = []
        
        for img_path in image_paths:
            # Read and encode image
            with open(img_path, 'rb') as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            requests.append({
                "image_base64": image_base64,
                "prompt": prompt
            })
        
        payload = {"requests": requests}
        
        async with self.session.post(
            f"{self.base_url}/ocr/batch",
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()


# ==================== Usage Examples ====================

async def example_single_ocr():
    """Example: Single image OCR"""
    async with DeepSeekOCRClient() as client:
        # Check health
        health = await client.health_check()
        print(f"API Status: {health}")
        
        # OCR from file
        result = await client.ocr_from_file(
            image_path="path/to/your/image.jpg",
            prompt="<image>\nFree OCR."
        )
        
        print(f"Request ID: {result['request_id']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Text:\n{result['text']}")


async def example_base64_ocr():
    """Example: OCR from base64"""
    async with DeepSeekOCRClient() as client:
        # Read image and encode
        with open("path/to/your/image.jpg", 'rb') as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        result = await client.ocr_from_base64(
            image_base64=image_base64,
            prompt="<image>\n<|grounding|>Convert the document to markdown."
        )
        
        print(result['text'])


async def example_streaming_ocr():
    """Example: Streaming OCR"""
    async with DeepSeekOCRClient() as client:
        print("Streaming OCR results:")
        async for chunk in client.ocr_stream(
            image_path="path/to/your/image.jpg"
        ):
            print(chunk, end='', flush=True)
        print()  # New line at end


async def example_batch_ocr():
    """Example: Batch OCR"""
    async with DeepSeekOCRClient() as client:
        image_paths = [
            "path/to/image1.jpg",
            "path/to/image2.jpg",
            "path/to/image3.jpg"
        ]
        
        result = await client.ocr_batch(
            image_paths=image_paths,
            prompt="<image>\nFree OCR."
        )
        
        print(f"Total Processing Time: {result['total_processing_time']:.2f}s")
        for i, res in enumerate(result['results']):
            print(f"\n--- Image {i+1} ---")
            print(f"Status: {res['status']}")
            print(f"Time: {res['processing_time']:.2f}s")
            print(f"Text: {res['text'][:100]}...")


async def example_concurrent_requests():
    """Example: Multiple concurrent requests"""
    async with DeepSeekOCRClient() as client:
        # Create multiple tasks
        tasks = [
            client.ocr_from_file(f"path/to/image{i}.jpg")
            for i in range(5)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            print(f"Image {i+1}: {result['processing_time']:.2f}s")


if __name__ == "__main__":
    # Run examples
    print("=== Single OCR Example ===")
    # asyncio.run(example_single_ocr())
    
    print("\n=== Streaming OCR Example ===")
    # asyncio.run(example_streaming_ocr())
    
    print("\n=== Batch OCR Example ===")
    # asyncio.run(example_batch_ocr())
    
    print("\n=== Concurrent Requests Example ===")
    # asyncio.run(example_concurrent_requests())

