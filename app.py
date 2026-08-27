"""
app.py
------
FastAPI Web Backend Server for AI OCR Document Reader.

Endpoints:
- POST /api/analyze: Upload an image or PDF document to extract OCR & LLM JSON.
- GET  /api/health: Check server & model status.

Usage:
    uvicorn app:app --reload --port 8000
"""

import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ocr import run_ocr, format_ocr_text
from llm import analyze_document_text
from models import DocumentResult

app = FastAPI(
    title="AI OCR Document Reader API",
    description="FastAPI REST Backend for PaddleOCR + Llama 3.2 3B via Ollama",
    version="1.0.0"
)

# Enable CORS middleware so local Node.js Frontend (Vite on port 5173/3000) can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "documents", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def read_root():
    """Root endpoint providing basic API information and documentation links."""
    return {
        "status": "online",
        "service": "AI OCR Document Reader API",
        "documentation": "/docs",
        "health_check": "/api/health",
        "frontend_dev_url": "http://localhost:5173"
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "online",
        "service": "AI OCR Document Reader API",
        "default_model": "llama3.2:3b"
    }


@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Accepts an uploaded image or PDF file, runs PaddleOCR, and extracts structured JSON using Llama 3.2 3B.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    # Save uploaded file to disk temporarily
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {err}")

    print(f"[API] Received uploaded file: '{filename}'")

    # Step 1: Execute PaddleOCR
    try:
        ocr_result = run_ocr(file_path)
    except Exception as ocr_err:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PaddleOCR processing error: {ocr_err}")

    # Step 2: Format reading-order text
    formatted_text = format_ocr_text(ocr_result)

    # Step 3: Analyze text with LLM (Llama 3.2 3B via Ollama)
    try:
        document_result: DocumentResult = analyze_document_text(
            ocr_text=formatted_text,
            model_name="llama3.2:3b"
        )
    except ConnectionError as conn_err:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Ollama server is offline or unreachable: {conn_err}"
        )
    except Exception as llm_err:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"LLM processing error: {llm_err}"
        )

    # Return structured response containing raw OCR lines, formatted text, and final Pydantic result
    return {
        "filename": filename,
        "ocr_items_count": len(ocr_result.items),
        "ocr_items": [item.model_dump() for item in ocr_result.items],
        "raw_text": formatted_text,
        "document_result": document_result.model_dump()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
