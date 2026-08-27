"""
app.py
------
FastAPI Web Backend Server for AI OCR Document Reader.

Smart Dual-Mode Operation:
- Cloud Mode (GROQ_API_KEY set): Uses Groq Llama 3.2 Vision for end-to-end
  OCR + JSON extraction in a single step. No PaddleOCR needed!
- Local Mode (No GROQ_API_KEY): Uses PaddleOCR + local Ollama Llama 3.2 3B.

Endpoints:
- POST /api/analyze  - Upload an image or PDF document to extract structured JSON
- GET  /api/health   - Check server & mode status

Usage:
    uvicorn app:app --reload --port 8000
"""

import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI OCR Document Reader API",
    description="FastAPI Backend with dual-mode: Groq Vision (cloud) or PaddleOCR + Ollama (local)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detect mode at startup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CLOUD_MODE = bool(GROQ_API_KEY)

# Vercel's filesystem is read-only; only /tmp is writable.
UPLOAD_DIR = "/tmp/ocr_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if CLOUD_MODE:
    print("[App] ☁️  Cloud Mode: Groq Llama 3.2 Vision API active")
else:
    print("[App] 💻 Local Mode: PaddleOCR + Ollama active")


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend status and active mode."""
    return {
        "status": "online",
        "mode": "cloud (Groq Vision)" if CLOUD_MODE else "local (PaddleOCR + Ollama)",
        "service": "AI OCR Document Reader API v2.0",
    }


@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Accepts an uploaded image or PDF file and extracts structured JSON.

    Cloud Mode: Sends image directly to Groq Llama 3.2 Vision API (no PaddleOCR).
    Local Mode: Runs PaddleOCR locally, then passes text to Ollama Llama 3.2 3B.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    # Save uploaded file temporarily to disk
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {err}")

    print(f"[API] Received file: '{filename}' | Mode: {'Cloud' if CLOUD_MODE else 'Local'}")

    # =========================================================================
    # CLOUD MODE: Groq Llama 3.2 Vision (Vercel deployment)
    # =========================================================================
    if CLOUD_MODE:
        try:
            from vision_llm import analyze_document_with_vision
            result = analyze_document_with_vision(file_path)
        except Exception as err:
            raise HTTPException(
                status_code=500,
                detail=f"Groq Vision API error: {err}"
            )

        return {
            "filename": filename,
            "mode": "cloud_vision",
            "ocr_items_count": None,  # Not applicable in Vision mode
            "ocr_items": [],
            "raw_text": result.get("raw_text", ""),
            "document_result": result.get("document_result", {})
        }

    # =========================================================================
    # LOCAL MODE: PaddleOCR + Ollama (Local development)
    # =========================================================================
    else:
        from ocr import run_ocr, format_ocr_text
        from llm import analyze_document_text

        try:
            ocr_result = run_ocr(file_path)
        except Exception as ocr_err:
            raise HTTPException(status_code=500, detail=f"PaddleOCR error: {ocr_err}")

        formatted_text = format_ocr_text(ocr_result)

        try:
            from models import DocumentResult
            document_result: DocumentResult = analyze_document_text(
                ocr_text=formatted_text,
                model_name="llama3.2:3b"
            )
        except ConnectionError as conn_err:
            raise HTTPException(status_code=503, detail=f"Ollama offline: {conn_err}")
        except Exception as llm_err:
            raise HTTPException(status_code=500, detail=f"LLM error: {llm_err}")

        return {
            "filename": filename,
            "mode": "local_paddleocr_ollama",
            "ocr_items_count": len(ocr_result.items),
            "ocr_items": [item.model_dump() for item in ocr_result.items],
            "raw_text": formatted_text,
            "document_result": document_result.model_dump()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
