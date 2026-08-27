"""
vision_llm.py
-------------
Cloud-mode document understanding using Groq Llama 3.2 Vision API.

How Groq Vision OCR works:
- Llama 3.2 Vision is a multimodal model: it has "eyes" that can read image pixels directly.
- Instead of first running PaddleOCR to extract text, we send the raw image to Groq.
- The Vision model reads the text from the image (OCR) AND understands document structure 
  (invoice, receipt, CV, etc.) all in a SINGLE API call.
- This is called "End-to-End Vision-LLM OCR" - faster, simpler, and no heavy dependencies!

Why this works on Vercel:
- No PaddleOCR or PaddlePaddle needed (saves 1.1 GB).
- Groq API is a lightweight HTTP call using the `groq` package (< 1 MB).
"""

import os
import base64
import json
import tempfile
from pathlib import Path
from PIL import Image
import io
from models import DocumentResult


def image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Converts an image file to a base64 encoded string for the Groq Vision API.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (base64_string, media_type) e.g. ("data...", "image/jpeg")
    """
    path = Path(image_path)
    ext = path.suffix.lower()

    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    # Open, convert to RGB (removes alpha transparency), and encode
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded, "image/jpeg"


def pdf_first_page_to_base64(pdf_path: str) -> tuple[str, str]:
    """
    Converts the first page of a PDF to a base64 image for Groq Vision.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (base64_string, media_type)
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise ImportError("pypdfium2 is required for PDF processing.") from e

    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[0]
    bitmap = page.render(scale=150 / 72)  # 150 DPI is sufficient for Vision API
    pil_image = bitmap.to_pil().convert("RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded, "image/jpeg"


def analyze_document_with_vision(file_path: str) -> dict:
    """
    Sends a document image directly to Groq Llama 3.2 Vision API for 
    end-to-end OCR and structured JSON extraction in a single API call.

    Args:
        file_path: Path to the uploaded image or PDF file.

    Returns:
        Dictionary containing:
        - document_result: Validated Pydantic DocumentResult as dict
        - raw_text: The OCR text extracted by the Vision model
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError("groq library is not installed. Run 'pip install groq'.") from e

    # Convert file to base64 image for Groq Vision API
    is_pdf = file_path.lower().endswith(".pdf")
    if is_pdf:
        print("[Vision OCR] Converting PDF first page to image...")
        b64_image, media_type = pdf_first_page_to_base64(file_path)
    else:
        print("[Vision OCR] Converting image to base64...")
        b64_image, media_type = image_to_base64(file_path)

    client = Groq(api_key=groq_api_key)

    system_prompt = (
        "You are an expert document OCR and understanding AI.\n"
        "Given a document image, you must:\n"
        "1. Read ALL text visible in the document image carefully.\n"
        "2. Determine the document_type (Invoice, Receipt, Resume, Form, Letter, Unknown).\n"
        "3. Extract the document title and date if present, otherwise return null.\n"
        "4. Extract key details into an 'entities' dictionary.\n"
        "5. Write a brief 1-2 sentence summary.\n"
        "6. NEVER invent information not visible in the document.\n"
        "7. Return ONLY valid JSON with keys: document_type, title, date, entities, summary, extracted_text.\n"
        "The 'extracted_text' field should contain all raw text you read from the document."
    )

    print("[Vision OCR] Sending image to Groq Llama 3.2 Vision API...")

    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2048
        )
    except Exception as err:
        raise RuntimeError(f"Groq Vision API call failed: {err}") from err

    raw_content = response.choices[0].message.content
    print("[Vision OCR] Received structured response from Groq Vision!")

    if not raw_content:
        raise ValueError("Groq Vision API returned an empty response.")

    # Parse and validate with Pydantic
    try:
        parsed = json.loads(raw_content)
    except Exception as e:
        raise ValueError(f"Groq Vision response was not valid JSON: {raw_content}") from e

    # Extract raw text from vision response if present
    raw_text = parsed.pop("extracted_text", "")

    # Validate into Pydantic model
    try:
        document_result = DocumentResult.model_validate(parsed)
    except Exception as e:
        # Flexible fallback if minor field mismatch
        document_result = DocumentResult(
            document_type=parsed.get("document_type", "Unknown"),
            title=parsed.get("title"),
            date=parsed.get("date"),
            entities=parsed.get("entities", {}),
            summary=parsed.get("summary")
        )

    return {
        "document_result": document_result.model_dump(),
        "raw_text": raw_text
    }
