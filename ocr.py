"""
ocr.py
------
This module handles Optical Character Recognition (OCR) using PaddleOCR.
Supports both image files (PNG, JPG) and PDF documents.

How PaddleOCR works:
1. Text Detection (DBNet): Finds rectangular regions in the image that contain text.
2. Angle Classification: Detects whether text is rotated or upside down and corrects it.
3. Text Recognition (CRNN / SVTR): Converts image pixels within detected regions into characters/strings.
"""

import os
import tempfile
from typing import List, Optional
from PIL import Image
from models import OCRResult, OCRItem


def process_pdf_pages(pdf_path: str) -> List[str]:
    """
    Renders PDF pages into temporary PNG images using pypdfium2.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        List of file paths to generated temporary page image files.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise ImportError(
            "pypdfium2 is required for PDF processing. Please install via 'pip install pypdfium2'."
        ) from e

    pdf = pdfium.PdfDocument(pdf_path)
    temp_dir = tempfile.mkdtemp(prefix="ocr_pdf_")
    image_paths = []

    for i, page in enumerate(pdf):
        # Render page to PIL Image at 200 DPI for high OCR accuracy
        bitmap = page.render(scale=200 / 72)
        pil_image = bitmap.to_pil().convert("RGB")
        
        page_path = os.path.join(temp_dir, f"page_{i + 1}.png")
        pil_image.save(page_path)
        image_paths.append(page_path)

    return image_paths


def run_ocr(image_path: str, lang: str = "en") -> OCRResult:
    """
    Executes PaddleOCR on the specified image or PDF file and extracts text with bounding boxes.

    Args:
        image_path: Path to the image or PDF file.
        lang: Language model to use for recognition (default is 'en' for English).

    Returns:
        OCRResult containing structured OCRItem objects with text, confidence, and bounding box.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If PaddleOCR fails during execution.
    """
    # Step 1: Check that the input file actually exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: File not found at path '{image_path}'")

    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise ImportError(
            "PaddleOCR is not installed. Please install it via 'pip install paddleocr paddlepaddle'."
        ) from e

    # Handle PDF files by converting pages to temporary images
    is_pdf = image_path.lower().endswith(".pdf")
    target_images = []
    
    if is_pdf:
        print(f"[OCR] PDF detected: Rendering pages from '{image_path}'...")
        target_images = process_pdf_pages(image_path)
    else:
        target_images = [image_path]

    print(f"[OCR] Initializing PaddleOCR (language: {lang})...")
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    except TypeError:
        # Fallback if specific kwargs vary
        ocr = PaddleOCR(lang=lang)

    ocr_items = []

    for img_idx, img_file in enumerate(target_images, start=1):
        print(f"[OCR] Processing page/image {img_idx}/{len(target_images)}: {img_file}...")
        
        # Ensure image is 3-channel RGB (converts RGBA/transparent PNGs cleanly)
        try:
            with Image.open(img_file) as img:
                if img.mode != "RGB":
                    rgb_img = img.convert("RGB")
                    # Save normalized RGB copy
                    img_file = img_file + "_normalized.jpg"
                    rgb_img.save(img_file)
        except Exception:
            pass  # Fall back to original file path if PIL read fails

        try:
            try:
                results = ocr.ocr(img_file, cls=True)
            except Exception:
                results = ocr.ocr(img_file)
        except Exception as err:
            raise RuntimeError(f"PaddleOCR processing failed for '{img_file}': {err}") from err

        if results:
            # Handle page result wrapper
            page_results = results[0] if isinstance(results, list) and len(results) > 0 else results
            if isinstance(page_results, dict):
                # Handle PaddleOCR 3.7 / PaddleX dictionary output
                texts = page_results.get("rec_text", page_results.get("rec_texts", []))
                scores = page_results.get("rec_score", page_results.get("rec_scores", []))
                polys = page_results.get("dt_polys", page_results.get("dt_polygons", []))
                
                for idx, text in enumerate(texts):
                    score = float(scores[idx]) if idx < len(scores) else 1.0
                    poly = polys[idx].tolist() if hasattr(polys[idx], 'tolist') else polys[idx] if idx < len(polys) else [[0,0],[0,0],[0,0],[0,0]]
                    ocr_items.append(OCRItem(text=str(text), confidence=score, bounding_box=poly))

            elif isinstance(page_results, (list, tuple)):
                for line in page_results:
                    if not line:
                        continue
                    if isinstance(line, dict):
                        text = line.get("text", line.get("rec_text", ""))
                        score = float(line.get("confidence", line.get("score", 1.0)))
                        bbox = line.get("box", line.get("points", [[0,0],[0,0],[0,0],[0,0]]))
                        if text:
                            ocr_items.append(OCRItem(text=str(text), confidence=score, bounding_box=bbox))
                    elif isinstance(line, (list, tuple)) and len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0])
                            confidence = float(text_info[1])
                        else:
                            text = str(text_info)
                            confidence = 1.0

                        ocr_items.append(OCRItem(
                            text=text,
                            confidence=confidence,
                            bounding_box=bbox
                        ))

    print(f"[OCR] Complete! Extracted {len(ocr_items)} text line(s) in total.")
    return OCRResult(image_path=image_path, items=ocr_items)


def format_ocr_text(ocr_result: OCRResult) -> str:
    """
    Converts raw OCR items into a clean, human-readable text block suitable for LLM prompts.

    Args:
        ocr_result: The OCRResult instance returned by run_ocr().

    Returns:
        A multiline string representing the document text in reading order.
    """
    if not ocr_result.items:
        return ""

    # Sort text lines by top-to-bottom vertical position (y coordinate of top-left corner)
    sorted_items = sorted(ocr_result.items, key=lambda item: item.bounding_box[0][1])

    text_lines = [item.text.strip() for item in sorted_items if item.text.strip()]
    formatted_text = "\n".join(text_lines)

    return formatted_text
