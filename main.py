"""
main.py
-------
Command line entrypoint for the AI OCR Document Reader pipeline.

Pipeline flow:
Image Path -> PaddleOCR (ocr.py) -> OCR items (text + coordinates) 
           -> Formatted text -> Ollama / Llama 3.1 8B (llm.py) 
           -> Validated Pydantic Schema (models.py) -> Pretty JSON

Usage:
    python main.py documents/sample_invoice.png
"""

import sys
import os
import json
from ocr import run_ocr, format_ocr_text
from llm import analyze_document_text


def print_section(title: str):
    """Utility helper to print visually distinct section headers in the terminal."""
    print("\n" + "=" * 65)
    print(f"  {title.upper()}")
    print("=" * 65)


def main():
    # Step 0: Command line argument check
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_document_image>")
        print("Example: python main.py documents/sample_invoice.png")
        sys.exit(1)

    image_path = sys.argv[1]

    # Validate image path exists before running pipeline
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    print(f"🚀 Starting Document Processing Pipeline for: '{image_path}'")

    # =========================================================================
    # STEP 1: Run PaddleOCR
    # =========================================================================
    print_section("Step 1: Running PaddleOCR Text Detection & Recognition")

    try:
        ocr_result = run_ocr(image_path)
    except FileNotFoundError as err:
        print(f"❌ File Error: {err}")
        sys.exit(1)
    except RuntimeError as err:
        print(f"❌ OCR Runtime Error: {err}")
        sys.exit(1)
    except Exception as err:
        print(f"❌ Unexpected error during OCR processing: {err}")
        sys.exit(1)

    # Print raw OCR output metadata (bounding box, text, confidence) for educational visibility
    print(f"\nFound {len(ocr_result.items)} text block(s) in document:\n")
    for idx, item in enumerate(ocr_result.items, start=1):
        # Format bounding box as simple point list
        points_str = ", ".join([f"({pt[0]:.0f},{pt[1]:.0f})" for pt in item.bounding_box])
        print(f"  [{idx:02d}] Text: \"{item.text}\"")
        print(f"       Confidence: {item.confidence:.4f}")
        print(f"       Coordinates: [{points_str}]\n")

    # =========================================================================
    # STEP 2: Format OCR Output into Text Block for LLM
    # =========================================================================
    print_section("Step 2: Formatting Raw OCR Text for LLM Input")

    formatted_ocr_text = format_ocr_text(ocr_result)

    print("Extracted Reading Order Text Block:")
    print("-----------------------------------")
    if formatted_ocr_text:
        print(formatted_ocr_text)
    else:
        print("(No text extracted from image)")
    print("-----------------------------------")

    # =========================================================================
    # STEP 3: Pass OCR Text to Llama 3.2 via Ollama for Structured Extraction
    # =========================================================================
    print_section("Step 3: Analyzing Text with Llama 3.2 3B via Ollama")

    try:
        document_structure = analyze_document_text(
            ocr_text=formatted_ocr_text, 
            model_name="llama3.2:3b"
        )
    except ConnectionError as err:
        print(f"❌ Ollama Connection Error:\n{err}")
        sys.exit(1)
    except ValueError as err:
        print(f"❌ Model / Validation Error:\n{err}")
        sys.exit(1)
    except Exception as err:
        print(f"❌ LLM Processing Error:\n{err}")
        sys.exit(1)

    # =========================================================================
    # STEP 4: Output Verified Pydantic JSON Result
    # =========================================================================
    print_section("Step 4: Final Structured Document Result (Pydantic Validated JSON)")

    # Convert Pydantic model to dictionary, then format as pretty JSON
    json_output = document_structure.model_dump()
    pretty_json = json.dumps(json_output, indent=2, ensure_ascii=False)

    print(pretty_json)
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
