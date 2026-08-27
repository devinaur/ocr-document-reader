"""
llm.py
------
This module connects Python to Large Language Models (LLMs).

Supports dual execution modes:
1. Groq Cloud API (if GROQ_API_KEY environment variable is present) - Used for Vercel / Cloud deployment.
2. Ollama Local Server (fallback) - Used for offline local development on your Mac.
"""

import os
import json
from models import DocumentResult


def analyze_document_text(ocr_text: str, model_name: str = "llama3.2:3b") -> DocumentResult:
    """
    Sends raw OCR document text to a Llama model via Groq API or Ollama local server.

    Args:
        ocr_text: The clean text block extracted from the document image by OCR.
        model_name: The name of the local Ollama model or Groq cloud model.

    Returns:
        DocumentResult: Validated Pydantic object with type, title, date, entities, and summary.
    """
    if not ocr_text.strip():
        print("[LLM Warning] OCR text is empty. Returning blank document result.")
        return DocumentResult(
            document_type="Unknown",
            title=None,
            date=None,
            entities={},
            summary="Empty document: No readable text detected by OCR."
        )

    # Construct system prompt instructing Llama on strict extraction rules
    system_prompt = (
        "You are an expert document understanding AI. "
        "Your task is to analyze raw OCR text extracted from a document and convert it into structured JSON.\n\n"
        "Rules:\n"
        "1. Determine the document_type (e.g. Invoice, Receipt, Resume, Form, Unknown).\n"
        "2. Extract document title and date if present. If not found, return null.\n"
        "3. Extract key details into the 'entities' dictionary (e.g. items, price, vendor, customer, amounts, line items).\n"
        "4. Write a short 1-2 sentence summary in 'summary'.\n"
        "5. NEVER invent or hallucinate information that is not in the text.\n"
        "6. Return NULL for any field where information is missing or unavailable.\n"
        "7. Return ONLY valid JSON matching this schema keys: 'document_type', 'title', 'date', 'entities', 'summary'."
    )

    user_prompt = f"Document OCR Text:\n---\n{ocr_text}\n---"

    groq_api_key = os.getenv("GROQ_API_KEY")

    # =========================================================================
    # MODE 1: GROQ CLOUD API (Vercel / Cloud Deployment)
    # =========================================================================
    if groq_api_key:
        print("[LLM] GROQ_API_KEY detected! Using Groq Cloud API (Llama 3.2 3B)...")
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("Groq library is not installed. Install via 'pip install groq'.") from e

        client = Groq(api_key=groq_api_key)
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.2-3b-preview",
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_content = chat_completion.choices[0].message.content
        except Exception as err:
            raise RuntimeError(f"Groq API request failed: {err}") from err

    # =========================================================================
    # MODE 2: LOCAL OLLAMA SERVER (Local Mac Development)
    # =========================================================================
    else:
        print(f"[LLM] GROQ_API_KEY not set. Using local Ollama server ('{model_name}')...")
        try:
            import ollama
        except ImportError as e:
            raise ImportError("The 'ollama' Python library is not installed.") from e

        try:
            schema = DocumentResult.model_json_schema()
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=schema,
                options={"temperature": 0.0}
            )
            raw_content = response.get("message", {}).get("content", "")
        except Exception as err:
            error_msg = str(err).lower()
            if "connect" in error_msg or "refused" in error_msg or "connection" in error_msg:
                raise ConnectionError(
                    "Could not connect to Ollama server! Make sure Ollama is running (`ollama serve`)."
                ) from err
            else:
                raise RuntimeError(f"Ollama request failed: {err}") from err

    if not raw_content:
        raise ValueError("LLM returned an empty response.")

    # Validate JSON output using Pydantic
    try:
        validated_result = DocumentResult.model_validate_json(raw_content)
        return validated_result
    except Exception as parse_err:
        try:
            dict_data = json.loads(raw_content)
            return DocumentResult.model_validate(dict_data)
        except Exception:
            raise ValueError(
                f"Failed to parse LLM response as valid DocumentResult schema!\n"
                f"Raw response was:\n{raw_content}\n"
                f"Validation error details: {parse_err}"
            ) from parse_err
