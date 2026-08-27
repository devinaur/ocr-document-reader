"""
models.py
---------
This module defines the data structures used throughout the application using Pydantic.

Why Pydantic?
- Pydantic ensures type safety and validates the raw dictionary output from OCR/LLM
  into predictable Python objects.
- It automatically handles missing fields or incorrect data types, allowing us to generate
  a strict JSON schema to send to Ollama/Llama 3.1.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class OCRItem(BaseModel):
    """
    Represents a single text element detected by PaddleOCR in an image.
    
    Attributes:
        text: The recognized string content.
        confidence: Floating point confidence score (0.0 to 1.0) calculated by PaddleOCR.
        bounding_box: Polygon coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 
                      defining where the text was located in the image pixels.
    """
    text: str = Field(..., description="Recognized text line")
    confidence: float = Field(..., description="Detection confidence score between 0.0 and 1.0")
    bounding_box: List[List[float]] = Field(
        ..., description="4 point bounding polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )


class OCRResult(BaseModel):
    """
    Container for all detected text blocks within a single document image.
    
    Attributes:
        image_path: Path to the input image file.
        items: List of detected OCRItem objects.
    """
    image_path: str
    items: List[OCRItem] = Field(default_factory=list)


class DocumentResult(BaseModel):
    """
    Structured representation of the document content extracted by Llama 3.1.
    
    This flexible schema accommodates invoices, receipts, forms, CVs, and general documents.
    
    Attributes:
        document_type: Category of the document (e.g. 'Invoice', 'Receipt', 'Resume/CV', 'Form', 'Unknown').
        title: Main heading or title of the document if present, otherwise None.
        date: Extracted document date string (e.g. '26/08/2026') or None.
        entities: Key-value dictionary containing specific extracted details (e.g. invoice items, total, sender, recipient).
        summary: Brief 1-2 sentence summary of what this document contains.
    """
    document_type: str = Field(
        ..., 
        description="Type/category of document (e.g., Invoice, Receipt, Resume, Form, Unknown)"
    )
    title: Optional[str] = Field(
        default=None, 
        description="Main title or heading of the document"
    )
    date: Optional[str] = Field(
        default=None, 
        description="Date mentioned on the document in string format"
    )
    entities: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Extracted key-value details (e.g., items, amounts, vendor, buyer)"
    )
    summary: Optional[str] = Field(
        default=None, 
        description="Brief concise summary of the document"
    )
