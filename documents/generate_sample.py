"""
generate_sample.py
------------------
Utility script to generate a sample invoice image for testing PaddleOCR and LLM extraction.
Creates `documents/sample_invoice.png`.
"""

import os
from PIL import Image, ImageDraw, ImageFont


def create_sample_invoice(output_path: str = "documents/sample_invoice.png"):
    # Create white canvas 600x700
    width, height = 600, 700
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Use default bitmap font
    font = ImageFont.load_default()

    lines = [
        "INVOICE",
        "",
        "Invoice No: INV-001",
        "Date: 26/08/2026",
        "",
        "Customer: Jane Doe",
        "Vendor: Tech Store Inc.",
        "",
        "Items Purchased:",
        "------------------------------------",
        "Keyboard 2 Rp 500000",
        "Mouse 3 Rp 150000",
        "------------------------------------",
        "",
        "Total: Rp 1450000",
        "",
        "Thank you for your business!"
    ]

    y_position = 40
    for line in lines:
        draw.text((40, y_position), line, fill=(0, 0, 0), font=font)
        y_position += 30

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"✅ Generated sample invoice image at: '{output_path}'")


if __name__ == "__main__":
    create_sample_invoice()
