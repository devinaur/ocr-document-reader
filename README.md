# 📄 Beginner-Friendly AI OCR Document Reader

An educational Python application demonstrating how Optical Character Recognition (**PaddleOCR**) and a local Large Language Model (**Llama 3.1 8B via Ollama**) work together to transform visual document images into validated, structured JSON data using **Pydantic**.

---

## 📌 Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [How PaddleOCR Works (High Level)](#3-how-paddleocr-works-high-level)
4. [Why an LLM is Used After OCR](#4-why-an-llm-is-used-after-ocr)
5. [How Ollama Connects Python to Llama](#5-how-ollama-connects-python-to-llama)
6. [Installing Python Dependencies](#6-installing-python-dependencies)
7. [Installing and Running Ollama](#7-installing-and-running-ollama)
8. [Downloading the Llama Model](#8-downloading-the-llama-model)
9. [How to Run the Application](#9-how-to-run-the-application)
10. [Example Input & Output](#10-example-input--output)

---

## 1. What This Project Does

This application takes a document image (such as an invoice, receipt, CV, or form) and processes it in two stages:
1. **PaddleOCR** reads the image pixels, identifies text locations (bounding boxes), and extracts raw text strings.
2. **Llama 3.1 8B** (running locally via **Ollama**) reads the raw text, understands its semantic structure, and formats key information into clean, validated JSON output matching a **Pydantic** schema.

---

## 2. Pipeline Architecture

```text
+-----------------------+
|  Document Image File  |  (e.g., sample_invoice.png)
+-----------+-----------+
            |
            v
+-----------------------+
|       PaddleOCR       |  DBNet (Text Detection) -> Angle Classifier -> CRNN/SVTR (Recognition)
+-----------+-----------+
            |
            v  [Raw OCR Items: text + confidence score + bounding box coordinates]
+-----------------------+
|  Clean Text Formatter |  Sorts text lines in reading order (top-to-bottom)
+-----------+-----------+
            |
            v  [Formatted Text Block]
+-----------------------+
|   Ollama (Llama 3.1)  |  Local Ollama server REST API (http://localhost:11434)
+-----------+-----------+
            |
            v  [Raw JSON string]
+-----------------------+
|   Pydantic Validation |  Validates schema against DocumentResult model
+-----------+-----------+
            |
            v
+-----------------------+
|  Pretty-Printed JSON  |  Final structured document summary
+-----------------------+
```

---

## 3. How PaddleOCR Works (High Level)

PaddleOCR performs text extraction in 3 main stages:

1. **Text Detection (DBNet - Differentiable Binarization)**:
   - Scans image pixel intensity and predicts bounding polygons `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` wherever words or sentences exist.
2. **Direction / Angle Classification**:
   - Detects if text lines are rotated (e.g. 90° or upside down) and automatically rotates them right-side up.
3. **Text Recognition (CRNN / SVTR)**:
   - Converts the cropped text regions into actual characters and words while returning a **confidence score** (e.g. `0.9850`).

---

## 4. Why an LLM is Used After OCR

- **OCR is literal, not smart**: OCR tells you *what* letters are on a page, but not *what they mean*. For instance, OCR yields `"Total: Rp 1450000"`, but doesn't know that `"1450000"` is the total invoice charge.
- **Handling unstructured layout variability**: Invoices, receipts, and CVs have wildly different layouts. Writing rigid regex patterns for every template breaks easily.
- **Semantic Understanding**: An LLM (Llama 3.1) understands natural language and document semantics. It maps raw text blocks directly to structured fields (`document_type`, `title`, `date`, `entities`, `summary`) without needing custom template rules.

---

## 5. How Ollama Connects Python to Llama

1. **Local Server (Daemon)**: Ollama runs a lightweight backend service at `http://localhost:11434`.
2. **Local Model Execution**: Ollama loads quantized open-weights (like Llama 3.1 8B) on your CPU/GPU.
3. **Python SDK**: The `ollama` Python library sends HTTP POST requests containing prompts to Ollama.
4. **Structured JSON Output**: Ollama allows passing a Pydantic JSON schema to force Llama to respond strictly in valid JSON format.

---

## 6. Installing Python Dependencies

### Prerequisites
- Python **3.11** installed.

### Step-by-Step Virtual Environment Setup

1. **Navigate to project directory**:
   ```bash
   cd ocr-document-reader
   ```

2. **Create virtual environment**:
   ```bash
   python3.11 -m venv venv
   ```

3. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows (Command Prompt):
     ```cmd
     venv\Scripts\activate
     ```

4. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 7. Installing and Running Ollama

1. **Download Ollama**:
   - Visit [https://ollama.com/download](https://ollama.com/download) and download installer for macOS, Linux, or Windows.
2. **Start Ollama service**:
   - Open terminal and start the server:
     ```bash
     ollama serve
     ```
   *(On macOS/Windows app installations, Ollama runs automatically in the background).*

---

## 8. Downloading the Llama Model

Pull the 8B Llama 3.1 model into Ollama by running:

```bash
ollama pull llama3.1:8b
```

Verify that the model is downloaded:

```bash
ollama list
```

---

## 9. How to Run the Application

1. **Generate a sample test document image**:
   ```bash
   python documents/generate_sample.py
   ```
   *(This creates `documents/sample_invoice.png`)*

2. **Run the pipeline**:
   ```bash
   python main.py documents/sample_invoice.png
   ```

---

## 10. Example Input & Output

### Input Document Image (`documents/sample_invoice.png`)
```text
INVOICE

Invoice No: INV-001
Date: 26/08/2026

Customer: Jane Doe
Vendor: Tech Store Inc.

Items Purchased:
Keyboard 2 Rp 500000
Mouse 3 Rp 150000

Total: Rp 1450000
```

### Raw OCR Terminal Output
```text
[01] Text: "INVOICE" | Confidence: 0.9982 | Coordinates: [(40,40), (120,40), (120,60), (40,60)]
[02] Text: "Invoice No: INV-001" | Confidence: 0.9950 | Coordinates: [(40,100), (220,100), (220,120), (40,120)]
...
```

### Final Structured JSON Output
```json
{
  "document_type": "Invoice",
  "title": "INVOICE",
  "date": "26/08/2026",
  "entities": {
    "invoice_number": "INV-001",
    "customer": "Jane Doe",
    "vendor": "Tech Store Inc.",
    "items": [
      {
        "name": "Keyboard",
        "quantity": 2,
        "price": "Rp 500000"
      },
      {
        "name": "Mouse",
        "quantity": 3,
        "price": "Rp 150000"
      }
    ],
    "total": "Rp 1450000"
  },
  "summary": "Invoice INV-001 issued on 26/08/2026 by Tech Store Inc. to Jane Doe for computer accessories totaling Rp 1450000."
}
```
