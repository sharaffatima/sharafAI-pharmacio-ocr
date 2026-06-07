"""High-level Hugging Face Qwen-VL extraction runner."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from services.huggingface.model import extract_document_json, extract_page_json, load_model
from common.pdf import render_pdf_pages
from services.huggingface.prompts import DEFAULT_MODEL, DEFAULT_PROMPT, PAGE_PROMPT, get_product_aware_prompt
from common.files import require_file
from common.json_io import write_json

print("Hugging Face Qwen-VL OCR engine loaded with model:", DEFAULT_MODEL)
print("Default prompt:", DEFAULT_PROMPT)


def extract_images_to_json(
    images: list[Image.Image],
    *,
    model_name: str = DEFAULT_MODEL,
    prompt: str | None = None,
    max_new_tokens: int = 2048,
    mode: str = "document",
    target_items: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    model, processor, torch = load_model(model_name)

    # Use product-aware prompt if target_items provided
    if target_items and prompt is None:
        prompt = get_product_aware_prompt(target_items)
    
    prompt = prompt or DEFAULT_PROMPT

    if mode == "document":
        print(f"Processing {len(images)} page(s) in one Qwen2-VL request with {model_name}")
        data = extract_document_json(
            images,
            model=model,
            processor=processor,
            torch=torch,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        return {
            "engine": "huggingface-qwen-vl",
            "model": model_name,
            "mode": mode,
            "result": data,
        }

    if mode != "pages":
        raise ValueError("mode must be 'document' or 'pages'")

    pages = []
    for page_number, image in enumerate(images, start=1):
        print(f"Processing page {page_number}/{len(images)} with {model_name}")
        pages.append(
            extract_page_json(
                image,
                page_number=page_number,
                model=model,
                processor=processor,
                torch=torch,
                prompt=PAGE_PROMPT,
                max_new_tokens=max_new_tokens,
            )
        )

    return {
        "engine": "huggingface-qwen-vl",
        "model": model_name,
        "mode": mode,
        "pages": pages,
    }


def extract_pdf_to_json(
    pdf_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    prompt: str | None = None,
    dpi: int = 180,
    max_new_tokens: int = 2048,
    mode: str = "document",
    target_items: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Extract PDF to JSON using Qwen-VL model.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the result JSON
        model_name: HuggingFace model name
        prompt: Custom extraction prompt
        dpi: PDF rendering DPI
        max_new_tokens: Maximum tokens for model output
        mode: "document" (all pages together) or "pages" (per-page)
        target_items: List of products to focus extraction on: [{"product_name": str, "strength": str|None}]
    """
    pdf = require_file(pdf_path, label="PDF")
    images = render_pdf_pages(pdf, dpi=dpi)
    
    # Use product-aware prompt if target_items provided and no custom prompt
    if target_items and prompt is None:
        prompt = get_product_aware_prompt(target_items)
    
    data = extract_images_to_json(
        images,
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        mode=mode,
        target_items=target_items,
    )

    if output_path:
        write_json(data, output_path)
    return data
