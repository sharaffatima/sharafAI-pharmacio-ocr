"""Gemini PDF extraction client."""

from __future__ import annotations

import os
from typing import Any

from services.gemini.prompts import DEFAULT_MODEL, DEFAULT_PROMPT, get_product_aware_prompt
from common.env import get_env_value
from common.files import require_file
from common.json_io import extract_json_object, write_json


def _get_api_key() -> str:
    api_key = get_env_value("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment or .env")
    return api_key


def extract_pdf_to_json(
    pdf_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    target_items: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Upload a PDF to Gemini and return structured JSON.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the result JSON
        model: Gemini model name (default: gemini-2.5-flash)
        prompt: Custom extraction prompt (uses product-aware if target_items provided)
        target_items: List of products to focus extraction on: [{"product_name": str, "strength": str|None}]
    """
    import google.generativeai as genai

    pdf = require_file(pdf_path, label="PDF")
    
    # Use product-aware prompt if target_items provided and no custom prompt
    if target_items and prompt is None:
        prompt = get_product_aware_prompt(target_items)
    
    prompt = prompt or DEFAULT_PROMPT
    
    genai.configure(api_key=_get_api_key())
    model_obj = genai.GenerativeModel(model)
    
    with open(pdf, "rb") as pdf_file:
        response = model_obj.generate_content([
            {"mime_type": "application/pdf", "data": pdf_file.read()},
            prompt
        ], stream=False)
    
    response.resolve()

    data = extract_json_object(response.text or "{}")
    if output_path:
        write_json(data, output_path)
    return data
