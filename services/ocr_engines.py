"""Select and execute the configured OCR engine without class-based wrappers."""

from __future__ import annotations
import os
from typing import Any

ENGINE_ALIASES = {
    "easy": "easyocr",
    "easy_ocr": "easyocr",
    "easyocr": "easyocr",
    "hf": "huggingface",
    "hugging_face": "huggingface",
    "huggingface": "huggingface",
    "qwen": "huggingface",
    "qwen2_vl": "huggingface",
    "google": "gemini",
    "gemini": "gemini",
}

SUPPORTED_ENGINES = ["easyocr", "gemini", "huggingface"]


def normalize_engine_name(engine: str | None) -> str:
    value = (engine or "easyocr").strip().lower().replace("-", "_")
    return ENGINE_ALIASES.get(value, value)


def extract_pdf_to_json(
    engine: str,
    pdf_path: str,
    output_path: str | None = None,
    target_items: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    engine = normalize_engine_name(engine)
    if engine == "easyocr":
        from services.easyocr_pipeline import process_pdf_to_json

        output_dir = output_path or os.path.dirname(pdf_path)
        _, merged_data = process_pdf_to_json(pdf_path, output_base_dir=output_dir)
        return merged_data

    if engine == "gemini":
        from services.gemini import extract_pdf_to_json as gemini_extract

        gemini_kwargs: dict[str, object] = {}
        if "model" in kwargs and kwargs["model"] is not None:
            gemini_kwargs["model"] = kwargs["model"]
        if target_items:
            gemini_kwargs["target_items"] = target_items

        return gemini_extract(pdf_path, output_path=output_path, **gemini_kwargs)

    if engine == "huggingface":
        from services.huggingface import extract_pdf_to_json as hf_extract

        hf_kwargs = {
            "dpi": kwargs.get("dpi", 180),
            "max_new_tokens": kwargs.get("max_new_tokens", 2048),
            "mode": kwargs.get("mode", "document"),
        }
        if "model_name" in kwargs:
            hf_kwargs["model_name"] = kwargs["model_name"]
        if target_items:
            hf_kwargs["target_items"] = target_items
        
        return hf_extract(pdf_path, output_path=output_path, **hf_kwargs)

    raise ValueError(f"Unsupported OCR engine: {engine}")
