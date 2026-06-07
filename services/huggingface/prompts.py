"""Qwen-VL extraction prompts and defaults."""

DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

DEFAULT_PROMPT = """
You are an OCR and document-understanding engine for pharmacy invoices, tables, and reports.
Analyze the provided document page image(s) and return only valid JSON. Do not wrap the response in markdown.

Extract all visible Arabic and English text. Preserve Arabic text as Arabic.
If the page has tables, reconstruct rows and columns as accurately as possible.

Use this JSON shape:
{
  "document_type": "invoice|table|report|unknown",
  "language": "ar|en|mixed|unknown",
  "metadata": {},
  "pages": [
    {
      "page": 1,
      "tables": [],
      "raw_text": "",
      "warnings": []
    }
  ],
  "tables": [
    {
      "page": 1,
      "title": null,
      "headers": [],
      "rows": [
        {"Col_1": "", "Col_2": ""}
      ]
    }
  ],
  "line_items": [],
  "raw_text": "",
  "warnings": []
}

If a value is missing, use null or an empty list.
"""

PAGE_PROMPT = """
You are an OCR and document-understanding engine for pharmacy invoices, tables, and reports.
Analyze this page image and return only valid JSON. Do not wrap the response in markdown.

Extract all visible Arabic and English text. Preserve Arabic text as Arabic.
If the page has tables, reconstruct rows and columns as accurately as possible.

Use this JSON shape:
{
  "page": null,
  "document_type": "invoice|table|report|unknown",
  "language": "ar|en|mixed|unknown",
  "metadata": {},
  "tables": [
    {
      "title": null,
      "headers": [],
      "rows": [
        {"Col_1": "", "Col_2": ""}
      ]
    }
  ],
  "line_items": [],
  "raw_text": "",
  "warnings": []
}

If a value is missing, use null or an empty list.
"""


def get_product_aware_prompt(target_products: list[dict[str, str]] | None = None) -> str:
    """Generate a product-aware extraction prompt if target products are provided."""
    if not target_products:
        return DEFAULT_PROMPT
    
    product_list = "\n".join([
        f"  - {item.get('product_name', '')} {f'({item.get(\"strength\", \"\")})' if item.get('strength') else ''}"
        for item in target_products[:20]  # Limit to first 20 for token efficiency
    ])
    
    return f"""
You are an OCR and document-understanding engine for pharmacy invoices, tables, and reports.
Analyze the provided document page image(s) and return only valid JSON. Do not wrap the response in markdown.

IMPORTANT: Focus especially on finding and extracting these target products:
{product_list}

Extract all visible Arabic and English text. Preserve Arabic text as Arabic.
If the page has tables, reconstruct rows and columns as accurately as possible.
Include raw_text field with all extracted text for product matching purposes.

Use this JSON shape:
{{
  "document_type": "invoice|table|report|unknown",
  "language": "ar|en|mixed|unknown",
  "metadata": {{}},
  "pages": [
    {{
      "page": 1,
      "tables": [],
      "raw_text": "",
      "warnings": []
    }}
  ],
  "tables": [
    {{
      "page": 1,
      "title": null,
      "headers": [],
      "rows": [
        {{"Col_1": "", "Col_2": ""}}
      ]
    }}
  ],
  "line_items": [],
  "raw_text": "",
  "target_product_matches": [],
  "warnings": []
}}

If a value is missing, use null or an empty list.
"""
