"""Gemini extraction prompts and defaults."""

DEFAULT_MODEL = "gemini-2.5-flash"

DEFAULT_PROMPT = """
You are an OCR and document-understanding engine for pharmacy invoices and tabular PDFs.
Return only valid JSON. Do not wrap the response in markdown.

Extract:
- document_type
- language
- vendor/supplier fields when available
- invoice number, date, tax, totals when available
- all tables as rows and columns
- line_items when the document looks like an invoice

Use this JSON shape:
{
  "document_type": "invoice|table|report|unknown",
  "language": "ar|en|mixed|unknown",
  "metadata": {},
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

If a value is missing, use null or an empty list. Preserve Arabic text as Arabic.
"""


def get_product_aware_prompt(target_products: list[dict[str, str]] | None = None) -> str:
    """Generate a product-aware extraction prompt if target products are provided."""
    if not target_products:
        return DEFAULT_PROMPT
    
    def format_product(item):
        name = item.get('product_name', '')
        strength = item.get('strength', '')
        strength_str = f' ({strength})' if strength else ''
        return f'  - {name}{strength_str}'
    
    product_list = "\n".join([
        format_product(item)
        for item in target_products[:20]  # Limit to first 20 for token efficiency
    ])
    
    return f"""
You are an OCR and document-understanding engine for pharmacy invoices and tabular PDFs.
Return only valid JSON. Do not wrap the response in markdown.

IMPORTANT: Focus especially on finding and extracting these target products:
{product_list}

Also extract:
- document_type
- language (preserve Arabic as Arabic)
- vendor/supplier fields when available
- invoice number, date, tax, totals when available
- all tables as rows and columns (preserve product names exactly as they appear)
- line_items with product names and quantities

Use this JSON shape:
{{
  "document_type": "invoice|table|report|unknown",
  "language": "ar|en|mixed|unknown",
  "metadata": {{}},
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

Preserve ALL Arabic text as Arabic. Include raw_text field with all extracted text to help with product matching.
If a value is missing, use null or an empty list.
"""
