# Extraction Refactor Session

Goal: state-of-art file extraction surface with simple top-level API.

Boundaries:
- `extraction` owns domain model: documents, parts, metadata.
- `ocr` owns OCR ML/runtime and can be disabled.
- Heavy features must be optional.
- Public surface stays simple: bytes, text, content.

Current decisions:
- Keep OCR runtime isolated under `core::ocr`.
- Move PP-DocLayout layout detector to `core::layout` so layout can run without OCR.
- Build extraction around canonical `ExtractedDocument`/parts IR.
- Derive text from parts/content.

Implemented:
- `extraction-text`: light byte/text/omniparse extraction, no Burn/OCR/docs.
- `extraction-documents`: PDF/Office/EPUB/text parsing, no Burn/OCR.
- `extraction`: document extraction + Magika detection.
- `layout`: PP-DocLayout detection, no OCR recognition.
- `ocr`: OCR bundle, depends on layout.
- `full`: excludes OCR.
- `full-ml`: includes OCR.
- Simple APIs: `extract_file_bytes`, `extract_file_text`, `extract_file_content`, `extract_bytes`, `extract_text_bytes`, `extract_content_bytes`.
- OCR adapter: `content_from_ocr_page` behind `ocr` feature.

Backlog:
- Final checks/review loop.
- Decide later whether to expose layout-to-extraction adapter.
- Future: parser-specific parts instead of text-only part for PDF/Office.
