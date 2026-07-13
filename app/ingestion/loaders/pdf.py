from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF locally using pypdf.
    Falls back to pdfplumber for pages that yield no text (e.g. image-heavy pages).
    """
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        text_parts: list[str] = []
        blank_pages: list[int] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(text)
            else:
                blank_pages.append(i + 1)

        # Fallback: use pdfplumber for any pages pypdf returned blank
        if blank_pages:
            try:
                import pdfplumber

                with pdfplumber.open(file_path) as pdf:
                    for page_num in blank_pages:
                        page = pdf.pages[page_num - 1]
                        fallback_text = page.extract_text() or ""
                        if fallback_text.strip():
                            text_parts.append(fallback_text)
            except Exception:
                pass

        full_text = "\n".join(text_parts)

        return full_text

    except Exception:
        raise