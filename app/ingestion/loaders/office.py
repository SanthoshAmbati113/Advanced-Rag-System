from unstructured.partition.auto import partition

def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using the Unstructured library.
    Unlike PDFs, these formats are structured and lightweight, so they are processed locally.
    """
    try:
        # Unstructured automatically detects if it's docx or pptx
        elements = partition(filename=file_path)
        full_text = "\n".join([str(el) for el in elements])

        return full_text
    except Exception as e:
        raise e