def parse_text(file_path: str):
    """
    Parses plain text files.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        raise e