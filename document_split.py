from langchain_core.documents import Document


def modify_isbn(isbn):
    isbn_digits = ''.join(ch for ch in isbn if ch.isdigit())
    return int(isbn_digits)


def split_lines_to_documents(raw_doc):
    try:
        def _extract_isbn(isbn_str):
            try:
                return modify_isbn(isbn_str)
            except Exception:
                pass
            digits = ''.join(ch for ch in isbn_str if ch.isdigit())
            return int(digits) if digits else None
    except NameError:
        def _extract_isbn(isbn_str):
            digits = ''.join(ch for ch in isbn_str if ch.isdigit())
            return int(digits) if digits else None

    full_text = "\n".join(getattr(d, "page_content", "") for d in raw_doc)
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]

    line_docs = []
    source = None
    if raw_doc and isinstance(raw_doc, (list, tuple)) and getattr(raw_doc[0], "metadata", None):
        source = raw_doc[0].metadata.get("source")

    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) == 2:
            isbn_part, desc = parts
            isbn_val = _extract_isbn(isbn_part)
        else:
            isbn_val = None
            desc = line

        md = {}
        if source:
            md["source"] = source
        if isbn_val is not None:
            md["isbn"] = isbn_val

        line_docs.append(Document(page_content=desc, metadata=md))

    return line_docs