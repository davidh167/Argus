"""Plain-text extraction from an uploaded resume/CV file (PDF or DOCX), for
the profile intake chat's "attach a resume" button -- the extracted text
loads into the chat input for the user to review/edit before sending, it's
never sent to the model directly and nothing here is persisted."""

import io

from docx import Document
from pypdf import PdfReader


class UnsupportedFileTypeError(Exception):
    pass


def extract_text(filename: str, content: bytes) -> str:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "pdf":
        return _extract_pdf_text(content)
    if extension == "docx":
        return _extract_docx_text(content)
    raise UnsupportedFileTypeError(
        f"Unsupported file type: .{extension or '(none)'} -- only .pdf and .docx are supported."
    )


def _extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _extract_docx_text(content: bytes) -> str:
    document = Document(io.BytesIO(content))
    # Paragraphs only, not table cells -- covers the common resume layout;
    # a resume built entirely from tables would need more, not assumed here.
    return "\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()
