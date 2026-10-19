import io
from unittest.mock import MagicMock, patch

import pytest
from docx import Document

from app.services import resume_parsing


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_extract_text_from_a_real_docx_file():
    content = _make_docx_bytes(["John Doe", "Software Engineer with 5 years of experience."])
    text = resume_parsing.extract_text("resume.docx", content)
    assert text == "John Doe\nSoftware Engineer with 5 years of experience."


def test_extract_text_from_docx_skips_blank_paragraphs():
    content = _make_docx_bytes(["John Doe", "", "   ", "Engineer"])
    text = resume_parsing.extract_text("resume.docx", content)
    assert text == "John Doe\nEngineer"


def test_extract_text_from_pdf_joins_pages_with_blank_line():
    fake_page_1 = MagicMock(extract_text=lambda: "Page one text")
    fake_page_2 = MagicMock(extract_text=lambda: "Page two text")
    with patch("app.services.resume_parsing.PdfReader") as mock_reader_cls:
        mock_reader_cls.return_value = MagicMock(pages=[fake_page_1, fake_page_2])
        text = resume_parsing.extract_text("resume.pdf", b"fake pdf bytes")
    assert text == "Page one text\n\nPage two text"


def test_extract_text_from_pdf_handles_a_page_with_no_extractable_text():
    fake_page = MagicMock(extract_text=lambda: None)
    with patch("app.services.resume_parsing.PdfReader") as mock_reader_cls:
        mock_reader_cls.return_value = MagicMock(pages=[fake_page])
        text = resume_parsing.extract_text("resume.pdf", b"fake pdf bytes")
    assert text == ""


def test_extract_text_raises_for_an_unsupported_extension():
    with pytest.raises(resume_parsing.UnsupportedFileTypeError):
        resume_parsing.extract_text("resume.txt", b"hello")


def test_extract_text_raises_for_a_filename_with_no_extension():
    with pytest.raises(resume_parsing.UnsupportedFileTypeError):
        resume_parsing.extract_text("resume", b"hello")
