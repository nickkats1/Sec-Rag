import pytest
from pypdf import PdfWriter


@pytest.fixture()
def pdf_file(tmp_path):
    """A single minimal one-page PDF file."""
    pdf_path = tmp_path / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


