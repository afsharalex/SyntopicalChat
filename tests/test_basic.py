"""Basic tests for the SyntopicalChat application."""

import os
from pathlib import Path

import pytest

from syntopicalchat import __version__
from syntopicalchat.pdf_processor.processor import PDFProcessor, PaperMetadata, PaperContent
from syntopicalchat.vector_db.storage import VectorDBStorage


def test_version():
    """Test that the version is defined."""
    assert __version__ == "0.1.0"


def test_paper_metadata():
    """Test that PaperMetadata can be created."""
    metadata = PaperMetadata(
        title="Test Paper",
        authors=["Author 1", "Author 2"],
        source_file=Path("test.pdf"),
    )
    assert metadata.title == "Test Paper"
    assert len(metadata.authors) == 2
    assert metadata.source_file == Path("test.pdf")


def test_paper_content():
    """Test that PaperContent can be created."""
    metadata = PaperMetadata(
        title="Test Paper",
        authors=["Author 1", "Author 2"],
        source_file=Path("test.pdf"),
    )
    content = PaperContent(
        metadata=metadata,
        text="This is a test paper.",
        sections={"abstract": "This is an abstract."},
    )
    assert content.metadata.title == "Test Paper"
    assert content.text == "This is a test paper."
    assert content.sections["abstract"] == "This is an abstract."


@pytest.mark.skipif(
    not os.path.exists("data/test.pdf"),
    reason="Test PDF file not found",
)
def test_pdf_processor():
    """Test that PDFProcessor can process a PDF file."""
    processor = PDFProcessor()
    assert processor is not None
    
    # This test requires a test PDF file to be present
    if os.path.exists("data/test.pdf"):
        content = processor.process_pdf(Path("data/test.pdf"))
        assert content.metadata.title is not None
        assert content.text is not None