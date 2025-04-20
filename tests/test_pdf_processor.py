"""Tests for the PDF Processor module."""

import pytest
from pathlib import Path

from syntopicalchat.pdf_processor.processor import PDFProcessor, PaperMetadata, PaperContent


@pytest.mark.unit
def test_pdf_processor_initialization():
    """Test that PDFProcessor can be initialized."""
    processor = PDFProcessor()
    assert processor is not None


@pytest.mark.unit
def test_extract_metadata(pdf_processor, sample_pdf):
    """Test extracting metadata from a PDF file."""
    metadata = pdf_processor.extract_metadata(sample_pdf)
    
    assert isinstance(metadata, PaperMetadata)
    assert metadata.title == "Test Academic Paper"
    assert len(metadata.authors) > 0
    assert "Test Author" in metadata.authors[0]
    assert metadata.source_file == sample_pdf


@pytest.mark.unit
def test_extract_text(pdf_processor, sample_pdf):
    """Test extracting text from a PDF file."""
    text = pdf_processor.extract_text(sample_pdf)
    
    assert isinstance(text, str)
    assert len(text) > 0
    assert "Test Academic Paper" in text
    assert "Abstract" in text
    assert "Introduction" in text


@pytest.mark.unit
def test_process_pdf(pdf_processor, sample_pdf):
    """Test processing a PDF file."""
    paper_content = pdf_processor.process_pdf(sample_pdf)
    
    assert isinstance(paper_content, PaperContent)
    assert paper_content.metadata.title == "Test Academic Paper"
    assert len(paper_content.text) > 0
    assert "abstract" in paper_content.sections


@pytest.mark.unit
def test_extract_sections(pdf_processor, sample_pdf):
    """Test extracting sections from a PDF file."""
    # First get the text
    text = pdf_processor.extract_text(sample_pdf)
    
    # Then extract sections using the private method
    sections = pdf_processor._extract_sections(text)
    
    assert isinstance(sections, dict)
    assert "abstract" in sections
    assert len(sections["abstract"]) > 0


@pytest.mark.unit
def test_extract_title_from_text(pdf_processor, sample_pdf):
    """Test extracting title from text."""
    # Create a reader to pass to the method
    import pypdf
    reader = pypdf.PdfReader(sample_pdf)
    
    # Extract title
    title = pdf_processor._extract_title_from_text(reader)
    
    assert isinstance(title, str)
    assert len(title) > 0
    assert "Test Academic Paper" in title


@pytest.mark.integration
def test_end_to_end_pdf_processing(pdf_processor, sample_pdf):
    """Test end-to-end PDF processing."""
    # Process the PDF
    paper_content = pdf_processor.process_pdf(sample_pdf)
    
    # Verify the results
    assert paper_content.metadata.title == "Test Academic Paper"
    assert len(paper_content.metadata.authors) > 0
    assert len(paper_content.text) > 0
    assert "abstract" in paper_content.sections
    assert paper_content.metadata.source_file == sample_pdf