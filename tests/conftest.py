"""Pytest configuration and fixtures for SyntopicalChat tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Generator, List

import pytest
from pypdf import PdfWriter

from syntopicalchat.pdf_processor.processor import PaperContent, PaperMetadata, PDFProcessor
from syntopicalchat.vector_db.storage import VectorDBStorage
from syntopicalchat.llm.chat import SyntopicalChat


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_pdf(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    # Create a simple PDF file
    pdf_path = temp_dir / "test_paper.pdf"
    
    writer = PdfWriter()
    
    # Add a page with title and content
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({
        "/Title": "Test Academic Paper",
        "/Author": "Test Author 1, Test Author 2",
        "/CreationDate": "D:20230101000000",
    })
    
    # Add some text to the page
    page = writer.pages[0]
    content = """
    Test Academic Paper
    
    Test Author 1, Test Author 2
    
    Abstract
    
    This is a test abstract for the academic paper. It contains some text that will be
    extracted by the PDF processor.
    
    Introduction
    
    This is the introduction section of the paper. It provides background information
    and context for the research.
    
    Methodology
    
    This section describes the methodology used in the research.
    
    Results
    
    This section presents the results of the research.
    
    Conclusion
    
    This section summarizes the findings and implications of the research.
    
    References
    
    1. Reference 1
    2. Reference 2
    """
    
    # Write the PDF file
    with open(pdf_path, "wb") as f:
        writer.write(f)
    
    return pdf_path


@pytest.fixture
def pdf_processor() -> PDFProcessor:
    """Create a PDF processor instance for testing."""
    return PDFProcessor()


@pytest.fixture
def sample_paper_metadata() -> PaperMetadata:
    """Create sample paper metadata for testing."""
    return PaperMetadata(
        title="Test Academic Paper",
        authors=["Test Author 1", "Test Author 2"],
        publication_date="2023-01-01",
        abstract="This is a test abstract.",
        keywords=["test", "academic", "paper"],
        source_file=Path("test_paper.pdf"),
    )


@pytest.fixture
def sample_paper_content(sample_paper_metadata: PaperMetadata) -> PaperContent:
    """Create sample paper content for testing."""
    return PaperContent(
        metadata=sample_paper_metadata,
        text="This is the full text of the test paper.",
        sections={
            "abstract": "This is a test abstract.",
            "introduction": "This is the introduction section.",
            "methodology": "This is the methodology section.",
            "results": "This is the results section.",
            "conclusion": "This is the conclusion section.",
        },
    )


@pytest.fixture
def vector_db(temp_dir: Path) -> VectorDBStorage:
    """Create a vector database instance for testing."""
    db_path = temp_dir / "test_chroma_db"
    return VectorDBStorage(persist_directory=db_path)


@pytest.fixture
def mock_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the OpenAI API key environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")


@pytest.fixture
def mock_arxiv_response() -> List[Dict]:
    """Create a mock response from the Arxiv API."""
    return [
        {
            "title": "Test Paper 1",
            "authors": ["Author 1", "Author 2"],
            "summary": "This is a summary of test paper 1.",
            "published": "2023-01-01",
            "pdf_url": "https://arxiv.org/pdf/1234.5678v1",
            "entry_id": "http://arxiv.org/abs/1234.5678v1",
            "arxiv_id": "1234.5678v1",
            "categories": ["cs.AI", "cs.LG"],
        },
        {
            "title": "Test Paper 2",
            "authors": ["Author 3", "Author 4"],
            "summary": "This is a summary of test paper 2.",
            "published": "2023-01-02",
            "pdf_url": "https://arxiv.org/pdf/8765.4321v1",
            "entry_id": "http://arxiv.org/abs/8765.4321v1",
            "arxiv_id": "8765.4321v1",
            "categories": ["cs.CL", "cs.AI"],
        },
    ]