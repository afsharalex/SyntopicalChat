"""PDF processor implementation for extracting text and metadata from academic papers."""

from pathlib import Path
from typing import Dict, List, Optional

import pypdf
from pydantic import BaseModel


class PaperMetadata(BaseModel):
    """Metadata for an academic paper."""

    title: str
    authors: List[str] = []
    publication_date: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = []
    doi: Optional[str] = None
    source_file: Path


class PaperContent(BaseModel):
    """Content of an academic paper."""

    metadata: PaperMetadata
    text: str
    sections: Dict[str, str] = {}


class PDFProcessor:
    """Processor for extracting text and metadata from PDF files."""

    def __init__(self):
        """Initialize the PDF processor."""
        pass

    def extract_metadata(self, pdf_path: Path) -> PaperMetadata:
        """
        Extract metadata from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Metadata for the paper.
        """
        reader = pypdf.PdfReader(pdf_path)
        info = reader.metadata
        
        # Extract title from metadata or first page
        title = info.title if info and info.title else self._extract_title_from_text(reader)
        
        # Extract authors from metadata
        authors = []
        if info and info.author:
            # Split author string by common separators
            authors = [a.strip() for a in info.author.split(",")]
        
        return PaperMetadata(
            title=title or pdf_path.stem,
            authors=authors,
            publication_date=info.creation_date.strftime("%Y-%m-%d") if info and info.creation_date else None,
            source_file=pdf_path
        )

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text from the PDF.
        """
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
            
        return text

    def process_pdf(self, pdf_path: Path) -> PaperContent:
        """
        Process a PDF file to extract content and metadata.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Processed paper content.
        """
        metadata = self.extract_metadata(pdf_path)
        text = self.extract_text(pdf_path)
        
        # Try to extract abstract and sections
        sections = self._extract_sections(text)
        
        # Update metadata with abstract if found
        if "abstract" in sections:
            metadata.abstract = sections["abstract"]
            
        return PaperContent(
            metadata=metadata,
            text=text,
            sections=sections
        )

    def _extract_title_from_text(self, reader: pypdf.PdfReader) -> str:
        """
        Extract title from the first page text.

        Args:
            reader: PDF reader object.

        Returns:
            Extracted title or empty string.
        """
        if not reader.pages:
            return ""
            
        first_page_text = reader.pages[0].extract_text()
        lines = first_page_text.split("\n")
        
        # Heuristic: first non-empty line is often the title
        for line in lines:
            if line.strip():
                return line.strip()
                
        return ""

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from paper text using heuristics.

        Args:
            text: Full text of the paper.

        Returns:
            Dictionary of section names to section content.
        """
        sections = {}
        
        # Simple heuristic to find abstract
        lower_text = text.lower()
        abstract_start = lower_text.find("abstract")
        
        if abstract_start != -1:
            # Find the end of the abstract (next section or introduction)
            intro_start = lower_text.find("introduction", abstract_start)
            if intro_start == -1:
                # If no introduction, look for other common section headers
                possible_ends = [
                    lower_text.find("keywords", abstract_start),
                    lower_text.find("1.", abstract_start),
                    lower_text.find("i.", abstract_start),
                    lower_text.find("background", abstract_start),
                ]
                possible_ends = [pos for pos in possible_ends if pos != -1]
                
                if possible_ends:
                    abstract_end = min(possible_ends)
                else:
                    # If no clear end, take a reasonable chunk
                    abstract_end = abstract_start + 1500
            else:
                abstract_end = intro_start
                
            # Extract the abstract text
            abstract_text = text[abstract_start:abstract_end].strip()
            # Remove the "abstract" header
            if abstract_text.lower().startswith("abstract"):
                abstract_text = abstract_text[8:].strip()
                
            sections["abstract"] = abstract_text
            
        return sections