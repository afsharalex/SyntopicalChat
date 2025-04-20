"""Arxiv client for searching and downloading papers from Arxiv."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arxiv
import requests


class ArxivClient:
    """Client for interacting with the Arxiv API."""

    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize the Arxiv client.

        Args:
            download_dir: Directory to download papers to. If None, a temporary directory will be used.
        """
        self.download_dir = download_dir or Path(tempfile.mkdtemp())
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search Arxiv for papers matching the query.

        Args:
            query: Query string to search for.
            max_results: Maximum number of results to return.

        Returns:
            List of paper metadata.
        """
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for result in client.results(search):
            paper_info = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
                "arxiv_id": result.get_short_id(),
                "categories": result.categories,
            }
            results.append(paper_info)

        return results

    def download_paper(self, paper_info: Dict) -> Path:
        """
        Download a paper from Arxiv.

        Args:
            paper_info: Paper metadata from search results.

        Returns:
            Path to the downloaded PDF file.
        """
        arxiv_id = paper_info["arxiv_id"]
        pdf_url = paper_info["pdf_url"]
        
        # Create a sanitized filename
        filename = f"{arxiv_id}.pdf"
        file_path = self.download_dir / filename
        
        # Download the PDF if it doesn't exist
        if not file_path.exists():
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return file_path

    def search_and_download(self, query: str, max_results: int = 5) -> List[Tuple[Dict, Path]]:
        """
        Search Arxiv and download matching papers.

        Args:
            query: Query string to search for.
            max_results: Maximum number of results to return.

        Returns:
            List of tuples containing paper metadata and path to downloaded PDF.
        """
        papers = self.search(query, max_results)
        results = []
        
        for paper in papers:
            try:
                pdf_path = self.download_paper(paper)
                results.append((paper, pdf_path))
            except Exception as e:
                print(f"Error downloading {paper['title']}: {str(e)}")
        
        return results