"""Tests for the Arxiv Integration module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from syntopicalchat.arxiv_integration.arxiv_client import ArxivClient


@pytest.mark.unit
def test_arxiv_client_initialization(temp_dir):
    """Test that ArxivClient can be initialized."""
    # Initialize with default download directory
    client = ArxivClient()
    assert client is not None
    assert client.download_dir.exists()
    
    # Initialize with custom download directory
    download_dir = temp_dir / "arxiv_papers"
    client = ArxivClient(download_dir=download_dir)
    assert client.download_dir == download_dir
    assert client.download_dir.exists()


@pytest.mark.unit
def test_search(mock_arxiv_response):
    """Test searching for papers on Arxiv."""
    with patch("arxiv.Client") as mock_client, \
         patch("arxiv.Search") as mock_search:
        # Mock the arxiv.Client and arxiv.Search classes
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the results method to return mock papers
        mock_results = []
        for paper_info in mock_arxiv_response:
            mock_paper = MagicMock()
            mock_paper.title = paper_info["title"]
            mock_paper.authors = [MagicMock(name=author) for author in paper_info["authors"]]
            mock_paper.summary = paper_info["summary"]
            mock_paper.published = MagicMock()
            mock_paper.published.strftime.return_value = paper_info["published"]
            mock_paper.pdf_url = paper_info["pdf_url"]
            mock_paper.entry_id = paper_info["entry_id"]
            mock_paper.get_short_id.return_value = paper_info["arxiv_id"]
            mock_paper.categories = paper_info["categories"]
            mock_results.append(mock_paper)
        
        mock_client_instance.results.return_value = mock_results
        
        # Initialize ArxivClient
        client = ArxivClient()
        
        # Search for papers
        results = client.search("quantum computing", max_results=2)
        
        # Verify the results
        assert isinstance(results, list)
        assert len(results) == len(mock_arxiv_response)
        for i, result in enumerate(results):
            assert result["title"] == mock_arxiv_response[i]["title"]
            assert result["authors"] == mock_arxiv_response[i]["authors"]
            assert result["summary"] == mock_arxiv_response[i]["summary"]
            assert result["published"] == mock_arxiv_response[i]["published"]
            assert result["pdf_url"] == mock_arxiv_response[i]["pdf_url"]
            assert result["entry_id"] == mock_arxiv_response[i]["entry_id"]
            assert result["arxiv_id"] == mock_arxiv_response[i]["arxiv_id"]
            assert result["categories"] == mock_arxiv_response[i]["categories"]


@pytest.mark.unit
def test_download_paper(temp_dir, mock_arxiv_response):
    """Test downloading a paper from Arxiv."""
    with patch("requests.get") as mock_get:
        # Mock the requests.get method
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_response
        
        # Initialize ArxivClient with a temporary download directory
        client = ArxivClient(download_dir=temp_dir)
        
        # Download a paper
        paper_info = mock_arxiv_response[0]
        pdf_path = client.download_paper(paper_info)
        
        # Verify the result
        assert pdf_path.exists()
        assert pdf_path.name == f"{paper_info['arxiv_id']}.pdf"
        assert pdf_path.parent == temp_dir
        
        # Verify the requests.get call
        mock_get.assert_called_once_with(paper_info["pdf_url"], stream=True)
        mock_response.raise_for_status.assert_called_once()
        mock_response.iter_content.assert_called_once_with(chunk_size=8192)


@pytest.mark.unit
def test_search_and_download(temp_dir, mock_arxiv_response):
    """Test searching and downloading papers from Arxiv."""
    with patch.object(ArxivClient, "search") as mock_search, \
         patch.object(ArxivClient, "download_paper") as mock_download:
        # Mock the search method
        mock_search.return_value = mock_arxiv_response
        
        # Mock the download_paper method
        mock_download.side_effect = [
            temp_dir / f"{paper['arxiv_id']}.pdf" for paper in mock_arxiv_response
        ]
        
        # Initialize ArxivClient with a temporary download directory
        client = ArxivClient(download_dir=temp_dir)
        
        # Search and download papers
        results = client.search_and_download("quantum computing", max_results=2)
        
        # Verify the results
        assert isinstance(results, list)
        assert len(results) == len(mock_arxiv_response)
        for i, (paper, pdf_path) in enumerate(results):
            assert paper == mock_arxiv_response[i]
            assert pdf_path == temp_dir / f"{paper['arxiv_id']}.pdf"
        
        # Verify the method calls
        mock_search.assert_called_once_with("quantum computing", max_results=2)
        assert mock_download.call_count == len(mock_arxiv_response)
        for i, paper in enumerate(mock_arxiv_response):
            mock_download.assert_any_call(paper)


@pytest.mark.integration
def test_end_to_end_arxiv(temp_dir):
    """Test end-to-end Arxiv integration."""
    with patch("arxiv.Client") as mock_client, \
         patch("arxiv.Search") as mock_search, \
         patch("requests.get") as mock_get:
        # Mock the arxiv.Client and arxiv.Search classes
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Create a mock paper
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper"
        mock_paper.authors = [MagicMock(name="Test Author")]
        mock_paper.summary = "This is a test paper."
        mock_paper.published = MagicMock()
        mock_paper.published.strftime.return_value = "2023-01-01"
        mock_paper.pdf_url = "https://arxiv.org/pdf/1234.5678v1"
        mock_paper.entry_id = "http://arxiv.org/abs/1234.5678v1"
        mock_paper.get_short_id.return_value = "1234.5678v1"
        mock_paper.categories = ["cs.AI"]
        
        # Mock the results method to return the mock paper
        mock_client_instance.results.return_value = [mock_paper]
        
        # Mock the requests.get method
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_response
        
        # Initialize ArxivClient with a temporary download directory
        client = ArxivClient(download_dir=temp_dir)
        
        # Search and download papers
        results = client.search_and_download("quantum computing", max_results=1)
        
        # Verify the results
        assert isinstance(results, list)
        assert len(results) == 1
        paper, pdf_path = results[0]
        assert paper["title"] == "Test Paper"
        assert paper["authors"] == ["Test Author"]
        assert paper["summary"] == "This is a test paper."
        assert paper["published"] == "2023-01-01"
        assert paper["pdf_url"] == "https://arxiv.org/pdf/1234.5678v1"
        assert paper["entry_id"] == "http://arxiv.org/abs/1234.5678v1"
        assert paper["arxiv_id"] == "1234.5678v1"
        assert paper["categories"] == ["cs.AI"]
        assert pdf_path.exists()
        assert pdf_path.name == "1234.5678v1.pdf"