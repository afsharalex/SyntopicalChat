"""Tests for the CLI interface."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from syntopicalchat.cli.main import app
from syntopicalchat.pdf_processor.processor import PDFProcessor
from syntopicalchat.vector_db.storage import VectorDBStorage
from syntopicalchat.llm.chat import SyntopicalChat
from syntopicalchat.arxiv_integration.arxiv_client import ArxivClient


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.mark.unit
def test_app_callback(cli_runner):
    """Test the app callback function."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "SyntopicalChat" in result.stdout
    assert "analyze academic papers" in result.stdout.lower()


@pytest.mark.unit
def test_upload_command(cli_runner, sample_pdf, temp_dir):
    """Test the upload command."""
    with patch.object(PDFProcessor, "process_pdf") as mock_process_pdf, \
         patch.object(VectorDBStorage, "add_paper") as mock_add_paper:
        # Mock the process_pdf method
        mock_paper_content = MagicMock()
        mock_paper_content.metadata.title = "Test Paper"
        mock_process_pdf.return_value = mock_paper_content
        
        # Mock the add_paper method
        mock_add_paper.return_value = ["test-paper-0"]
        
        # Run the command
        result = cli_runner.invoke(
            app, 
            ["upload", str(sample_pdf), "--db-path", str(temp_dir / "test_db")]
        )
        
        # Verify the result
        assert result.exit_code == 0
        assert "Uploading PDF files" in result.stdout
        assert "Successfully processed Test Paper" in result.stdout
        assert "Upload complete" in result.stdout
        
        # Verify the method calls
        mock_process_pdf.assert_called_once_with(sample_pdf)
        mock_add_paper.assert_called_once_with(mock_paper_content)


@pytest.mark.unit
def test_list_command(cli_runner, temp_dir):
    """Test the list command."""
    with patch.object(VectorDBStorage, "get_all_papers") as mock_get_all_papers:
        # Mock the get_all_papers method
        mock_get_all_papers.return_value = [
            {
                "title": "Test Paper 1",
                "authors": "Author 1, Author 2",
                "publication_date": "2023-01-01",
            },
            {
                "title": "Test Paper 2",
                "authors": "Author 3, Author 4",
                "publication_date": "2023-01-02",
            },
        ]
        
        # Run the command
        result = cli_runner.invoke(
            app, 
            ["list", "--db-path", str(temp_dir / "test_db")]
        )
        
        # Verify the result
        assert result.exit_code == 0
        assert "Listing uploaded papers" in result.stdout
        assert "Test Paper 1" in result.stdout
        assert "Test Paper 2" in result.stdout
        assert "Author 1, Author 2" in result.stdout
        assert "Author 3, Author 4" in result.stdout
        
        # Verify the method calls
        mock_get_all_papers.assert_called_once()


@pytest.mark.unit
def test_chat_command(cli_runner, temp_dir, mock_openai_env):
    """Test the chat command."""
    with patch.object(SyntopicalChat, "chat") as mock_chat, \
         patch.object(typer, "prompt") as mock_prompt:
        # Mock the chat method
        mock_chat.return_value = {
            "answer": "This is a test answer.",
            "source_documents": [
                MagicMock(metadata={"title": "Test Paper"})
            ]
        }
        
        # Mock the prompt method to first return a question, then exit
        mock_prompt.side_effect = ["What is the main topic?", "exit"]
        
        # Run the command
        result = cli_runner.invoke(
            app, 
            ["chat", "--db-path", str(temp_dir / "test_db")]
        )
        
        # Verify the result
        assert result.exit_code == 0
        assert "Welcome to SyntopicalChat" in result.stdout
        assert "This is a test answer" in result.stdout
        assert "Sources" in result.stdout
        assert "Test Paper" in result.stdout
        assert "Goodbye" in result.stdout
        
        # Verify the method calls
        mock_chat.assert_called_once_with("What is the main topic?")
        assert mock_prompt.call_count == 2


@pytest.mark.unit
def test_analyze_command(cli_runner, temp_dir, mock_openai_env):
    """Test the analyze command."""
    with patch.object(SyntopicalChat, "analyze_topic") as mock_analyze_topic:
        # Mock the analyze_topic method
        mock_analyze_topic.return_value = {
            "answer": "This is a test analysis.",
            "source_documents": [
                MagicMock(metadata={"title": "Test Paper"})
            ]
        }
        
        # Run the command
        result = cli_runner.invoke(
            app, 
            ["analyze", "quantum computing", "--db-path", str(temp_dir / "test_db")]
        )
        
        # Verify the result
        assert result.exit_code == 0
        assert "Performing syntopical analysis" in result.stdout
        assert "This is a test analysis" in result.stdout
        assert "Sources" in result.stdout
        assert "Test Paper" in result.stdout
        
        # Verify the method calls
        mock_analyze_topic.assert_called_once_with("quantum computing")


@pytest.mark.unit
def test_start_command_folder_option(cli_runner, temp_dir, mock_openai_env):
    """Test the start command with folder option."""
    with patch("rich.prompt.Prompt.ask") as mock_ask, \
         patch("pathlib.Path.glob") as mock_glob, \
         patch.object(PDFProcessor, "process_pdf") as mock_process_pdf, \
         patch.object(VectorDBStorage, "add_paper") as mock_add_paper, \
         patch.object(SyntopicalChat, "chat") as mock_chat, \
         patch.object(typer, "prompt") as mock_prompt:
        # Mock the ask method to choose folder option
        mock_ask.return_value = "folder"
        
        # Mock the glob method to return sample PDFs
        sample_pdf = temp_dir / "test_paper.pdf"
        mock_glob.return_value = [sample_pdf]
        
        # Mock the process_pdf method
        mock_paper_content = MagicMock()
        mock_paper_content.metadata.title = "Test Paper"
        mock_process_pdf.return_value = mock_paper_content
        
        # Mock the add_paper method
        mock_add_paper.return_value = ["test-paper-0"]
        
        # Mock the chat method
        mock_chat.return_value = {
            "answer": "This is a test answer.",
            "source_documents": [
                MagicMock(metadata={"title": "Test Paper"})
            ]
        }
        
        # Mock the prompt method to exit after one question
        mock_prompt.side_effect = ["What is the main topic?", "exit"]
        
        # Run the command
        result = cli_runner.invoke(
            app, 
            ["start", "--db-path", str(temp_dir / "test_db")]
        )
        
        # Verify the result
        assert result.exit_code == 0
        assert "Welcome to SyntopicalChat" in result.stdout
        assert "Processing papers" in result.stdout
        assert "Successfully processed Test Paper" in result.stdout
        assert "Processing complete" in result.stdout
        assert "This is a test answer" in result.stdout
        assert "Goodbye" in result.stdout


@pytest.mark.unit
def test_start_command_arxiv_option(cli_runner, temp_dir, mock_openai_env, mock_arxiv_response):
    """Test the start command with arxiv option."""
    with patch("rich.prompt.Prompt.ask") as mock_ask, \
         patch("rich.prompt.IntPrompt.ask") as mock_int_ask, \
         patch.object(ArxivClient, "search_and_download") as mock_search_and_download, \
         patch.object(PDFProcessor, "process_pdf") as mock_process_pdf, \
         patch.object(VectorDBStorage, "add_paper") as mock_add_paper, \
         patch.object(SyntopicalChat, "chat") as mock_chat, \
         patch.object(typer, "prompt") as mock_prompt:
        # Mock the ask method to choose arxiv option and query
        mock_ask.side_effect = ["arxiv", "quantum computing"]
        
        # Mock the int_ask method to return max results
        mock_int_ask.return_value = 2
        
        # Mock the search_and_download method
        pdf_paths = [temp_dir / f"{paper['arxiv_id']}.pdf" for paper in mock_arxiv_response]
        mock_search_and_download.return_value = list(zip(mock_arxiv_response, pdf_paths))
        
        # Mock the process_pdf method
        mock_paper_contents = []
        for i, paper in enumerate(mock_arxiv_response):
            mock_paper_content = MagicMock()
            mock_paper_content.metadata.title = paper["title"]
            mock_paper_contents.append(mock_paper_content)
        mock_process_pdf.side_effect = mock_paper_contents
        
        # Mock the add_paper method
        mock_add_paper.side_effect = [["paper-1-0"], ["paper-2-0"]]
        
        # Mock the chat method
        mock_chat.return_value = {
            "answer": "This is a test answer.",
            "source_documents": [
                MagicMock(metadata={"title": "Test Paper 1"})
            ]
        }
        
        # Mock the prompt method to exit after one question
        mock_prompt.side_effect = ["What is the main topic?", "exit"]
        
        # Run the command
        result = cli_runner.invoke(
            app, 
            ["start", "--db-path", str(temp_dir / "test_db")]
        )
        
        # Verify the result
        assert result.exit_code == 0
        assert "Welcome to SyntopicalChat" in result.stdout
        assert "Searching Arxiv" in result.stdout
        assert "Downloaded" in result.stdout
        assert "Processing papers" in result.stdout
        assert "Processing complete" in result.stdout
        assert "This is a test answer" in result.stdout
        assert "Goodbye" in result.stdout


@pytest.mark.integration
def test_end_to_end_cli(cli_runner, sample_pdf, temp_dir, mock_openai_env):
    """Test end-to-end CLI functionality."""
    with patch.object(PDFProcessor, "process_pdf") as mock_process_pdf, \
         patch.object(VectorDBStorage, "add_paper") as mock_add_paper, \
         patch.object(VectorDBStorage, "get_all_papers") as mock_get_all_papers, \
         patch.object(SyntopicalChat, "chat") as mock_chat, \
         patch.object(SyntopicalChat, "analyze_topic") as mock_analyze_topic, \
         patch.object(typer, "prompt") as mock_prompt:
        # Mock the process_pdf method
        mock_paper_content = MagicMock()
        mock_paper_content.metadata.title = "Test Paper"
        mock_process_pdf.return_value = mock_paper_content
        
        # Mock the add_paper method
        mock_add_paper.return_value = ["test-paper-0"]
        
        # Mock the get_all_papers method
        mock_get_all_papers.return_value = [
            {
                "title": "Test Paper",
                "authors": "Author 1, Author 2",
                "publication_date": "2023-01-01",
            }
        ]
        
        # Mock the chat method
        mock_chat.return_value = {
            "answer": "This is a test answer.",
            "source_documents": [
                MagicMock(metadata={"title": "Test Paper"})
            ]
        }
        
        # Mock the analyze_topic method
        mock_analyze_topic.return_value = {
            "answer": "This is a test analysis.",
            "source_documents": [
                MagicMock(metadata={"title": "Test Paper"})
            ]
        }
        
        # Mock the prompt method to exit after one question
        mock_prompt.side_effect = ["What is the main topic?", "exit"]
        
        # Run the upload command
        upload_result = cli_runner.invoke(
            app, 
            ["upload", str(sample_pdf), "--db-path", str(temp_dir / "test_db")]
        )
        assert upload_result.exit_code == 0
        assert "Successfully processed Test Paper" in upload_result.stdout
        
        # Run the list command
        list_result = cli_runner.invoke(
            app, 
            ["list", "--db-path", str(temp_dir / "test_db")]
        )
        assert list_result.exit_code == 0
        assert "Test Paper" in list_result.stdout
        
        # Run the analyze command
        analyze_result = cli_runner.invoke(
            app, 
            ["analyze", "quantum computing", "--db-path", str(temp_dir / "test_db")]
        )
        assert analyze_result.exit_code == 0
        assert "This is a test analysis" in analyze_result.stdout
        
        # Run the chat command
        chat_result = cli_runner.invoke(
            app, 
            ["chat", "--db-path", str(temp_dir / "test_db")]
        )
        assert chat_result.exit_code == 0
        assert "This is a test answer" in chat_result.stdout