"""Main CLI entry point for the SyntopicalChat application."""

import os
import glob
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt

from syntopicalchat.pdf_processor.processor import PDFProcessor
from syntopicalchat.vector_db.storage import VectorDBStorage
from syntopicalchat.llm.chat import SyntopicalChat
from syntopicalchat.arxiv_integration.arxiv_client import ArxivClient

# Create Typer app
app = typer.Typer(
    name="syntopicalchat",
    help="A CLI application for syntopical analysis of academic papers using LLMs.",
    add_completion=False,
)

# Create console for rich output
console = Console()


@app.callback()
def callback():
    """SyntopicalChat - Analyze academic papers using LLMs."""
    pass


@app.command()
def upload(
    pdf_paths: List[Path] = typer.Argument(
        ..., help="Paths to PDF files to upload", exists=True
    ),
    db_path: Path = typer.Option(
        "data/chroma_db", "--db-path", "-d", help="Path to the vector database"
    ),
):
    """Upload PDF files to the system."""
    console.print(Panel("Uploading PDF files...", title="SyntopicalChat"))

    # Initialize PDF processor
    pdf_processor = PDFProcessor()

    # Initialize vector database
    vector_db = VectorDBStorage(persist_directory=db_path)

    # Process each PDF file
    for pdf_path in pdf_paths:
        try:
            console.print(f"Processing [bold]{pdf_path}[/bold]...")

            # Process PDF
            paper_content = pdf_processor.process_pdf(pdf_path)

            # Add to vector database
            vector_db.add_paper(paper_content)

            console.print(f"✅ Successfully processed [bold]{paper_content.metadata.title}[/bold]")

        except Exception as e:
            console.print(f"❌ Error processing {pdf_path}: {str(e)}", style="bold red")

    console.print(Panel("Upload complete!", title="SyntopicalChat"))


@app.command()
def list(
    db_path: Path = typer.Option(
        "data/chroma_db", "--db-path", "-d", help="Path to the vector database"
    ),
):
    """List all uploaded papers."""
    console.print(Panel("Listing uploaded papers...", title="SyntopicalChat"))

    # Initialize vector database
    vector_db = VectorDBStorage(persist_directory=db_path)

    # Get all papers
    papers = vector_db.get_all_papers()

    if not papers:
        console.print("No papers found in the database.")
        return

    # Create table
    table = Table(title="Uploaded Papers")
    table.add_column("Title", style="cyan")
    table.add_column("Authors", style="green")
    table.add_column("Date", style="yellow")

    # Add rows
    for paper in papers:
        table.add_row(
            paper.get("title", "Unknown"),
            paper.get("authors", "Unknown"),
            paper.get("publication_date", "Unknown"),
        )

    console.print(table)


@app.command()
def chat(
    db_path: Path = typer.Option(
        "data/chroma_db", "--db-path", "-d", help="Path to the vector database"
    ),
    model: str = typer.Option(
        "gpt-3.5-turbo", "--model", "-m", help="OpenAI model to use"
    ),
):
    """Start a chat session about the uploaded papers."""
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "❌ OPENAI_API_KEY environment variable is not set. "
            "Please set it to use the chat interface.",
            style="bold red",
        )
        return

    console.print(Panel(
        "Welcome to SyntopicalChat! You can now ask questions about the papers you've uploaded.\n"
        "Type 'exit' or 'quit' to end the session.",
        title="SyntopicalChat"
    ))

    # Initialize vector database
    vector_db = VectorDBStorage(persist_directory=db_path)

    # Initialize chat
    chat = SyntopicalChat(vector_db=vector_db, model_name=model)

    # Start chat loop
    while True:
        # Get user input
        user_input = typer.prompt("\n> ")

        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            console.print("Goodbye! 👋")
            break

        try:
            with console.status("Thinking...", spinner="dots"):
                # Get response
                response = chat.chat(user_input)

            # Print response
            console.print(Markdown(response["answer"]))

            # Print sources
            if response["source_documents"]:
                console.print("\n[bold]Sources:[/bold]")
                for i, doc in enumerate(response["source_documents"][:3], 1):
                    title = doc.metadata.get("title", "Unknown")
                    console.print(f"{i}. {title}")

        except Exception as e:
            console.print(f"❌ Error: {str(e)}", style="bold red")


@app.command()
def analyze(
    topic: str = typer.Argument(..., help="Topic to analyze"),
    db_path: Path = typer.Option(
        "data/chroma_db", "--db-path", "-d", help="Path to the vector database"
    ),
    model: str = typer.Option(
        "gpt-3.5-turbo", "--model", "-m", help="OpenAI model to use"
    ),
):
    """Perform a syntopical analysis on a specific topic."""
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "❌ OPENAI_API_KEY environment variable is not set. "
            "Please set it to use the chat interface.",
            style="bold red",
        )
        return

    console.print(Panel(
        f"Performing syntopical analysis on topic: [bold]{topic}[/bold]",
        title="SyntopicalChat"
    ))

    # Initialize vector database
    vector_db = VectorDBStorage(persist_directory=db_path)

    # Initialize chat
    chat = SyntopicalChat(vector_db=vector_db, model_name=model)

    try:
        with console.status("Analyzing...", spinner="dots"):
            # Get analysis
            response = chat.analyze_topic(topic)

        # Print analysis
        console.print(Markdown(response["answer"]))

        # Print sources
        if response["source_documents"]:
            console.print("\n[bold]Sources:[/bold]")
            for i, doc in enumerate(response["source_documents"][:5], 1):
                title = doc.metadata.get("title", "Unknown")
                console.print(f"{i}. {title}")

    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="bold red")


@app.command()
def start(
    db_path: Path = typer.Option(
        os.environ.get("DB_PATH", "data/chroma_db"), "--db-path", "-d", help="Path to the vector database"
    ),
    model: str = typer.Option(
        "gpt-3.5-turbo", "--model", "-m", help="OpenAI model to use"
    ),
):
    """
    Start the SyntopicalChat application.

    This command provides an interactive interface to:
    1. Choose between specifying a folder with PDFs or using Arxiv
    2. Process the selected papers
    3. Start a chat session about the papers
    """
    console.print(Panel(
        "Welcome to SyntopicalChat!\n"
        "This application allows you to chat with an LLM about academic papers.",
        title="SyntopicalChat"
    ))

    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "❌ OPENAI_API_KEY environment variable is not set. "
            "Please set it to use the chat interface.",
            style="bold red",
        )
        return

    # Initialize components
    pdf_processor = PDFProcessor()
    vector_db = VectorDBStorage(persist_directory=db_path)

    # Prompt user to choose between folder and Arxiv
    choice = Prompt.ask(
        "Would you like to [bold]specify a folder[/bold] with PDFs or [bold]search Arxiv[/bold]?",
        choices=["folder", "arxiv"],
        default="folder"
    )

    pdf_paths = []

    if choice.lower() == "folder":
        # Prompt for folder path with default from environment
        default_pdf_dir = os.environ.get("PDF_DIR", "data/pdfs")
        folder_path = Prompt.ask("Enter the path to the folder containing PDFs", default=default_pdf_dir)
        folder_path = Path(folder_path).expanduser().resolve()

        if not folder_path.exists() or not folder_path.is_dir():
            console.print(f"❌ Folder {folder_path} does not exist or is not a directory.", style="bold red")
            return

        # Find all PDFs in the folder
        pdf_paths = list(folder_path.glob("*.pdf"))

        if not pdf_paths:
            console.print(f"❌ No PDF files found in {folder_path}.", style="bold red")
            return

        console.print(f"Found [bold]{len(pdf_paths)}[/bold] PDF files in {folder_path}.")

    else:  # Arxiv
        # Prompt for Arxiv query
        query = Prompt.ask("Enter your Arxiv search query")
        max_results = IntPrompt.ask("Enter the maximum number of papers to download", default=5)

        console.print(f"Searching Arxiv for [bold]{query}[/bold]...")

        # Initialize Arxiv client with download directory from environment
        arxiv_dir = os.environ.get("ARXIV_DIR", "data/arxiv_papers")
        arxiv_client = ArxivClient(download_dir=Path(arxiv_dir))

        # Search and download papers
        with console.status("Searching and downloading papers from Arxiv...", spinner="dots"):
            results = arxiv_client.search_and_download(query, max_results)

        if not results:
            console.print("❌ No papers found on Arxiv matching your query.", style="bold red")
            return

        console.print(f"Downloaded [bold]{len(results)}[/bold] papers from Arxiv.")

        # Extract PDF paths
        pdf_paths = [pdf_path for _, pdf_path in results]

    # Process PDFs and add to vector database
    console.print(Panel("Processing papers and adding to vector database...", title="SyntopicalChat"))

    for pdf_path in pdf_paths:
        try:
            console.print(f"Processing [bold]{pdf_path}[/bold]...")

            # Process PDF
            paper_content = pdf_processor.process_pdf(pdf_path)

            # Add to vector database
            vector_db.add_paper(paper_content)

            console.print(f"✅ Successfully processed [bold]{paper_content.metadata.title}[/bold]")

        except Exception as e:
            console.print(f"❌ Error processing {pdf_path}: {str(e)}", style="bold red")

    console.print(Panel("Processing complete!", title="SyntopicalChat"))

    # Start chat session
    console.print(Panel(
        "You can now chat with the LLM about the papers.\n"
        "Type 'exit' or 'quit' to end the session.",
        title="SyntopicalChat"
    ))

    # Initialize chat
    chat_interface = SyntopicalChat(vector_db=vector_db, model_name=model)

    # Start chat loop
    while True:
        # Get user input
        user_input = typer.prompt("\n> ")

        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            console.print("Goodbye! 👋")
            break

        try:
            with console.status("Thinking...", spinner="dots"):
                # Get response
                response = chat_interface.chat(user_input)

            # Print response
            console.print(Markdown(response["answer"]))

            # Print sources
            if response["source_documents"]:
                console.print("\n[bold]Sources:[/bold]")
                for i, doc in enumerate(response["source_documents"][:3], 1):
                    title = doc.metadata.get("title", "Unknown")
                    console.print(f"{i}. {title}")

        except Exception as e:
            console.print(f"❌ Error: {str(e)}", style="bold red")


if __name__ == "__main__":
    app()
