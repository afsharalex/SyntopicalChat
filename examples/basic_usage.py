#!/usr/bin/env python
"""Example script demonstrating basic usage of SyntopicalChat."""

import os
from pathlib import Path

from syntopicalchat.pdf_processor.processor import PDFProcessor
from syntopicalchat.vector_db.storage import VectorDBStorage
from syntopicalchat.llm.chat import SyntopicalChat


def main():
    """Run a basic example of SyntopicalChat."""
    print("SyntopicalChat Basic Usage Example")
    print("==================================")
    
    # Check if PDF files are provided
    pdf_dir = Path("examples/pdfs")
    if not pdf_dir.exists():
        print(f"Creating example PDF directory at {pdf_dir}")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please place some PDF files in {pdf_dir} and run this script again.")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}. Please add some PDF files and try again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files.")
    
    # Initialize components
    pdf_processor = PDFProcessor()
    vector_db = VectorDBStorage(persist_directory="examples/data/chroma_db")
    
    # Process PDFs
    print("\nProcessing PDFs:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
        paper_content = pdf_processor.process_pdf(pdf_file)
        vector_db.add_paper(paper_content)
        print(f"    Title: {paper_content.metadata.title}")
        print(f"    Authors: {', '.join(paper_content.metadata.authors)}")
        if paper_content.metadata.abstract:
            abstract = paper_content.metadata.abstract
            print(f"    Abstract: {abstract[:100]}..." if len(abstract) > 100 else abstract)
    
    # List all papers
    papers = vector_db.get_all_papers()
    print(f"\nStored {len(papers)} papers in the vector database.")
    
    # Check if OpenAI API key is set for chat functionality
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nNote: OPENAI_API_KEY environment variable is not set.")
        print("To use the chat functionality, please set this environment variable:")
        print("  export OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize chat
    chat = SyntopicalChat(vector_db=vector_db)
    
    # Example queries
    example_queries = [
        "What are the main topics covered in these papers?",
        "Compare the methodologies used in these papers.",
        "What are the key findings across these papers?",
    ]
    
    print("\nExample Queries:")
    for i, query in enumerate(example_queries, 1):
        print(f"\n{i}. {query}")
        response = chat.chat(query)
        print(f"\nResponse: {response['answer'][:200]}...")
        print("\nSources:")
        for j, doc in enumerate(response["source_documents"][:2], 1):
            print(f"  {j}. {doc.metadata.get('title', 'Unknown')}")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()