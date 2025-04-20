"""Tests for the Vector DB Storage module."""

import pytest
from pathlib import Path

from langchain.schema import Document
from syntopicalchat.vector_db.storage import VectorDBStorage
from syntopicalchat.pdf_processor.processor import PaperContent


@pytest.mark.unit
def test_vector_db_initialization(temp_dir):
    """Test that VectorDBStorage can be initialized."""
    db_path = temp_dir / "test_db"
    vector_db = VectorDBStorage(persist_directory=db_path)
    
    assert vector_db is not None
    assert vector_db.persist_directory == db_path
    assert vector_db.embedding_function is not None
    assert vector_db.vector_store is not None
    assert vector_db.text_splitter is not None


@pytest.mark.unit
def test_add_paper(vector_db, sample_paper_content):
    """Test adding a paper to the vector database."""
    # Add the paper to the database
    ids = vector_db.add_paper(sample_paper_content)
    
    # Verify the results
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(id, str) for id in ids)
    assert all(sample_paper_content.metadata.title in id for id in ids)


@pytest.mark.unit
def test_search(vector_db, sample_paper_content):
    """Test searching for documents in the vector database."""
    # Add the paper to the database
    vector_db.add_paper(sample_paper_content)
    
    # Search for documents
    results = vector_db.search("test abstract")
    
    # Verify the results
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(doc, Document) for doc in results)
    assert all(sample_paper_content.metadata.title in doc.metadata.get("title", "") for doc in results)


@pytest.mark.unit
def test_get_all_papers(vector_db, sample_paper_content):
    """Test getting all papers from the vector database."""
    # Add the paper to the database
    vector_db.add_paper(sample_paper_content)
    
    # Get all papers
    papers = vector_db.get_all_papers()
    
    # Verify the results
    assert isinstance(papers, list)
    assert len(papers) > 0
    assert all(isinstance(paper, dict) for paper in papers)
    assert any(sample_paper_content.metadata.title == paper.get("title", "") for paper in papers)


@pytest.mark.unit
def test_delete_paper(vector_db, sample_paper_content):
    """Test deleting a paper from the vector database."""
    # Add the paper to the database
    vector_db.add_paper(sample_paper_content)
    
    # Verify the paper is in the database
    papers_before = vector_db.get_all_papers()
    assert any(sample_paper_content.metadata.title == paper.get("title", "") for paper in papers_before)
    
    # Delete the paper
    vector_db.delete_paper(sample_paper_content.metadata.title)
    
    # Verify the paper is no longer in the database
    papers_after = vector_db.get_all_papers()
    assert not any(sample_paper_content.metadata.title == paper.get("title", "") for paper in papers_after)


@pytest.mark.integration
def test_end_to_end_vector_db(vector_db, sample_paper_content):
    """Test end-to-end vector database operations."""
    # Add the paper to the database
    ids = vector_db.add_paper(sample_paper_content)
    assert len(ids) > 0
    
    # Search for documents
    results = vector_db.search("test abstract")
    assert len(results) > 0
    
    # Get all papers
    papers = vector_db.get_all_papers()
    assert len(papers) > 0
    
    # Delete the paper
    vector_db.delete_paper(sample_paper_content.metadata.title)
    
    # Verify the paper is no longer in the database
    papers_after = vector_db.get_all_papers()
    assert not any(sample_paper_content.metadata.title == paper.get("title", "") for paper in papers_after)