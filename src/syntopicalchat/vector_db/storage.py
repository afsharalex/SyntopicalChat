"""Vector database storage implementation for document embeddings."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from syntopicalchat.pdf_processor.processor import PaperContent


class VectorDBStorage:
    """Storage for document embeddings using ChromaDB."""

    def __init__(
        self,
        persist_directory: Union[str, Path] = "data/chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector database storage.

        Args:
            persist_directory: Directory to persist the database.
            embedding_model_name: Name of the HuggingFace embedding model to use.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Initialize the vector store
        self.vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_function,
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def add_paper(self, paper: PaperContent) -> List[str]:
        """
        Add a paper to the vector database.

        Args:
            paper: Paper content to add.

        Returns:
            List of IDs for the added chunks.
        """
        # Create metadata for the document
        metadata = {
            "title": paper.metadata.title,
            "authors": ", ".join(paper.metadata.authors),
            "publication_date": paper.metadata.publication_date or "",
            "source_file": str(paper.metadata.source_file),
        }
        
        if paper.metadata.abstract:
            metadata["abstract"] = paper.metadata.abstract
            
        # Split the text into chunks
        texts = self.text_splitter.split_text(paper.text)
        
        # Create documents with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_id": f"{paper.metadata.title}-{i}",
                }
            )
            for i, chunk in enumerate(texts)
        ]
        
        # Add documents to the vector store
        ids = [f"{paper.metadata.title}-{i}" for i in range(len(texts))]
        self.vector_store.add_documents(documents, ids=ids)
        
        # Persist the database
        self.vector_store.persist()
        
        return ids

    def search(
        self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for documents similar to the query.

        Args:
            query: Query string.
            k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            List of documents similar to the query.
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata,
        )

    def get_all_papers(self) -> List[Dict]:
        """
        Get metadata for all papers in the database.

        Returns:
            List of paper metadata.
        """
        # Get all documents from the collection
        collection = self.vector_store._collection
        
        # Get unique paper titles
        results = collection.get(
            include=["metadatas"],
        )
        
        # Extract unique papers based on title
        papers = {}
        for metadata in results["metadatas"]:
            if metadata and "title" in metadata:
                title = metadata["title"]
                if title not in papers:
                    papers[title] = metadata
                    
        return list(papers.values())

    def delete_paper(self, title: str) -> None:
        """
        Delete a paper from the vector database.

        Args:
            title: Title of the paper to delete.
        """
        # Get the collection
        collection = self.vector_store._collection
        
        # Delete documents with matching title
        collection.delete(
            where={"title": title},
        )
        
        # Persist the database
        self.vector_store.persist()