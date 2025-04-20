"""Tests for the LLM Chat module."""

import pytest
from unittest.mock import MagicMock, patch

from langchain.schema import Document
from syntopicalchat.llm.chat import SyntopicalChat


@pytest.mark.unit
def test_syntopical_chat_initialization(vector_db, mock_openai_env):
    """Test that SyntopicalChat can be initialized."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
        # Mock the ChatOpenAI class
        mock_chat_openai.return_value = MagicMock()
        
        # Initialize SyntopicalChat
        chat = SyntopicalChat(vector_db=vector_db)
        
        # Verify the initialization
        assert chat is not None
        assert chat.vector_db == vector_db
        assert chat.llm is not None
        assert chat.memory is not None
        assert chat.chain is not None


@pytest.mark.unit
def test_create_chain(vector_db, mock_openai_env):
    """Test creating a conversational retrieval chain."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai, \
         patch("langchain.chains.ConversationalRetrievalChain.from_llm") as mock_from_llm:
        # Mock the ChatOpenAI class
        mock_chat_openai.return_value = MagicMock()
        
        # Mock the from_llm method
        mock_chain = MagicMock()
        mock_from_llm.return_value = mock_chain
        
        # Initialize SyntopicalChat
        chat = SyntopicalChat(vector_db=vector_db)
        
        # Call the _create_chain method
        chain = chat._create_chain()
        
        # Verify the chain creation
        assert chain is mock_chain
        mock_from_llm.assert_called_once()


@pytest.mark.unit
def test_enhance_query(vector_db, mock_openai_env):
    """Test enhancing a query with syntopical analysis context."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
        # Mock the ChatOpenAI class
        mock_chat_openai.return_value = MagicMock()
        
        # Initialize SyntopicalChat
        chat = SyntopicalChat(vector_db=vector_db)
        
        # Enhance a query
        query = "What is the main topic?"
        enhanced_query = chat._enhance_query(query)
        
        # Verify the enhanced query
        assert query in enhanced_query
        assert "syntopical analysis" in enhanced_query.lower()
        assert "perspectives" in enhanced_query.lower()
        assert "methodologies" in enhanced_query.lower()


@pytest.mark.unit
def test_chat(vector_db, mock_openai_env):
    """Test chatting with the language model."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai, \
         patch("langchain.chains.ConversationalRetrievalChain.from_llm") as mock_from_llm:
        # Mock the ChatOpenAI class
        mock_chat_openai.return_value = MagicMock()
        
        # Mock the chain
        mock_chain = MagicMock()
        mock_chain.return_value = {
            "answer": "This is a test answer.",
            "source_documents": [
                Document(page_content="Test content", metadata={"title": "Test Paper"})
            ]
        }
        mock_from_llm.return_value = mock_chain
        
        # Initialize SyntopicalChat
        chat = SyntopicalChat(vector_db=vector_db)
        
        # Chat with the model
        response = chat.chat("What is the main topic?")
        
        # Verify the response
        assert "answer" in response
        assert response["answer"] == "This is a test answer."
        assert "source_documents" in response
        assert len(response["source_documents"]) == 1
        assert response["source_documents"][0].metadata["title"] == "Test Paper"


@pytest.mark.unit
def test_analyze_topic(vector_db, mock_openai_env):
    """Test analyzing a topic."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai, \
         patch("langchain.chains.ConversationalRetrievalChain.from_llm") as mock_from_llm:
        # Mock the ChatOpenAI class
        mock_chat_openai.return_value = MagicMock()
        
        # Mock the chain
        mock_chain = MagicMock()
        mock_chain.return_value = {
            "answer": "This is a test analysis.",
            "source_documents": [
                Document(page_content="Test content", metadata={"title": "Test Paper"})
            ]
        }
        mock_from_llm.return_value = mock_chain
        
        # Initialize SyntopicalChat
        chat = SyntopicalChat(vector_db=vector_db)
        
        # Analyze a topic
        response = chat.analyze_topic("quantum computing")
        
        # Verify the response
        assert "answer" in response
        assert response["answer"] == "This is a test analysis."
        assert "source_documents" in response
        assert len(response["source_documents"]) == 1
        assert response["source_documents"][0].metadata["title"] == "Test Paper"


@pytest.mark.unit
def test_reset_conversation(vector_db, mock_openai_env):
    """Test resetting the conversation history."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
        # Mock the ChatOpenAI class
        mock_chat_openai.return_value = MagicMock()
        
        # Initialize SyntopicalChat
        chat = SyntopicalChat(vector_db=vector_db)
        
        # Mock the memory
        chat.memory = MagicMock()
        
        # Reset the conversation
        chat.reset_conversation()
        
        # Verify the memory was cleared
        chat.memory.clear.assert_called_once()


@pytest.mark.integration
def test_end_to_end_chat(vector_db, sample_paper_content, mock_openai_env):
    """Test end-to-end chat functionality."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai, \
         patch("langchain.chains.ConversationalRetrievalChain.from_llm") as mock_from_llm:
        # Mock the ChatOpenAI class
        mock_chat_openai.return_value = MagicMock()
        
        # Mock the chain
        mock_chain = MagicMock()
        mock_chain.return_value = {
            "answer": "This is a test answer about the paper.",
            "source_documents": [
                Document(
                    page_content="Test content",
                    metadata={"title": sample_paper_content.metadata.title}
                )
            ]
        }
        mock_from_llm.return_value = mock_chain
        
        # Add the paper to the vector database
        vector_db.add_paper(sample_paper_content)
        
        # Initialize SyntopicalChat
        chat = SyntopicalChat(vector_db=vector_db)
        
        # Chat with the model
        response = chat.chat("What is the main topic of the paper?")
        
        # Verify the response
        assert "answer" in response
        assert "This is a test answer" in response["answer"]
        assert "source_documents" in response
        assert len(response["source_documents"]) == 1
        assert response["source_documents"][0].metadata["title"] == sample_paper_content.metadata.title
        
        # Analyze a topic
        response = chat.analyze_topic("test topic")
        
        # Verify the response
        assert "answer" in response
        assert "This is a test answer" in response["answer"]
        
        # Reset the conversation
        chat.reset_conversation()