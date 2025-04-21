"""Chat implementation for interacting with language models via Langchain."""

import os
from typing import Dict, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from syntopicalchat.vector_db.storage import VectorDBStorage


class SyntopicalChat:
    """Chat interface for syntopical analysis of academic papers."""

    def __init__(
        self,
        vector_db: VectorDBStorage,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize the chat interface.

        Args:
            vector_db: Vector database storage.
            model_name: Name of the OpenAI model to use.
            temperature: Temperature for the model.
            max_tokens: Maximum number of tokens to generate.
        """
        self.vector_db = vector_db

        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it to use the chat interface."
            )

        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # Initialize the retrieval chain
        self.chain = self._create_chain()

    def _create_chain(self) -> ConversationalRetrievalChain:
        """
        Create a conversational retrieval chain.

        Returns:
            Conversational retrieval chain.
        """
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            memory=self.memory,
            return_source_documents=True,
        )

    def chat(self, query: str) -> Dict:
        """
        Chat with the language model about the papers.

        Args:
            query: Query string.

        Returns:
            Response from the language model and source documents.
        """
        # Add syntopical analysis context to the query
        enhanced_query = self._enhance_query(query)

        # Get response from the chain
        response = self.chain({"question": enhanced_query})

        return {
            "answer": response["answer"],
            "source_documents": response["source_documents"],
        }

    def _enhance_query(self, query: str) -> str:
        """
        Enhance the query with syntopical analysis context.

        Args:
            query: Original query string.

        Returns:
            Enhanced query string.
        """
        # Add context for syntopical analysis
        return (
            f"Perform a syntopical analysis across multiple academic papers to answer: {query}\n"
            "Consider different perspectives, methodologies, and findings from all relevant papers. "
            "Identify agreements, disagreements, and complementary insights between the papers. "
            "Cite specific papers when referencing their content."
        )

    def analyze_topic(self, topic: str) -> Dict:
        """
        Perform a syntopical analysis on a specific topic.

        Args:
            topic: Topic to analyze.

        Returns:
            Analysis results.
        """
        # Create a structured analysis prompt
        analysis_prompt = (
            f"Perform a comprehensive syntopical analysis on the topic: '{topic}'\n\n"
            "Please structure your analysis as follows:\n"
            "1. Overview of the topic and its significance\n"
            "2. Key perspectives and approaches across the papers\n"
            "3. Major agreements between the papers\n"
            "4. Notable disagreements or contradictions\n"
            "5. Gaps in the literature and potential future research directions\n"
            "6. Synthesis of the most important insights\n\n"
            "For each point, cite specific papers and explain how they contribute to the understanding of the topic."
        )

        # Get response from the chain
        response = self.chat(analysis_prompt)

        return response

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.memory.clear()
