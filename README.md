# SyntopicalChat

A CLI application that enables syntopical analysis of academic papers using Large Language Models (LLMs).

## Overview

SyntopicalChat allows users to upload PDFs of related academic papers, which are then converted into embeddings and stored in a vector database. Users can then chat with an LLM via Langchain about the uploaded papers, performing a syntopical analysis of a given topic across multiple papers.

## Features

- Upload and process PDF documents containing academic papers
- Search and download papers from Arxiv based on query terms
- Extract text and metadata from PDFs
- Convert document content into vector embeddings
- Store embeddings in a vector database for efficient retrieval
- Chat interface for querying the content of multiple papers
- Syntopical analysis across papers on specific topics
- Interactive CLI with options for folder input or Arxiv search

## Installation

### Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)
- Docker and Docker Compose (optional, for containerized usage)

### Setup (Local Installation)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/syntopicalchat.git
   cd syntopicalchat
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Docker Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/syntopicalchat.git
   cd syntopicalchat
   ```

2. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Create the necessary directories for mounted volumes:
   ```bash
   mkdir -p data/pdfs data/chroma_db data/arxiv_papers
   ```

4. Build and run the Docker container:
   ```bash
   docker-compose up --build
   ```

   This will start the application in interactive mode, allowing you to chat with the LLM about academic papers.

## Usage

### Basic Commands (Local Installation)

```bash
# Start the interactive application (recommended)
# This will prompt you to choose between a folder with PDFs or Arxiv search
syntopicalchat start

# Upload a PDF to the system
syntopicalchat upload path/to/paper.pdf

# Upload multiple PDFs at once
syntopicalchat upload path/to/paper1.pdf path/to/paper2.pdf

# List all uploaded papers
syntopicalchat list

# Start a chat session about the uploaded papers
syntopicalchat chat

# Perform a syntopical analysis on a specific topic
syntopicalchat analyze "quantum computing applications in cryptography"
```

### Docker Usage

When using Docker, you can place your PDF files in the `data/pdfs` directory, which is mounted as a volume in the container. This allows the application to access your files without copying them into the container.

```bash
# Start the interactive application (default command)
docker-compose up

# Run a specific command
docker-compose run --rm syntopicalchat upload /data/pdfs/paper.pdf

# List all uploaded papers
docker-compose run --rm syntopicalchat list

# Start a chat session about the uploaded papers
docker-compose run --rm syntopicalchat chat

# Perform a syntopical analysis on a specific topic
docker-compose run --rm syntopicalchat analyze "quantum computing applications in cryptography"
```

### Using the Mounted Volumes

The Docker setup includes three mounted volumes:

1. `./data/pdfs:/data/pdfs` - Place your PDF files in the `data/pdfs` directory on your host machine to make them available to the application.
2. `./data/chroma_db:/data/chroma_db` - The vector database is stored here, allowing persistence between container runs.
3. `./data/arxiv_papers:/data/arxiv_papers` - Papers downloaded from Arxiv are stored here.

Example workflow with Docker:

1. Place your academic papers in the `data/pdfs` directory
2. Run the application: `docker-compose up`
3. When prompted for a folder path, use the default `/data/pdfs`
4. Chat with the LLM about your papers

### Example Sessions

#### Using the Interactive Start Command

```
$ syntopicalchat start
Welcome to SyntopicalChat!
This application allows you to chat with an LLM about academic papers.

Would you like to specify a folder with PDFs or search Arxiv? [folder/arxiv]: arxiv

Enter your Arxiv search query: quantum computing
Enter the maximum number of papers to download [5]: 3

Searching Arxiv for quantum computing...
Downloaded 3 papers from Arxiv.

Processing papers and adding to vector database...
Processing /data/arxiv_papers/2311.12226v1.pdf...
✅ Successfully processed Quantum Computing: A Gentle Introduction

Processing /data/arxiv_papers/2207.02102v1.pdf...
✅ Successfully processed Quantum Machine Learning: A Tutorial

Processing /data/arxiv_papers/2310.01272v1.pdf...
✅ Successfully processed Applications of Quantum Computing in Cryptography

Processing complete!

You can now chat with the LLM about the papers.
Type 'exit' or 'quit' to end the session.

> What are the main applications of quantum computing discussed in these papers?
The papers discuss several key applications of quantum computing:

1. Cryptography and Security:
   - Quantum key distribution for secure communication
   - Post-quantum cryptographic algorithms resistant to quantum attacks
   - Breaking classical encryption methods using Shor's algorithm

2. Machine Learning:
   - Quantum neural networks with potential speedups for training
   - Quantum support vector machines for classification tasks
   - Quantum principal component analysis for dimensionality reduction

3. Optimization Problems:
   - Solving complex optimization problems using quantum annealing
   - Quantum approximate optimization algorithms
   - Applications in logistics, finance, and resource allocation

The papers emphasize that while many of these applications are still theoretical or in early experimental stages, they represent significant potential advantages over classical computing approaches.

Sources:
1. Applications of Quantum Computing in Cryptography
2. Quantum Machine Learning: A Tutorial
3. Quantum Computing: A Gentle Introduction
```

#### Using Individual Commands

```
$ syntopicalchat upload paper1.pdf paper2.pdf paper3.pdf
Successfully uploaded 3 papers to the system.

$ syntopicalchat chat
Welcome to SyntopicalChat! You can now ask questions about the papers you've uploaded.

> What are the main methodologies used across these papers?
The papers primarily use three methodologies:
1. Experimental design with control groups (Papers 1 and 3)
2. Computational modeling (Paper 2)
3. Statistical analysis of large datasets (Papers 1, 2, and 3)

Each paper applies these methods differently, with Paper 1 focusing on...
```

## Project Structure

```
syntopicalchat/
├── src/
│   └── syntopicalchat/
│       ├── pdf_processor/       # PDF handling and text extraction
│       ├── vector_db/           # Vector database operations
│       ├── llm/                 # LLM integration via Langchain
│       ├── arxiv_integration/   # Arxiv search and download functionality
│       └── cli/                 # Command-line interface
├── tests/                       # Unit and integration tests
├── data/                        # Data directory (created when running)
│   ├── pdfs/                    # Directory for PDF files (mounted volume in Docker)
│   ├── chroma_db/               # Vector database storage (mounted volume in Docker)
│   └── arxiv_papers/            # Downloaded Arxiv papers (mounted volume in Docker)
├── Dockerfile                   # Docker configuration for containerization
├── docker-compose.yml           # Docker Compose configuration for easy deployment
├── pyproject.toml               # Project configuration and dependencies
└── README.md                    # This file
```

## License

[MIT License](LICENSE)

## Testing

SyntopicalChat uses pytest for testing. To run the tests:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=syntopicalchat

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

Test files are organized by module in the `tests/` directory:

- `test_pdf_processor.py`: Tests for PDF processing functionality
- `test_vector_db.py`: Tests for vector database operations
- `test_llm_chat.py`: Tests for LLM chat functionality
- `test_arxiv_integration.py`: Tests for Arxiv integration
- `test_cli.py`: Tests for command-line interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

When contributing code, please ensure that you:

1. Add tests for any new functionality
2. Maintain or improve test coverage
3. Update documentation as needed
