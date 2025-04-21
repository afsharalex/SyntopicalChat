FROM python:3.12-slim

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create data directories
RUN mkdir -p /data/pdfs /data/chroma_db /data/arxiv_papers

# Set environment variables for data paths
ENV PDF_DIR=/data/pdfs \
    DB_PATH=/data/chroma_db \
    ARXIV_DIR=/data/arxiv_papers

# Set the entrypoint
ENTRYPOINT ["python", "-m", "syntopicalchat.cli.main"]
