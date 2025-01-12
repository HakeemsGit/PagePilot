# Documentation Assistant

An AI-powered documentation assistant that helps scrape, process, and query documentation using RAG (Retrieval Augmented Generation).

## Features

- Documentation scraping and processing
- Embeddings generation and storage using Milvus
- RAG-based query system
- User-friendly Gradio interface

## Installation

This project uses `uv` for package management. To get started:

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

Run the Gradio interface:
```bash
python main.py
```
