# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Python FastAPI application implementing a RAG (Retrieval-Augmented Generation) system for movie data. The system uses ChromaDB as a vector database, OpenAI embeddings for semantic search, and GPT models for answer generation.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variable
export OPENAI_API_KEY=your_api_key_here
```

### Running the Application
```bash
# Start the FastAPI server with auto-reload
uvicorn app.main:app --reload

# Access the interactive API documentation
# http://127.0.0.1:8000/docs
```

### Using Aider (LM Studio Integration)
```bash
export LM_STUDIO_API_BASE=http://localhost:1234/v1
export LM_STUDIO_API_KEY=dummy-api-key
aider --model lm_studio/qwen2.5-coder-32b-instruct --no-show-model-warnings --edit-format diff
```

## Architecture

### Core Components

**RAGService** (`app/rag.py`)
- Orchestrates the RAG pipeline: CSV ingestion → embedding → vector search → answer generation
- On first query, automatically indexes CSV data if ChromaDB is empty
- Handles question answering by retrieving top-k relevant movie chunks and generating contextual responses

**DatabaseService** (`app/database.py`)
- Wraps ChromaDB persistent client for vector storage and retrieval
- Automatically builds index on first use by embedding all movie data from CSV
- Stores movie metadata alongside embeddings for efficient retrieval

**AIService** (`app/services.py`)
- Interfaces with OpenAI API for embeddings (`text-embedding-3-small`) and chat completions (`gpt-4o-mini`)
- Batches embedding requests (100 texts per batch) for efficiency
- Generates answers using RAG context with temperature=0 for consistency

### Data Flow

1. CSV data (`data/movies500Trimmed.csv`) contains movie metadata: title, genres, overview, release_date, runtime, credits
2. On first query, each movie row is converted to JSON text and embedded using OpenAI
3. Embeddings and metadata are stored in ChromaDB (`storage/chroma_db/`)
4. User questions are embedded and matched against the vector database
5. Top-k relevant movie chunks are formatted as context for GPT answer generation

### Configuration

All settings are centralized in `app/config.py` using Pydantic Settings:
- Loads from `.env` file (if present) or environment variables
- Key settings: `openai_api_key`, `csv_path`, `chroma_path`, `embedding_model`, `chat_model`, `top_k`
- Defaults: ChromaDB stored in `storage/chroma_db/`, CSV at `data/movies500Trimmed.csv`, top_k=5

### API Structure

Single endpoint exposed via FastAPI:
- `POST /chat` - accepts `{"question": "..."}`, returns `{"answer": "..."}`
- Controller instantiates global `AIService` and `RAGService` on startup
- Handles exceptions by returning 500 with error details

## Key Implementation Details

- **Vector Database**: ChromaDB is used in persistent mode. The `storage/` directory is gitignored and created on first run.
- **Lazy Indexing**: The database checks if the collection is empty before indexing. Delete `storage/` to rebuild the index.
- **CSV Encoding**: The CSV file uses `latin-1` encoding, specified in the reader.
- **Movie ID Uniqueness**: IDs in ChromaDB are generated as `{movie_id}_{index}` to ensure uniqueness.
- **Empty Embedding Handling**: Search results handle missing embeddings by returning empty numpy arrays.

## Project Structure

```
app/
├── main.py          # FastAPI app initialization and router registration
├── controllers.py   # API endpoints and request/response models
├── rag.py          # RAG orchestration and CSV processing
├── services.py     # OpenAI API integration (embeddings + chat)
├── database.py     # ChromaDB vector database operations
├── models.py       # MovieChunk dataclass definition
└── config.py       # Centralized settings with Pydantic

data/               # CSV movie datasets (latin-1 encoding)
storage/           # ChromaDB persistent storage (gitignored)
```
