# python-ai-rag
Python chat controller for AI RAG

## Quick start

1. Install deps: `pip install -r requirements.txt`
2. Set env: `export OPENAI_API_KEY=...`
3. Run: `uvicorn app.main:app --reload`
4. Open: `http://127.0.0.1:8000/docs`

## Structure

- `app/` FastAPI app and RAG service
- `data/` CSV data
- `storage/` SQLite vector store
