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

## For using aider_chat agent in terminal
export LM_STUDIO_API_BASE=http://localhost:1234/v1
export LM_STUDIO_API_KEY=dummy-api-key
> aider --model lm_studio/qwen2.5-coder-32b-instruct --no-show-model-warnings



