# Mini-Rag

## Setup

```bash
git clone <your-repo>
cd mini-rag

# create uv-managed venv
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# install dependencies
uv pip install -r pyproject.toml

"""

Minimal, production-minded RAG pipeline for Movie Plots using:
- OpenAI for embeddings & chat completions
- Pinecone as vector DB

Features:
- Load subset of CSV (Title, Plot)
- Chunk long plots (words_per_chunk)
- Create / upsert chunks to Pinecone with metadata
- Retrieve top-k relevant chunks for a query
- Generate JSON output: { answer, contexts, reasoning }
- Logging, retries, error handling

Usage:
python mini_rag_pinecone.py --csv wiki_movie_plots.csv --rows 300 --upsert
python mini_rag_pinecone.py --query "Which movie features an AI antagonist?" --top_k 5
"""

 
 