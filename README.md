# Mini-Rag


pip install sentence-transformers scikit-learn pandas numpy
# optional but recommended for speed/LLM:
pip install faiss-cpu openai



export OPENAI_API_KEY=sk-...   # optional, only if you want OpenAI
python mini_rag.py --csv path/to/wiki_movie_plots.csv --rows 300 --k 5 --query "Which movie features an AI antagonist?"
# Mini RAG with uv

This is a minimal Retrieval-Augmented Generation (RAG) system for movie plots.

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

