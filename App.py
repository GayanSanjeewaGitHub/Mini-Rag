"""
Mini RAG system (Movie Plots) - mini_rag.py

This single-file script implements a minimal Retrieval-Augmented Generation (RAG)
pipeline for answering questions about movie plots.

Features:
- Load a CSV of movie plots (expects columns: Title, Plot)
- Sample a subset (default 300 rows)
- Chunk long plots (~300 words per chunk)
- Embed chunks (uses OpenAI embeddings if OPENAI_API_KEY is set; otherwise uses sentence-transformers)
- Build an in-memory vector store (FAISS if available, otherwise sklearn NearestNeighbors brute force)
- Retrieve top-k relevant chunks for a query
- Generate an answer using an LLM (OpenAI if OPENAI_API_KEY provided, otherwise a simple template-based answer)
- Output structured JSON: { answer, contexts, reasoning }

Usage (example):
python mini_rag.py --csv wiki_movie_plots.csv --rows 300 --k 5 --query "Which movie features an AI antagonist?"

Dependencies:
pip install -r requirements.txt

requirements.txt (recommended):
faiss-cpu        # optional, speeds retrieval
sentence-transformers
scikit-learn
numpy
pandas
openai           # optional, for embeddings + LLM

Notes:
- If you want LLM generation, set OPENAI_API_KEY in env before running.
- The script is intentionally simple and clear for a take-home assignment.

"""

from __future__ import annotations
import argparse
import json
import os
import math
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Try optional imports (faiss, openai). We'll gracefully degrade if unavailable.
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBT = True
except Exception:
    _HAS_SBT = False

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# fallback from scikit-learn for nearest neighbors
from sklearn.neighbors import NearestNeighbors


# ----------------------------- Helpers ---------------------------------

def load_and_sample(csv_path: str, n: int = 300) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns Title and Plot (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get("title")
    plot_col = cols.get("plot")
    if title_col is None or plot_col is None:
        raise ValueError("CSV must contain 'Title' and 'Plot' columns (case-insensitive).")

    df = df[[title_col, plot_col]].rename(columns={title_col: "Title", plot_col: "Plot"})
    df = df.dropna(subset=["Plot"])  # drop rows with no plot
    if n and n < len(df):
        df = df.sample(n, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


def chunk_text(text: str, words_per_chunk: int = 300) -> List[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text.strip()]
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk]).strip()
        chunks.append(chunk)
    return chunks


def build_chunks(df: pd.DataFrame, words_per_chunk: int = 300) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (texts, metadata_list)
    metadata includes: title, chunk_index, original_row_index
    """
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        title = str(row["Title"]) if pd.notna(row["Title"]) else ""
        plot = str(row["Plot"]) if pd.notna(row["Plot"]) else ""
        piece_list = chunk_text(plot, words_per_chunk)
        for ci, piece in enumerate(piece_list):
            texts.append(piece)
            metas.append({"title": title, "chunk_index": ci, "row_index": int(idx)})
    return texts, metas


# --------------------------- Embeddings --------------------------------

class Embedder:
    def __init__(self, model_name: str | None = None):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = bool(self.openai_key) and _HAS_OPENAI
        if self.use_openai:
            openai.api_key = self.openai_key
            # model choice for embeddings
            self.model = model_name or "text-embedding-3-small"
            print("Using OpenAI embeddings ->", self.model)
        else:
            # fallback to sentence-transformers
            model_name = model_name or "all-MiniLM-L6-v2"
            if not _HAS_SBT:
                raise RuntimeError("No embedding backend available. Install openai or sentence-transformers.")
            self.sbert = SentenceTransformer(model_name)
            print("Using Sentence-Transformers embeddings ->", model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.use_openai:
            # Batch the requests (small batches)
            embeddings: List[List[float]] = []
            BATCH = 16
            for i in range(0, len(texts), BATCH):
                batch = texts[i : i + BATCH]
                resp = openai.Embedding.create(model=self.model, input=batch)
                batch_emb = [r["embedding"] for r in resp["data"]]
                embeddings.extend(batch_emb)
            return np.array(embeddings, dtype=np.float32)
        else:
            embs = self.sbert.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            # Ensure float32
            return embs.astype(np.float32)


# --------------------------- Vector Store -------------------------------

class VectorStore:
    def __init__(self, embeddings: np.ndarray, metas: List[Dict[str, Any]]):
        self.embeddings = embeddings
        self.metas = metas
        self.dim = embeddings.shape[1]

        if _HAS_FAISS:
            try:
                self.index = faiss.IndexFlatIP(self.dim)
                # normalize for cosine similarity
                faiss.normalize_L2(self.embeddings)
                self.index.add(self.embeddings)
                self._use_faiss = True
                print("FAISS index built.")
            except Exception as e:
                print("FAISS error, falling back:", e)
                self._use_faiss = False
                self._build_sklearn()
        else:
            self._use_faiss = False
            self._build_sklearn()

    def _build_sklearn(self):
        # sklearn NearestNeighbors with cosine metric
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.embeddings)
        print("Sklearn NearestNeighbors index built.")

    def query(self, q_emb: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        # q_emb shape (dim,) or (1,dim)
        q = q_emb.reshape(1, -1).astype(np.float32)
        if self._use_faiss:
            # faiss expects normalized vectors for IP
            faiss.normalize_L2(q)
            D, I = self.index.search(q, top_k)
            results = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                results.append((self.metas[idx], float(score)))
            return results
        else:
            # sklearn returns distances (cosine), convert to similarity
            dist, idxs = self.nn.kneighbors(q, n_neighbors=top_k)
            results = []
            for d, i in zip(dist[0].tolist(), idxs[0].tolist()):
                sim = 1 - d  # cosine similarity
                results.append((self.metas[i], float(sim)))
            return results


# ---------------------------- Retrieval ---------------------------------

def retrieve(query: str, embedder: Embedder, store: VectorStore, top_k: int = 5) -> List[Dict[str, Any]]:
    q_emb = embedder.embed([query])[0]
    hits = store.query(q_emb, top_k=top_k)
    # Build contexts (include snippet & metadata)
    contexts = []
    for meta, score in hits:
        contexts.append({"title": meta["title"], "chunk_index": meta["chunk_index"], "score": score})
    return contexts


# --------------------------- LLM Generation ------------------------------

def generate_answer_openai(query: str, contexts: List[Dict[str, Any]], texts: List[str]) -> Tuple[str, str]:
    """
    Use OpenAI ChatCompletion to generate answer + reasoning.
    contexts: list of metas; texts: all chunk texts in same order as metas
    Returns (answer, reasoning)
    """
    # assemble context snippets (include short excerpt from texts based on row_index & chunk_index)
    snippets = []
    for c in contexts:
        # attempt to find corresponding chunk text
        idx = None
        # find first matching meta in store
        for i, m in enumerate(store_metas_glob):
            if m["title"] == c["title"] and m["chunk_index"] == c["chunk_index"]:
                idx = i
                break
        if idx is None:
            snippets.append("")
        else:
            snippet_text = texts[idx]
            snippets.append(snippet_text)

    system_prompt = (
        "You are a helpful assistant that answers questions about movie plots. "
        "Use the provided context snippets from Wikipedia movie plots to form a concise, accurate answer. "
        "If not found, say you could not find a definitive answer."
    )

    user_prompt = f"Question: {query}\n\nContext snippets:\n"
    for i, s in enumerate(snippets, 1):
        user_prompt += f"[{i}] {s}\n\n"
    user_prompt += "\nPlease provide:\n1) a short answer (1-3 sentences)\n2) a short reasoning explaining which contexts you used.\nReturn ONLY a JSON with keys: answer, reasoning."

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # Attempt to parse JSON from model output
    try:
        parsed = json.loads(text)
        return parsed.get("answer", ""), parsed.get("reasoning", "")
    except Exception:
        # If not JSON, split heuristically
        parts = text.split("Reasoning:")
        ans = parts[0].strip()
        reason = parts[1].strip() if len(parts) > 1 else ""
        return ans, reason


def generate_answer_template(query: str, contexts: List[Dict[str, Any]], texts: List[str]) -> Tuple[str, str]:
    """Fallback: simple template-based answer using the top context."""
    if not contexts:
        return "I couldn't find relevant movie plot information.", "No matching contexts returned by retrieval."
    top = contexts[0]
    # find text index
    idx = None
    for i, m in enumerate(store_metas_glob):
        if m["title"] == top["title"] and m["chunk_index"] == top["chunk_index"]:
            idx = i
            break
    snippet = texts[idx] if idx is not None else ""
    # naive answer: mention title and snippet first sentence
    first_sentence = snippet.split(".")[0].strip()
    answer = f"The movie *{top['title']}* appears relevant. {first_sentence}."
    reasoning = f"Top context is from '{top['title']}' (score={top['score']:.3f}). Used that plot snippet to form the answer."
    return answer, reasoning


# ------------------------------- CLI -----------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV containing Title and Plot columns")
    p.add_argument("--rows", type=int, default=300, help="Number of rows to sample from CSV")
    p.add_argument("--words_per_chunk", type=int, default=300, help="Words per chunk for splitting plots")
    p.add_argument("--k", type=int, default=5, help="Top-k contexts to retrieve")
    p.add_argument("--query", type=str, required=True, help="Query/question to ask the RAG system")
    p.add_argument("--use_openai", action="store_true", help="Force using OpenAI for embeddings/LLM (requires OPENAI_API_KEY)")
    return p.parse_args()


def main():
    args = parse_args()

    df = load_and_sample(args.csv, n=args.rows)
    print(f"Loaded {len(df)} rows from {args.csv}")

    texts, metas = build_chunks(df, words_per_chunk=args.words_per_chunk)
    print(f"Created {len(texts)} chunks (words_per_chunk={args.words_per_chunk})")

    # initialize embedder
    embedder = Embedder()
    embs = embedder.embed(texts)
    print("Embeddings shape:", embs.shape)

    # build vector store
    store = VectorStore(embs, metas)

    # store globals for helper usage (a light hack for small script)
    global store_metas_glob
    store_metas_glob = metas

    # retrieval
    q_embs = embedder.embed([args.query])
    hits = store.query(q_embs[0], top_k=args.k)
    contexts = [{**meta, "score": score} for meta, score in hits]

    # Choose generation method
    if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            answer, reasoning = generate_answer_openai(args.query, contexts, texts)
        except Exception as e:
            print("OpenAI generation failed, falling back to template. Error:", e)
            answer, reasoning = generate_answer_template(args.query, contexts, texts)
    else:
        answer, reasoning = generate_answer_template(args.query, contexts, texts)

    # produce contexts as text snippets for output
    output_contexts = []
    for c in contexts:
        # find snippet
        idx = None
        for i, m in enumerate(metas):
            if m["title"] == c["title"] and m["chunk_index"] == c["chunk_index"]:
                idx = i
                break
        snippet_text = texts[idx] if idx is not None else ""
        output_contexts.append(snippet_text)

    out = {"answer": answer, "contexts": output_contexts, "reasoning": reasoning}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
