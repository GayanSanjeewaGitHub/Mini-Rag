#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone ,ServerlessSpec
from dotenv import load_dotenv
import openai
import kagglehub
import shutil
import unicodedata

# External SDKs
load_dotenv() 
LOG = logging.getLogger("mini_rag")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "mini-rag-movie")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # adjust to available models
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))

if not OPENAI_API_KEY:
    LOG.error("OPENAI_API_KEY environment variable not set.")
    raise SystemExit(1)
if not PINECONE_API_KEY:
    LOG.error("PINECONE_API_KEY environment variable not set.")
    raise SystemExit(1)

openai.api_key = OPENAI_API_KEY


try:
    import openai
except Exception as e:
    raise RuntimeError("openai library is required. Install with `pip install openai`.") from e


pinecone=''
try:
    pinecone = Pinecone(api_key="PINECONE_API_KEY")
except Exception as e:
    raise RuntimeError("pinecone library is required add it please. `.") from e



 
def retry(fn=None, *, retries=3, delay=1.0, backoff=2.0):
    """Simple retry decorator (sync)."""
    def deco(func):
        def wrapper(*args, **kwargs):
            _retries = retries
            _delay = delay
            last_exc = None
            for i in range(_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    LOG.warning("Retry %d/%d after exception: %s", i + 1, _retries, e)
                    time.sleep(_delay)
                    _delay *= backoff
            LOG.error("All retries failed for function %s", func.__name__)
            raise last_exc
        return wrapper
    return deco if fn is None else deco(fn)

def chunk_text(text: str, words_per_chunk: int = 300) -> List[str]:
    """Split text on word boundaries into ~words_per_chunk chunks."""
    if not text:
        return []
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text.strip()]
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk]).strip()
        chunks.append(chunk)
    return chunks

 
@dataclass
class PineconeConfig:
    api_key: str
    environment: str
    index_name: str
    dimension: int = 1536

class PineconeStore:
    def __init__(self, cfg: PineconeConfig):
        self.cfg = cfg
        # Initialize Pinecone client
        self.client = Pinecone(
            api_key=cfg.api_key,
            environment=cfg.environment
        )
        LOG.info("Connected to Pinecone environment=%s", cfg.environment)

        # Ensure the index exists
        self.index_name = self._ensure_index(cfg.index_name, cfg.dimension)

        # Get the Index object for upsert/query
        self._index = self.client.Index(cfg.index_name)

    def _ensure_index(self, name: str, dimension: int):
        # print(self.client.list_indexes())
        """Create index if it doesn't exist using ServerlessSpec."""
        try:
            if name not in self.client.list_indexes()[0]["name"]:
                LOG.info("Creating Pinecone index '%s' (dim=%d)", name, dimension)
                spec = ServerlessSpec(
                    # No need to specify region or cloud here; client already has it
                    #replicas=1\
                    cloud="aws",
                    region='us-east-1'
                    
                )
                self.client.create_index(
                    name=name,
                    dimension=dimension,
                    metric="cosine",
                    spec=spec
                )
                # Wait briefly for index to be ready
                time.sleep(2)
            else:
                LOG.info("Pinecone index '%s' already exists", name)
            return name
        except Exception:
            LOG.exception("Failed to ensure index present")
            raise

    @retry(retries=3, delay=1.0)
    def upsert(self, vectors: List[Tuple[str, List[float], dict]]):
        """Upsert list of (id, vector, metadata)."""
        try:
            LOG.info("Upserting %d vectors to Pinecone index %s", len(vectors), self.cfg.index_name)
            self._index.upsert(vectors=vectors)
        except Exception:
            LOG.exception("Pinecone upsert error")
            raise

    @retry(retries=3, delay=1.0)
    def query(self, vector: List[float], top_k: int = 5) -> List[dict]:
        """Query Pinecone index for top_k nearest neighbors. Returns list of {id, score, metadata}."""
        try:
            res = self._index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            matches = res.get("matches", [])
            out = [{"id": m["id"], "score": m.get("score", 0.0), "metadata": m.get("metadata", {})} for m in matches]
            return out
        except Exception:
            LOG.exception("Pinecone query error")
            raise

 
@retry(retries=3, delay=1.0)
def embed_texts(texts: List[str], model: str = OPENAI_EMBED_MODEL) -> List[List[float]]:
    """Call OpenAI embeddings API in batches."""
    LOG.info("Embedding %d texts (model=%s)", len(texts), model)
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        try:
            resp = openai.Embedding.create(model=model, input=batch)
        except Exception:
            LOG.exception("OpenAI embedding call failed")
            raise
        for item in resp["data"]:
            embeddings.append(item["embedding"])
    return embeddings

@retry(retries=3, delay=1.0)
def generate_answer_with_references(query: str, snippets: List[str], model: str = OPENAI_CHAT_MODEL) -> Tuple[str, str]:
    """
    Generate answer and reasoning. We craft a deterministic system+user prompt and ask model to return only JSON.
    Returns (answer, reasoning)
    """
    LOG.info("Generating answer via OpenAI chat model=%s", model)
    system_msg = (
        "You are a helpful assistant that answers questions about movie plots. "
        "Use ONLY the provided context snippets (do not hallucinate). "
        "If the answer is not contained in the snippets, be explicit and say you couldn't find it."
    )
    user_msg = f"Question: {query}\n\nContext snippets:\n"
    for i, s in enumerate(snippets, start=1):
        user_msg += f"[{i}] {s}\n\n"
    user_msg += (
        "Please provide a JSON object with keys: answer (one-3 sentences), reasoning (short explanation mentioning which snippets were used). "
        "Return ONLY the JSON, and ensure it is parseable."
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception:
        LOG.exception("OpenAI chat completion failed")
        raise

    text = resp["choices"][0]["message"]["content"].strip()
    # Try parsing JSON; otherwise fallback simple heuristics
    try:
        parsed = json.loads(text)
        answer = parsed.get("answer", "")
        reasoning = parsed.get("reasoning", "")
        return answer, reasoning
    except Exception:
        LOG.warning("Chat model did not return JSON - falling back to heuristic parsing")
        # naive split: first paragraph = answer, rest = reasoning
        parts = text.split("\n\n")
        answer = parts[0].strip()
        reasoning = "\n\n".join(parts[1:]).strip() if len(parts) > 1 else ""
        return answer, reasoning




def ensure_dataset(local_dir: str = "data"):
    """
    Ensure dataset exists locally. If the folder is empty, download from KaggleHub.
    """
    # Create local data dir if not exists
    os.makedirs(local_dir, exist_ok=True)

    # Check if the folder has files
    if not os.listdir(local_dir):
        print(" 'data/' folder is empty. Downloading dataset...")
        path = kagglehub.dataset_download("jrobischon/wikipedia-movie-plots")
        print(" Dataset downloaded to:", path)

        # Copy dataset into local_dir
        for item in os.listdir(path):
            src = os.path.join(path, item)
            dest = os.path.join(local_dir, item)

            if os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)

        print(f"📥 Dataset copied into '{local_dir}'")
    else:
        print(f"✅ Found existing dataset in '{local_dir}'")

 
 
def build_chunks_from_csv(csv_path: str, sample_rows: int = 300, words_per_chunk: int = 300) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Loads CSV, samples rows, chunks plot texts, returns list of chunk texts and matching metadata.
    metadata: { title, row_index, chunk_index, original_plot_preview }
    """
    LOG.info("Loading CSV: %s (sample=%s)", csv_path, sample_rows)
    ensure_dataset("data")
    df = pd.read_csv(csv_path)
    # find title and plot columns case-insensitive
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get("title")
    plot_col = cols.get("plot")
    if title_col is None or plot_col is None:
        LOG.error("CSV must contain 'Title' and 'Plot' columns (case-insensitive). Found: %s", list(df.columns))
        raise ValueError("CSV must contain Title and Plot columns")

    df = df[[title_col, plot_col]].rename(columns={title_col: "Title", plot_col: "Plot"})
    df = df.dropna(subset=["Plot"])
    if sample_rows and sample_rows < len(df):
        df = df.sample(sample_rows, random_state=42).reset_index(drop=True)
    texts = []
    metas = []
    for idx, row in df.iterrows():
        title = str(row["Title"]) if pd.notna(row["Title"]) else ""
        plot = str(row["Plot"])
        chunks = chunk_text(plot, words_per_chunk=words_per_chunk)
        for ci, c in enumerate(chunks):
            texts.append(c)
            metas.append({"title": title, "row_index": int(idx), "chunk_index": int(ci), "preview": c[:240]})
    LOG.info("Built %d chunks from %d rows", len(texts), len(df))
    return texts, metas

def upsert_chunks_to_pinecone(store: PineconeStore, texts: List[str], metas: List[Dict[str, Any]], batch_size: int = 100):
    """Compute embeddings and upsert into Pinecone in batches."""
    LOG.info("Upserting chunks to Pinecone (batches of %d)", batch_size)
    # compute embeddings in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Upsert batches"):
        batch_texts = texts[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]
        emb_batch = embed_texts(batch_texts)
        vectors = []
        for j, emb in enumerate(emb_batch):
            meta = batch_metas[j]
            # create a stable id: e.g., title_row_chunk
            # ensure id length limits for pinecone (should be safe)
            safe_title = unicodedata.normalize('NFKD', meta["title"]) \
                       .encode('ascii', 'ignore') \
                       .decode('ascii') \
                       .replace(" ", "_")[:50]
            vec_id = f"{safe_title}_r{meta['row_index']}_c{meta['chunk_index']}"
            vectors.append((vec_id, emb, meta))
        store.upsert(vectors)

def retrieve_contexts(store: PineconeStore, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Embed query, query Pinecone, and return ordered list of contexts (text + metadata + score)."""
    q_emb = embed_texts([query])[0]
    matches = store.query(q_emb, top_k=top_k)
    # matches have metadata; we will return metadata plus score
    LOG.info("Retrieved %d matches", len(matches))
    return matches

 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, help="Path to wiki_movie_plots CSV" ,default="data/wiki_movie_plots_deduped.csv")
    p.add_argument("--rows", type=int, default=300, help="How many rows to sample for building index")
    p.add_argument("--words_per_chunk", type=int, default=300, help="Words per chunk")
    p.add_argument("--upsert", action="store_true", help="Run build+upsert into Pinecone" , default=False)
    p.add_argument("--query", type=str, help="Query to ask the RAG system (run after index exists)" , default="Which movie features an AI antagonist?")
    p.add_argument("--top_k", type=int, default=5, help="Top-k contexts to retrieve")
    p.add_argument("--batch_size", type=int, default=100, help="Upsert batch size")
    return p.parse_args()

def main():
    args = parse_args()
    # Validate env
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        LOG.error("OPENAI_API_KEY and PINECONE_API_KEY must be set.")
        sys.exit(2)

    # initialize PineconeStore
    # dimension: we assume 1536 for text-embedding-3-small. If you change model, update dimension accordingly.
    cfg = PineconeConfig(api_key=PINECONE_API_KEY, environment=PINECONE_ENV or "", index_name=PINECONE_INDEX, dimension=1536)
    store = PineconeStore(cfg)

    # If user requested upsert: build chunks and upsert to pinecone
    if args.upsert:
        if not args.csv:
            LOG.error("--csv must be provided when --upsert is used.")
            sys.exit(2)
        texts, metas = build_chunks_from_csv(args.csv, sample_rows=args.rows, words_per_chunk=args.words_per_chunk)
        upsert_chunks_to_pinecone(store, texts, metas, batch_size=args.batch_size)
        LOG.info("Upsert completed.")
        return

    # If user provided a query, retrieve and generate
    if args.query:
        LOG.info("Running retrieval for query: %s", args.query)
        matches = retrieve_contexts(store, args.query, top_k=args.top_k)
        # Extract snippets
        snippets = []
        for m in matches:
            meta = m.get("metadata", {})
            # If we stored preview in metadata use it, or use metadata keys to recompose
            snippet = meta.get("preview") or meta.get("text") or ""
            snippets.append(snippet)
        # Generate answer using OpenAI chat
        if not snippets:
            LOG.warning("No snippets returned from Pinecone - returning no-answer JSON")
            out = {"answer": "", "contexts": [], "reasoning": "No matches found in vector store."}
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return
        try:
            answer, reasoning = generate_answer_with_references(args.query, snippets, model=OPENAI_CHAT_MODEL)
        except Exception as e:
            LOG.exception("LLM generation failed; falling back to simple template")
            # Fallback: mention the top match title if exists
            top_meta = matches[0].get("metadata", {})
            top_title = top_meta.get("title", "Unknown")
            answer = f"I found plot snippets for '{top_title}', but could not generate a final answer due to an LLM error."
            reasoning = f"Top match: {top_title}. Error: {str(e)}"

        out = {"answer": answer, "contexts": snippets, "reasoning": reasoning}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    LOG.info("No action provided. Use --upsert to build index, or --query to run retrieval + answer.")
    LOG.info("Example: python mini_rag_pinecone.py --csv wiki_movie_plots.csv --rows 300 --upsert")
    LOG.info("Then: python mini_rag_pinecone.py --query 'Which movie features an AI antagonist?' --top_k 5")

if __name__ == "__main__":
    main()
