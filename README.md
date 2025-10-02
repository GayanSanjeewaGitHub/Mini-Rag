Mini-RAG-Movies

A tiny end-to-end RAG (Retrieval-Augmented Generation) demo that lets you ask natural-language questions about Wikipedia movie plots.
It downloads the public “Wikipedia Movie Plots” dataset, chunks it, embeds the chunks with OpenAI text-embedding-3-small, stores them in Pinecone serverless, and answers questions with GPT-4o-mini.

Prerequisites

Python 3.9+
An OpenAI API key (embedding + chat)
A Pinecone API key (vector DB)

Create a .env file in the project root:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PINECONE_ENV=us-east-1-aws          # region shown in Pinecone console
PINECONE_INDEX=minirag       # will be created automatically if not availble

# 1. Clone / unzip the repo
cd Mini-Rag

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
requirements.txt (already in repo)
Copy
openai>=1.0.0
pinecone-client[grpc]>=3.0.0
pandas>=1.5.0
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0
kagglehub>=0.2.0


Ingest data (one-time)
The first run downloads the 30 MB Kaggle dataset automatically.

python App.py --csv data/wiki_movie_plots_deduped.csv \
              --rows 300 \
              --words_per_chunk 300 \
              --upsert


--rows – how many movies to index (≈ 5 k chunks for 300 rows)
--upsert must be present to trigger ingestion.


The command creates the Pinecone index (1536-dim, cosine) and uploads vectors in batches of 100.
When you see
INFO mini_rag - Upsert completed.
the index is ready


Chat / query
Now drop the --upsert flag and ask anything:
python App.py --query "Which movie features an AI antagonist?" --top_k 5

You’ll get a JSON reply:


{
  "answer": "The Terminator (1984) features Skynet, an AI that becomes self-aware and...",
  "reasoning": "Used snippets [1] and [3] which mention Skynet...",
  "contexts": [
    "In 2029 the artificial intelligence Skynet...",
    ...
  ]
}
Change the question any time:

python App.py --query "A movie where a chef falls in love" --top_k 3


Common options

| Flag                | Default                             | Purpose                         |
| ------------------- | ----------------------------------- | ------------------------------- |
| `--csv`             | `data/wiki_movie_plots_deduped.csv` | local CSV path                  |
| `--rows`            | 300                                 | #movies to sample (set 0 → all) |
| `--words_per_chunk` | 300                                 | chunk size                      |
| `--batch_size`      | 100                                 | upsert batch                    |
| `--top_k`           | 5                                   | retrieved chunks                |
| `--upsert`          | False                               | **Must be True for first run**  |



Tips & Troubleshooting
Permission denied – make sure you point --csv to the file, not the folder.
Non-ASCII id error – already fixed in code (Unicode → ASCII).
Rate limits – script auto-retries (3×) on OpenAI or Pinecone errors.
Re-indexing – simply run the --upsert command again; duplicates are overwritten by vector-id.
Full dataset – omit --rows or use --rows 0 to index everything (~35 k movies).
 
 
deactivate              # leave venv
rm -rf venv data        # remove env + cached dataset
Delete the Pinecone index from the web console if you no longer need it.