# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "libzim>=3.7,<4",
#   "beautifulsoup4>=4.12,<5",
#   "llama-cpp-python>=0.3,<0.4",
#   "numpy>=1.26,<3",
# ]
# ///
import heapq
import sqlite3
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama

DB = Path("/mnt/local_compute/RAG_dbs/wikibooks_rag_pipeline.db")
MODEL = Path("/mnt/local_compute/models/embeddings/nomic-embed-text-v1.5.Q8_0.gguf")
BATCH_SIZE = 4096
TOP_K = 10
THREADS = 4

QUERIES = [
    "How do I learn Python programming?",
    "What are the principles of organic gardening?",
    "How does a solar photovoltaic cell work?",
]

for path in (DB, MODEL):
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path}")

print(f"Database: {DB}")
print(f"Model: {MODEL}")
print(f"Embedding configuration: CPU-only, {THREADS} threads")
print("Loading embedding model...", flush=True)

start = time.perf_counter()
model = Llama(
    model_path=str(MODEL),
    embedding=True,
    n_ctx=2048,
    n_gpu_layers=0,
    n_threads=THREADS,
    verbose=False,
)
load_seconds = time.perf_counter() - start
print(f"Model initialization: {load_seconds:.2f}s", flush=True)

start = time.perf_counter()
response = model.create_embedding(
    input=[f"search_query: {query}" for query in QUERIES]
)
query_vectors = np.asarray(
    [item["embedding"] for item in response["data"]],
    dtype=np.float32,
)
embed_seconds = time.perf_counter() - start

if query_vectors.shape != (len(QUERIES), 768):
    raise SystemExit(f"Unexpected query-vector shape: {query_vectors.shape}")

query_norms = np.linalg.norm(query_vectors, axis=1)
if not np.isfinite(query_vectors).all() or np.any(query_norms == 0):
    raise SystemExit("Query embeddings contain invalid values.")

print(f"Query embedding: {embed_seconds:.2f}s for {len(QUERIES)} queries")
print("Opening database read-only...", flush=True)

con = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
con.execute("PRAGMA cache_size=-65536")
con.execute("PRAGMA mmap_size=536870912")

counts = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
print(f"Chunks to scan: {counts:,}")

heaps = [[] for _ in QUERIES]
seen = 0
invalid = 0
scan_start = time.perf_counter()

cursor = con.execute("""
    SELECT section, text, vector, vector_norm
    FROM chunks
""")

while True:
    rows = cursor.fetchmany(BATCH_SIZE)
    if not rows:
        break

    valid_rows = []
    vectors = []

    for section, text, blob, stored_norm in rows:
        vector = np.frombuffer(blob, dtype=np.float32)
        if (
            vector.size != 768
            or not np.isfinite(vector).all()
            or not np.isfinite(stored_norm)
            or stored_norm <= 0
        ):
            invalid += 1
            continue
        valid_rows.append((section, text, float(stored_norm)))
        vectors.append(vector)

    if vectors:
        matrix = np.vstack(vectors)
        scores = (matrix @ query_vectors.T) / (
            np.asarray([row[2] for row in valid_rows], dtype=np.float32)[:, None]
            * query_norms[None, :]
        )

        for row_index, (section, text, _) in enumerate(valid_rows):
            for query_index in range(len(QUERIES)):
                score = float(scores[row_index, query_index])
                if not np.isfinite(score):
                    continue

                heap = heaps[query_index]
                item = (score, seen, section or "", text or "")
                if len(heap) < TOP_K:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)

            seen += 1

scan_seconds = time.perf_counter() - scan_start
con.close()

print(f"\nVector scan and ranking: {scan_seconds:.2f}s")
print(f"Valid vectors processed: {seen:,}")
print(f"Invalid vectors skipped: {invalid:,}")
print(f"Throughput: {seen / scan_seconds:,.0f} vectors/s")
print(f"End-to-end time (excluding model initialization): "
      f"{embed_seconds + scan_seconds:.2f}s")

for query, heap in zip(QUERIES, heaps):
    print(f"\n{'=' * 78}\nQUERY: {query}\n{'=' * 78}")
    for rank, (score, _, section, text) in enumerate(
        sorted(heap, reverse=True), 1
    ):
        excerpt = " ".join(text.split())
        if len(excerpt) > 360:
            excerpt = excerpt[:357] + "..."
        print(f"\n{rank:2}. cosine={score:.5f} | section={section}")
        print(f"    {excerpt}")

model.close()
