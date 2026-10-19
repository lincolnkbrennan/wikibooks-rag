# /// script
# requires-python = ">=3.11"
# dependencies = [
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
CANDIDATE_K = 100
MAX_PER_ENTRY = 2
THREADS = 4
VECTOR_DIM = 768
VECTOR_BYTES = VECTOR_DIM * np.dtype(np.float32).itemsize

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
print("Ranking: cosine similarity, max two chunks per source entry")
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

if query_vectors.shape != (len(QUERIES), VECTOR_DIM):
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
    SELECT
        c.chunk_id,
        e.entry_id,
        e.title,
        e.path,
        c.section,
        c.vector,
        c.vector_norm
    FROM chunks AS c
    JOIN entry_versions AS v ON v.version_id = c.version_id
    JOIN entries AS e ON e.entry_id = v.entry_id
""")

while True:
    rows = cursor.fetchmany(BATCH_SIZE)
    if not rows:
        break

    metadata = []
    blobs = []
    norms = []

    for chunk_id, entry_id, title, path, section, blob, stored_norm in rows:
        if blob is None or len(blob) != VECTOR_BYTES:
            invalid += 1
            continue
        try:
            norm = float(stored_norm)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not np.isfinite(norm) or norm <= 0:
            invalid += 1
            continue

        metadata.append((
            int(chunk_id),
            int(entry_id),
            title or "",
            path or "",
            section or "",
        ))
        blobs.append(blob)
        norms.append(norm)

    if not blobs:
        continue

    matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(
        len(blobs), VECTOR_DIM
    )
    norm_array = np.asarray(norms, dtype=np.float32)

    valid_mask = np.isfinite(matrix).all(axis=1)
    invalid += int((~valid_mask).sum())

    if not valid_mask.all():
        matrix = matrix[valid_mask]
        norm_array = norm_array[valid_mask]
        metadata = [
            row for row, is_valid in zip(metadata, valid_mask) if is_valid
        ]

    batch_count = len(metadata)
    if batch_count == 0:
        continue

    scores = (matrix @ query_vectors.T) / (
        norm_array[:, None] * query_norms[None, :]
    )

    for query_index in range(len(QUERIES)):
        column = scores[:, query_index]
        finite_indices = np.flatnonzero(np.isfinite(column))
        if finite_indices.size == 0:
            continue

        local_k = min(CANDIDATE_K, finite_indices.size)
        if finite_indices.size > local_k:
            chosen = finite_indices[
                np.argpartition(column[finite_indices], -local_k)[-local_k:]
            ]
        else:
            chosen = finite_indices

        heap = heaps[query_index]
        for index in chosen:
            score = float(column[index])
            chunk_id, entry_id, title, path, section = metadata[int(index)]
            item = (
                score,
                seen + int(index),
                chunk_id,
                entry_id,
                title,
                path,
                section,
            )

            if len(heap) < CANDIDATE_K:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    seen += batch_count

scan_seconds = time.perf_counter() - scan_start
print(f"\nVector scan and candidate ranking: {scan_seconds:.2f}s")
print(f"Valid vectors processed: {seen:,}")
print(f"Invalid vectors skipped: {invalid:,}")
print(f"Throughput: {seen / scan_seconds:,.0f} vectors/s")
print(
    f"End-to-end time (excluding model initialization): "
    f"{embed_seconds + scan_seconds:.2f}s"
)

for query, heap in zip(QUERIES, heaps):
    candidates = sorted(heap, reverse=True)
    selected = []
    counts_by_entry = {}

    for item in candidates:
        score, _, chunk_id, entry_id, title, path, section = item
        if counts_by_entry.get(entry_id, 0) >= MAX_PER_ENTRY:
            continue

        selected.append(item)
        counts_by_entry[entry_id] = counts_by_entry.get(entry_id, 0) + 1

        if len(selected) == TOP_K:
            break

    print(f"\n{'=' * 78}\nQUERY: {query}\n{'=' * 78}")
    print(
        f"Candidates retained: {len(candidates)}; "
        f"diversity limit: {MAX_PER_ENTRY} chunks per source entry"
    )

    for rank, (score, _, chunk_id, entry_id, title, path, section) in enumerate(
        selected, 1
    ):
        text = con.execute(
            "SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if text is None:
            raise SystemExit(f"Selected chunk disappeared: chunk_id={chunk_id}")
        excerpt = " ".join((text[0] or "").split())
        if len(excerpt) > 360:
            excerpt = excerpt[:357] + "..."

        print(f"\n{rank:2}. cosine={score:.5f} | source={title}")
        print(f"    path={path} | entry_id={entry_id} | section={section}")
        print(f"    {excerpt}")

con.close()
model.close()
