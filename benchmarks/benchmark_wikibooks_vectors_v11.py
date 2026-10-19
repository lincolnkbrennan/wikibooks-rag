# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "llama-cpp-python>=0.3,<0.4",
#   "numpy>=1.26,<3",
# ]
# ///
import csv
import heapq
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from llama_cpp import Llama

DB = Path("/mnt/local_compute/RAG_dbs/wikibooks_rag_pipeline.db")
MODEL = Path("/mnt/local_compute/models/embeddings/nomic-embed-text-v1.5.Q8_0.gguf")

BATCH_SIZE = 4096
TOP_K = 10
CANDIDATE_K = 500
MAX_PER_ENTRY = 2
THREADS = 4
VECTOR_DIM = 768
VECTOR_BYTES = VECTOR_DIM * np.dtype(np.float32).itemsize

JUDGMENT_LABELS = (
    "directly answers",
    "partially relevant",
    "related but insufficient",
    "irrelevant",
)

QUERIES = [
    "How do SQL indexes improve database query performance?",
    "How does fermentation cause bread dough to rise?",
    "How do bicycle gears change mechanical advantage?",
]


def normalize_source(title, path):
    """Heuristically group obvious print-version and canonical variants."""
    normalized_title = re.sub(
        r"\s*/\s*print\s+version\s*$",
        "",
        title.strip(),
        flags=re.IGNORECASE,
    )
    normalized_title = re.sub(
        r"\s*\(\s*print\s+version\s*\)\s*$",
        "",
        normalized_title,
        flags=re.IGNORECASE,
    )
    normalized_title = re.sub(r"\s+", " ", normalized_title).strip().casefold()

    normalized_path = re.sub(
        r"/print[_ ]version/?$",
        "",
        path.strip(),
        flags=re.IGNORECASE,
    ).rstrip("/").casefold()

    # Title-based grouping is intentionally heuristic: it may miss
    # duplicates or merge distinct entries with identical titles.
    return normalized_title or normalized_path


def retrieve_candidates(con, query_vectors, query_norms):
    """Scan all stored vectors and retain a deterministic candidate pool."""
    heaps = [[] for _ in QUERIES]
    valid = 0
    invalid = 0
    scan_start = time.perf_counter()

    cursor = con.execute("""
        SELECT
            c.chunk_id,
            c.version_id,
            c.chunk_index,
            v.entry_id,
            e.title,
            e.path,
            c.section,
            c.text,
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

        for (
            chunk_id, version_id, chunk_index, entry_id,
            title, path, section, text, blob, stored_norm
        ) in rows:
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
                int(version_id),
                int(chunk_index),
                int(entry_id),
                title or "",
                path or "",
                section or "",
                text or "",
            ))
            blobs.append(blob)
            norms.append(norm)

        if not blobs:
            continue

        matrix = np.frombuffer(
            b"".join(blobs), dtype=np.float32
        ).reshape(len(blobs), VECTOR_DIM)
        norm_array = np.asarray(norms, dtype=np.float32)

        valid_mask = np.isfinite(matrix).all(axis=1)
        invalid += int((~valid_mask).sum())

        if not valid_mask.all():
            matrix = matrix[valid_mask]
            norm_array = norm_array[valid_mask]
            metadata = [
                row
                for row, is_valid in zip(metadata, valid_mask)
                if is_valid
            ]

        batch_count = len(metadata)
        valid += batch_count
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
                    np.argpartition(
                        column[finite_indices], -local_k
                    )[-local_k:]
                ]
            else:
                chosen = finite_indices

            heap = heaps[query_index]

            for index in chosen:
                (
                    chunk_id, version_id, chunk_index, entry_id,
                    title, path, section, text,
                ) = metadata[int(index)]
                score = float(column[index])

                # chunk_id breaks score ties deterministically.
                # Negative IDs make the heap root the worst retained item.
                item = (
                    score,
                    -chunk_id,
                    chunk_id,
                    version_id,
                    chunk_index,
                    entry_id,
                    title,
                    path,
                    section,
                    text,
                )

                if len(heap) < CANDIDATE_K:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)

    scan_seconds = time.perf_counter() - scan_start

    candidates_by_query = [
        sorted(heap, key=lambda item: (-item[0], item[2]))
        for heap in heaps
    ]

    return candidates_by_query, valid, invalid, scan_seconds


def diversify(candidates):
    """
    Limit chunks per entry and suppress normalized-title duplicates.
    This is an experimental heuristic, not a verified source identity map.
    """
    selected = []
    counts_by_entry = {}
    seen_source_groups = set()

    for original_rank, item in enumerate(candidates, 1):
        (
            score, _, chunk_id, version_id, chunk_index, entry_id,
            title, path, section, text,
        ) = item

        if counts_by_entry.get(entry_id, 0) >= MAX_PER_ENTRY:
            continue

        source_group = normalize_source(title, path)
        if source_group in seen_source_groups:
            continue

        selected.append((original_rank, item))
        counts_by_entry[entry_id] = counts_by_entry.get(entry_id, 0) + 1
        seen_source_groups.add(source_group)

        if len(selected) == TOP_K:
            break

    return selected


def excerpt_for(text, limit=360):
    excerpt = " ".join((text or "").split())
    if len(excerpt) > limit:
        return excerpt[:limit - 3] + "..."
    return excerpt


def print_result_list(label, candidates, diversified=False):
    print(f"\n{label}")

    if diversified:
        results = diversify(candidates)
    else:
        results = [
            (rank, item)
            for rank, item in enumerate(candidates[:TOP_K], 1)
        ]

    if not results:
        print("  No candidates.")
        return []

    for output_rank, (original_rank, item) in enumerate(results, 1):
        (
            score, _, chunk_id, version_id, chunk_index, entry_id,
            title, path, section, text,
        ) = item

        print(
            f"\n  {output_rank:2}. cosine={score:.5f}"
            f" | original_candidate_rank={original_rank}"
        )
        print(
            f"      chunk_id={chunk_id}"
            f" | version_id={version_id}"
            f" | chunk_index={chunk_index}"
        )
        print(f"      entry_id={entry_id} | source={title}")
        print(f"      path={path} | section={section}")
        print(f"      {excerpt_for(text)}")

    return results


def write_judgment_template(all_candidates, output_path):
    fields = [
        "query",
        "diversified_rank",
        "original_candidate_rank",
        "cosine_similarity",
        "chunk_id",
        "version_id",
        "chunk_index",
        "entry_id",
        "title",
        "path",
        "section",
        "excerpt",
        "relevance_judgment",
        "review_notes",
    ]

    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for query, candidates in zip(QUERIES, all_candidates):
            for diversified_rank, (original_rank, item) in enumerate(
                diversify(candidates), 1
            ):
                (
                    score, _, chunk_id, version_id, chunk_index, entry_id,
                    title, path, section, text,
                ) = item

                writer.writerow({
                    "query": query,
                    "diversified_rank": diversified_rank,
                    "original_candidate_rank": original_rank,
                    "cosine_similarity": f"{score:.8f}",
                    "chunk_id": chunk_id,
                    "version_id": version_id,
                    "chunk_index": chunk_index,
                    "entry_id": entry_id,
                    "title": title,
                    "path": path,
                    "section": section,
                    "excerpt": " ".join(text.split()),
                    "relevance_judgment": "",
                    "review_notes": "",
                })


def main():
    for path in (DB, MODEL):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    judgment_path = Path(
        f"wikibooks_vector_v11_relevance_judgments_{timestamp}.tsv"
    )

    if judgment_path.exists():
        raise SystemExit(
            f"WARNING: judgment file already exists: {judgment_path}"
        )

    print(f"Database: {DB}")
    print(f"Model: {MODEL}")
    print(f"Embedding configuration: CPU-only, {THREADS} threads")
    print(f"Candidate pool per query: {CANDIDATE_K}")
    print(f"Final result count per query: {TOP_K}")
    print(f"Entry diversity limit: {MAX_PER_ENTRY} chunks per entry")
    print("Ranking: cosine similarity; original scores are preserved")
    print("Source diversity: heuristic normalized-title grouping")
    print("Repeatability: two independent full database scans")
    print("Database access: read-only")
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

    try:
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
            raise SystemExit(
                f"Unexpected query-vector shape: {query_vectors.shape}"
            )

        query_norms = np.linalg.norm(query_vectors, axis=1)
        if (
            not np.isfinite(query_vectors).all()
            or not np.isfinite(query_norms).all()
            or np.any(query_norms <= 0)
        ):
            raise SystemExit("Query embeddings contain invalid values.")

        con = sqlite3.connect(
            DB.resolve().as_uri() + "?mode=ro",
            uri=True,
        )

        try:
            con.execute("PRAGMA cache_size=-65536")
            con.execute("PRAGMA mmap_size=536870912")

            total_chunks = con.execute(
                "SELECT count(*) FROM chunks"
            ).fetchone()[0]

            print(f"Chunks to scan per pass: {total_chunks:,}")
            print(
                f"Query embedding: {embed_seconds:.2f}s"
                f" for {len(QUERIES)} queries"
            )

            run_results = []
            run_stats = []

            for run_number in (1, 2):
                print(
                    f"\nStarting retrieval pass {run_number}/2...",
                    flush=True,
                )
                candidates, valid, invalid, scan_seconds = (
                    retrieve_candidates(con, query_vectors, query_norms)
                )

                run_results.append(candidates)
                run_stats.append((valid, invalid, scan_seconds))

                print(
                    f"Pass {run_number}: {scan_seconds:.2f}s; "
                    f"valid vectors={valid:,}; invalid vectors={invalid:,}"
                )

                if valid + invalid != total_chunks:
                    raise RuntimeError(
                        f"Pass {run_number} counted {valid + invalid:,} "
                        f"rows, but chunks table contains {total_chunks:,}."
                    )

            print("\n=== Repeatability comparison ===")
            all_exact = True

            for query_index, query in enumerate(QUERIES):
                first = run_results[0][query_index]
                second = run_results[1][query_index]

                first_pool_ids = [item[2] for item in first]
                second_pool_ids = [item[2] for item in second]

                first_top_ids = first_pool_ids[:TOP_K]
                second_top_ids = second_pool_ids[:TOP_K]

                first_diverse_ids = [
                    item[2] for _, item in diversify(first)
                ]
                second_diverse_ids = [
                    item[2] for _, item in diversify(second)
                ]

                pool_match = first_pool_ids == second_pool_ids
                top_match = first_top_ids == second_top_ids
                diverse_match = first_diverse_ids == second_diverse_ids

                all_exact = (
                    all_exact and pool_match and top_match and diverse_match
                )

                print(f"\nQuery: {query}")
                print(
                    f"  Ordered candidate-pool IDs: "
                    f"{'EXACT MATCH' if pool_match else 'DIFFER'} "
                    f"({len(first_pool_ids)} vs {len(second_pool_ids)})"
                )
                print(
                    f"  Original top-{TOP_K} IDs: "
                    f"{'EXACT MATCH' if top_match else 'DIFFER'}"
                )
                print(
                    f"  Diversified top-{TOP_K} IDs: "
                    f"{'EXACT MATCH' if diverse_match else 'DIFFER'}"
                )

                if not pool_match:
                    for position, (left, right) in enumerate(
                        zip(first_pool_ids, second_pool_ids), 1
                    ):
                        if left != right:
                            print(
                                f"  First pool difference at rank {position}: "
                                f"{left} vs {right}"
                            )
                            break

                print_result_list(
                    "Original top 10 (no source diversification)",
                    first,
                    diversified=False,
                )
                print_result_list(
                    "Diversified top 10 (original ranks and scores preserved)",
                    first,
                    diversified=True,
                )

            print("\n=== Repeatability summary ===")
            print(
                "Exact agreement across candidate pools, original top 10, "
                "and diversified top 10: "
                f"{'YES' if all_exact else 'NO'}"
            )

            write_judgment_template(run_results[0], judgment_path)
            print(f"\nRelevance judgment template: {judgment_path}")
            print("Judgment values are blank for manual review.")
            print("Allowed labels:")
            for label in JUDGMENT_LABELS:
                print(f"  - {label}")

            print("\n=== Timing summary ===")
            print(f"Model initialization: {load_seconds:.2f}s")
            print(f"Query embedding: {embed_seconds:.2f}s")
            print(
                "Two-pass vector scan time: "
                f"{sum(item[2] for item in run_stats):.2f}s"
            )
            print(
                "Total elapsed time excluding model initialization: "
                f"{embed_seconds + sum(item[2] for item in run_stats):.2f}s"
            )

        finally:
            con.close()

    finally:
        model.close()


if __name__ == "__main__":
    main()
