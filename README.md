# Kiwix Search and Wikibooks Vector Benchmarks

## Search tool

`gpu-search.py` searches configured local Kiwix ZIM archives and can use
the external `gpu-ask` command to synthesize evidence-grounded answers.

Requirements:
- Python 3
- `libzim` Python bindings
- `gpu-ask` for model-assisted judging and synthesis (optional with `--no-llm`)

The default ZIM root is specific to the original author's machine. Override
it with `--zim-root /path/to/kiwix_zims`. The expected subdirectories and
archive filenames are configured in `gpu-search.py`.

## Wikibooks vector benchmarks

The scripts in `benchmarks/` evaluate retrieval over a local SQLite vector
database using `numpy` and `llama-cpp-python`. They currently use local
database and embedding-model paths; adjust those paths for your environment
before running.

The SQLite database, embedding model, and ZIM archives are intentionally
not included. Benchmark reports and relevance-judgment templates are kept
under `reports/` as experiment evidence.
