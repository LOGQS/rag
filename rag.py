from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb


# How many lines of context get printed before/after every chunk
CONTEXT_LINES = 2         # keep in one place so logic stays consistent

# ────────────────────────────  INTERNAL HELPERS  ────────────────────────────
def _sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(persist_dir: Path) -> Dict[str, str]:
    """Load file hash manifest for incremental indexing."""
    manifest_file = persist_dir / "manifest.json"
    try:
        with manifest_file.open() as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_manifest(persist_dir: Path, manifest: Dict[str, str]) -> None:
    """Save file hash manifest."""
    manifest_file = persist_dir / "manifest.json"
    with manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)


def _get_parser_for_file(
    file_path: Path,
    token_chunk: int,          # for prose  (SentenceSplitter, tokens)
    line_chunk: int = 200,     # for code   (CodeSplitter,   lines)
    overlap: int =  200,        # shared     (tokens or lines)
):
    """Return a node-parser tuned per file type."""
    EXT2LANG = {
        ".py":  "python",
        ".js":  "javascript",
        ".ts":  "typescript",
        ".jsx": "javascript", 
        ".tsx": "typescript", 
        ".c":   "c",
        ".cpp": "cpp",
        ".cc":  "cpp",
        ".cxx": "cpp",
        ".h":   "c", 
        ".hpp": "cpp", 
        ".hxx": "cpp",
        ".java":"java",
        ".kt":  "kotlin",
        ".kts": "kotlin",
        ".scala":"scala",
        ".go":  "go",
        ".rs":  "rust",
        ".cs":  "csharp",
        ".php": "php",
        ".rb":  "ruby",
        ".swift":"swift",
        ".m":   "objective-c",
        ".mm":  "objective-c",
        ".pl":  "perl",
        ".pm":  "perl",
        ".sh":  "bash",
        ".bat": "bash",
        ".ps1": "powershell",
        ".lua": "lua",
        ".r":   "r",
        ".jl":  "julia",
        ".dart":"dart",
        ".groovy":"groovy",
        ".vb":  "visual-basic",
        ".vbs": "visual-basic",
        ".fs":  "fsharp",
        ".fsx": "fsharp",
        ".fsi": "fsharp",
        ".fsproj":"fsharp",
        ".sql": "sql",
        ".asm": "assembly",
        ".s":   "assembly",
        ".clj": "clojure",
        ".cljs":"clojure",
        ".cljc":"clojure",
        ".edn": "clojure",
        ".erl": "erlang",
        ".hrl": "erlang",
        ".ex":  "elixir",
        ".exs": "elixir",
        ".el":  "emacs-lisp",
        ".lisp":"commonlisp",
        ".scm": "scheme",
        ".ss":  "scheme",
        ".rkt": "racket",
        ".ml":  "ocaml",
        ".mli": "ocaml",
        ".ocaml":"ocaml",
        ".nim": "nim",
        ".d":   "d",
        ".vala":"vala",
        ".v":   "verilog",
        ".sv":  "verilog",
        ".svh": "verilog",
        ".verilog":"verilog",
        ".vhdl":"vhdl",
        ".ada": "ada",
        ".adb": "ada",
        ".ads": "ada",
        ".pas": "pascal",
        ".pp":  "pascal",
        ".inc": "pascal",
        ".tcl": "tcl",
        ".awk": "awk",
        ".psql":"sql",
        ".psm1":"powershell",
        ".psd1":"powershell",
    }
    lang = EXT2LANG.get(file_path.suffix.lower())
    if lang:
        # ❶  MUCH BIGGER chunks – default 400 lines, 64-line overlap
        return CodeSplitter(
            language=lang,
            chunk_lines=line_chunk,
            chunk_lines_overlap=overlap,
        )
    # ❷  Prose → regex tokenizer (thread-safe) with 2× bigger token window
    return SentenceSplitter(
        chunk_size=token_chunk * 2,
        chunk_overlap=overlap,
        tokenizer="regex",
    )


def _process_single_document(args):
    doc, token_chunk, overlap = args          # <-  renamed for clarity
    file_path = Path(doc.metadata.get("file_path", ""))
    parser = _get_parser_for_file(file_path, token_chunk, overlap)
    return parser.get_nodes_from_documents([doc])


def _hash_file_parallel(file_path: Path) -> tuple:
    """Hash a single file - designed for parallel execution."""
    try:
        current_hash = _sha256(file_path)
        return str(file_path), current_hash, None
    except Exception as e:
        return str(file_path), None, str(e)


# ────────────────────────────  A) INDEX  ────────────────────────────
def build_index(
    paths: List[str],
    persist_dir: str = "rag_index",
    chunk_size: int = 4096,
    overlap: int = 4096,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    incremental: bool = True,
    max_workers: int = None,
) -> VectorStoreIndex:
    """
    Turn a list of files (or directories) into a persisted Chroma vector index.

    paths        File or directory paths.
    persist_dir  Folder where Chroma's SQLite + HNSW binaries live.
    chunk_size   Token count per chunk for text files.
    overlap      Overlap between chunks for text files.
    embed_model  HuggingFace model name or 'openai', 'nomic-embed', etc.
    incremental  If True, only process changed files (based on SHA-256 hash).
    max_workers  Maximum number of parallel workers (default: CPU count).
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    pdir = Path(persist_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest for incremental processing
    manifest = _load_manifest(pdir) if incremental else {}
    
    # Resolve all target files
    all_files = []
    for path_str in paths:
        path = Path(path_str).resolve()
        if path.is_dir():
            # Recursively find all files
            all_files.extend([f for f in path.rglob("*") if f.is_file()])
        else:
            all_files.append(path)
    
    print(f"[build_index] Found {len(all_files)} files to check")
    
    # Parallel file hashing for incremental processing
    changed_files = []
    if incremental and all_files:
        print(f"[build_index] Checking file hashes in parallel with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all hash jobs
            hash_futures = {executor.submit(_hash_file_parallel, f): f for f in all_files}
            
            # Collect results
            for future in as_completed(hash_futures):
                file_path_str, current_hash, error = future.result()
                
                if error:
                    print(f"Warning: Could not hash {file_path_str}: {error}")
                    continue
                
                stored_hash = manifest.get(file_path_str)
                if stored_hash != current_hash:
                    changed_files.append(Path(file_path_str))
                    manifest[file_path_str] = current_hash
    else:
        # If not incremental, process all files
        changed_files = all_files
        for file_path in all_files:
            try:
                manifest[str(file_path)] = _sha256(file_path)
            except Exception as e:
                print(f"Warning: Could not hash {file_path}: {e}")
    
    # If no changes and index exists, return existing index
    if not changed_files and incremental:
        print(f"[build_index] No changes detected, loading existing index from {persist_dir}")
        try:
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection = chroma_client.get_or_create_collection("default")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
            return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_ctx)
        except Exception as e:
            print(f"Warning: Could not load existing index: {e}")
            print("Building fresh index...")
            changed_files = all_files  # Process all files
    
    if not changed_files:
        print("No files to process")
        return None
    
    print(f"[build_index] Processing {len(changed_files)} {'changed' if incremental else ''} files")
    
    # 1. Read documents for changed files
    docs = SimpleDirectoryReader(
        input_files=[str(f) for f in changed_files],
        recursive=False,  # We already resolved files
        exclude_hidden=True,
    ).load_data()

    # 2. Configure embedding model
    if embed_model.startswith("BAAI/") or embed_model.startswith("sentence-transformers/"):
        # Use better local embedding model with GPU support
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbedding(
                model_name=embed_model,
                device=device,
                trust_remote_code=True,
            )
            print(f"[build_index] Using {embed_model} on {device}")
        except ImportError:
            print("Warning: torch not available, falling back to CPU")
            embedding_model = HuggingFaceEmbedding(model_name=embed_model)
    else:
        # Handle string model names (OpenAI, etc.)
        if not embed_model.startswith(("local:", "openai", "nomic-embed")):
            embed_model = f"local:{embed_model}"
        embedding_model = embed_model
    
    Settings.embed_model = embedding_model
    
    # 3. Process documents in parallel
    print(f"[build_index] Parsing documents in parallel with {max_workers} workers...")
    all_nodes = []
    
    # Prepare arguments for parallel processing
    doc_args = [(doc, chunk_size, overlap) for doc in docs]
    
    # Use ThreadPoolExecutor for I/O-bound document parsing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all parsing jobs
        parse_futures = [executor.submit(_process_single_document, args) for args in doc_args]
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(parse_futures), 1):
            try:
                nodes = future.result()
                all_nodes.extend(nodes)
                if i % 10 == 0 or i == len(parse_futures):
                    print(f"[build_index] Processed {i}/{len(parse_futures)} documents")
            except Exception as e:
                print(f"Warning: Failed to process document: {e}")
    
    print(f"[build_index] Generated {len(all_nodes)} nodes from {len(docs)} documents")
    
    # 4. Set up Chroma vector store
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("default")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Create/update index (embedding generation happens here and is auto-parallelized by HF)
    print(f"[build_index] Creating embeddings and building index...")
    if incremental and manifest:
        # For incremental updates, add to existing index
        try:
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_ctx)
            # Insert new nodes (embeddings computed in parallel by HuggingFace)
            index.insert_nodes(all_nodes)
        except Exception as e:
            print(f"Could not update existing index: {e}, creating new one")
            index = VectorStoreIndex(nodes=all_nodes, storage_context=storage_ctx, show_progress=True)
    else:
        # Create fresh index (embeddings computed in parallel by HuggingFace)
        index = VectorStoreIndex(nodes=all_nodes, storage_context=storage_ctx, show_progress=True)
    
    # 6. Persist everything
    index.storage_context.persist()
    _save_manifest(pdir, manifest)
    
    print(f"[build_index] Successfully indexed {len(all_nodes)} chunks from {len(changed_files)} files")
    return index


# ────────────────────────────  B) RETRIEVE  ─────────────────────────
def retrieve_chunks(
    query: str,
    top_k: int = 5,
    persist_dir: str = "rag_index",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity: str = "cosine",
) -> List[Dict[str, Any]]:
    """
    Fetch the `top_k` most relevant chunks for `query`.
    Retrieval is already optimized and fast, no parallel processing needed.
    """
    # Configure embedding model (same logic as build_index)
    if embed_model.startswith("BAAI/") or embed_model.startswith("sentence-transformers/"):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbedding(
                model_name=embed_model,
                device=device,
                trust_remote_code=True,
            )
        except ImportError:
            embedding_model = HuggingFaceEmbedding(model_name=embed_model)
    else:
        if not embed_model.startswith(("local:", "openai", "nomic-embed")):
            embed_model = f"local:{embed_model}"
        embedding_model = embed_model
    
    Settings.embed_model = embedding_model
    
    # Load existing Chroma collection
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("default")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # Re-hydrate index and run retrieval
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_ctx)
    
    raw_k = max(top_k * 2, top_k + 5)      # over-sample a bit
    retriever = index.as_retriever(
        similarity_top_k=raw_k,
        similarity_measure=similarity,
    )
    nodes = retriever.retrieve(query)[:top_k]   # trim to requested size

    return [
        {
            "chunk": n.get_content(),
            "score": n.score,
            "source": n.node.metadata.get("file_path", "unknown"),
            "line_range": estimate_line_numbers(
                n.node.metadata.get("file_path", "unknown"),
                n.node.start_char_idx or 0,
                n.node.end_char_idx   or 0,
                CONTEXT_LINES,
            ),
            "start_char_idx": n.node.start_char_idx,
            "end_char_idx":   n.node.end_char_idx,
            "node_id": n.node.node_id,
            "all_metadata": dict(n.node.metadata),
        }
        for n in nodes
    ]


# ────────────────────────────  UTILITIES  ───────────────────────────
def estimate_line_numbers(
    file_path: str,
    start_char: int,
    end_char: int,
    context: int = CONTEXT_LINES,
) -> str:
    """
    Inclusive 1-based line numbers that will actually be printed,
    i.e. [chunk ± context].
    """
    try:
        cache = estimate_line_numbers.__dict__.setdefault("_cache", {})
        if file_path not in cache:
            cache[file_path] = Path(file_path).read_text(encoding="utf-8")
        text = cache[file_path]

        start_ln0 = text.count("\n", 0, start_char)       
        end_ln0   = text.count("\n", 0, end_char)

        ctx_start = max(0, start_ln0 - context)
        ctx_end   = end_ln0 + context
        # Return 1-based human numbers
        ctx_start_h = ctx_start + 1
        ctx_end_h   = ctx_end   + 1
        return f"{ctx_start_h}-{ctx_end_h}" if ctx_start_h != ctx_end_h else str(ctx_start_h)
    except Exception:
        return "unknown"


def get_chunk_with_context(
    file_path: str,
    start_char: int,
    end_char: int,
    context_lines: int = CONTEXT_LINES,
) -> str:
    """
    Build the display block, tagging:
      >>>   for the real chunk
            four spaces for context
    """
    try:
        text  = Path(file_path).read_text(encoding="utf-8")
        lines = text.splitlines()

        start_ln0 = text.count("\n", 0, start_char)          # 0-based index
        end_ln0   = text.count("\n", 0, end_char)

        ctx_s = max(0, start_ln0 - context_lines)
        ctx_e = min(len(lines) - 1, end_ln0 + context_lines)

        tagged = []
        for i in range(ctx_s, ctx_e + 1):
            prefix = ">>> " if start_ln0 <= i <= end_ln0 else "    "
            tagged.append(f"{prefix}{lines[i]}")
        return "\n".join(tagged)
    except Exception:
        return "Could not read original file context"
