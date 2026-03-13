#!/usr/bin/env python3
"""
RAG CLI chatbot: load a JSON/JSONL knowledge base, embed with Ollama, retrieve top-k,
and stream answers from Gemma 3:1b with source citations.

Usage:
  python rag_chat.py
  # Embeddings are cached by default (next to KB as {stem}_embeddings.jsonl). First run builds and saves; later runs load from cache.
  python rag_chat.py --kb devtac_rag.jsonl --top-k 5
  python rag_chat.py --no-cache          # do not use cache (always build embeddings)
  python rag_chat.py --regenerate        # ignore cache and rebuild embeddings
  python rag_chat.py --no-stream
"""

import argparse
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Avoid UnicodeEncodeError on Windows when printing sources with special chars
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace")

try:
    import ollama
except ImportError:
    print("Error: pip install ollama", file=sys.stderr)
    sys.exit(1)


SYSTEM_PROMPT_TEMPLATE = """You are a professional support assistant. Be clear and helpful without being stiff. Use a neutral, business-appropriate tone: no emojis, no slang, no overly casual phrases like "Hey!" or "pretty cool." You can add a brief line of context or a relevant follow-up when useful. Cite sources with [1], [2], etc. If the answer is not in the context, say so and suggest they reach out for more information. Answer ONLY from the context below; do not make things up.

Context:
{context}"""


def load_kb(path: Path) -> list[dict]:
    """Load knowledge base from .json (array) or .jsonl."""
    with open(path, encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            docs = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))
            return docs
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "documents" in data:
        return data["documents"]
    return [data]


def ensure_embeddings(
    docs: list[dict],
    embed_model: str,
    batch_size: int = 10,
) -> None:
    """Fill doc['embedding'] for each doc that lacks it (in-place)."""
    to_embed = [(i, d) for i, d in enumerate(docs) if not d.get("embedding")]
    if not to_embed:
        return
    indices = [i for i, _ in to_embed]
    texts = [docs[i]["text_for_embedding"] or docs[i].get("text", "") for i in indices]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out = ollama.embed(model=embed_model, input=batch)
        embs = out.get("embeddings", [])
        for j, idx in enumerate(indices[start : start + batch_size]):
            if j < len(embs):
                docs[idx]["embedding"] = embs[j]


def _cache_meta_path(cache_path: Path) -> Path:
    """Path for metadata file (generated_at, embed_model)."""
    return cache_path.parent / (cache_path.stem + "_meta.json")


def load_cache_meta(cache_path: Path) -> dict | None:
    """Load cache metadata (generated_at, embed_model). Return None if missing or invalid."""
    meta_path = _cache_meta_path(cache_path)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_cache_meta(cache_path: Path, embed_model: str) -> None:
    """Write cache metadata with current timestamp."""
    meta_path = _cache_meta_path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embed_model": embed_model,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=0)


def _format_generated_at(iso_str: str) -> str:
    """Format ISO timestamp for display (e.g. 2025-03-13 14:30 UTC)."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return iso_str


def load_cached_embeddings(cache_path: Path) -> list[dict] | None:
    """Load docs from cache file if it exists. Return None if missing or invalid."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (json.JSONDecodeError, OSError):
        return None


def save_cached_embeddings(docs: list[dict], cache_path: Path, embed_model: str) -> None:
    """Write docs (with embeddings) to cache JSONL and save metadata with timestamp."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    save_cache_meta(cache_path, embed_model)


def dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _query_terms(query: str) -> set[str]:
    """Normalize query into set of lowercased words (min length 2) for keyword matching."""
    words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", query.lower())
    return set(words)


def _chunk_text(doc: dict) -> str:
    """Concatenate title, summary, and original_question so we can match query terms."""
    parts = [doc.get("title", ""), doc.get("metadata", {}).get("solution_summary", "")]
    oq = (doc.get("metadata") or {}).get("original_question", "")
    if oq:
        parts.append(oq)
    return " ".join(parts).lower()


def retrieve(docs: list[dict], query: str, embed_model: str, top_k: int) -> list[tuple[dict, float]]:
    """Embed query, score by dot product, then re-rank by query-term overlap so specific fields (e.g. email vs address) rank higher."""
    out = ollama.embed(model=embed_model, input=query)
    query_vec = out["embeddings"][0]
    scored = []
    for d in docs:
        emb = d.get("embedding")
        if not emb:
            continue
        scored.append((d, dot(query_vec, emb)))
    scored.sort(key=lambda x: -x[1])
    # Re-rank top candidates by semantic score + keyword boost (reduces confusion between similar chunks like developer email vs address)
    pool_size = min(len(scored), max(top_k * 2, 15))
    pool = scored[:pool_size]
    terms = _query_terms(query)
    if terms:
        def reorder_key(item: tuple[dict, float]) -> tuple[float, float]:
            doc, sem_score = item
            text = _chunk_text(doc)
            matches = sum(1 for t in terms if t in text)
            keyword_boost = 0.05 * matches  # small boost so semantic still dominates
            return (-(sem_score + keyword_boost), -matches)
        pool.sort(key=reorder_key)
    return pool[:top_k]


def build_context(chunks: list[tuple[dict, float]]) -> str:
    """Format retrieved chunks as numbered context for the prompt. Include original Q&A so the model can tell apart similar chunks (e.g. developer email vs address)."""
    parts = []
    for i, (doc, _) in enumerate(chunks, 1):
        title = doc.get("title", "Untitled")
        meta = doc.get("metadata") or {}
        summary = meta.get("solution_summary", "")
        url = meta.get("source_url", "N/A")
        oq = meta.get("original_question", "")
        oa = meta.get("original_answer", "")
        block = f"[{i}] Title: {title}\nSummary: {summary}\nSource: {url}"
        if oq:
            block += f"\nOriginal question: {oq}"
        if oa:
            block += f"\nAnswer: {oa}"
        parts.append(block)
    return "\n\n".join(parts)


def print_sources(chunks: list[tuple[dict, float]]) -> None:
    """Print Sources section with [i] title (url)."""
    print("\nSources:")
    for i, (doc, _) in enumerate(chunks, 1):
        title = doc.get("title", "Untitled")
        meta = doc.get("metadata") or {}
        url = meta.get("source_url", "")
        if url:
            print(f"  [{i}] {title} ({url})")
        else:
            print(f"  [{i}] {title}")


def chat_stream(model: str, system_content: str, user_content: str) -> None:
    """Stream LLM response to stdout."""
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        stream=True,
    )
    for chunk in stream:
        msg = chunk.get("message", {})
        content = msg.get("content", "")
        if content:
            print(content, end="", flush=True)
    print(flush=True)


def chat_no_stream(model: str, system_content: str, user_content: str) -> None:
    """Single LLM call, then print full response."""
    r = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )
    content = (r.get("message") or {}).get("content", "") or ""
    print(content, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG CLI chatbot with citations (Ollama).")
    parser.add_argument(
        "--kb",
        type=str,
        default="devtac_rag_pretty.json",
        help="Path to knowledge base (JSON array or JSONL)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="embeddinggemma:latest",
        help="Ollama embedding model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:1b",
        help="Ollama chat model",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to pass to the LLM",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print full response at once instead of streaming",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Ignore cache and regenerate embeddings",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not load or save embeddings cache (always build from KB)",
    )
    args = parser.parse_args()

    kb_path = Path(args.kb)
    if not kb_path.exists():
        print(f"Error: Knowledge base not found: {kb_path}", file=sys.stderr)
        sys.exit(1)

    # Cache is used by default; store next to KB as {stem}_embeddings.jsonl
    cache_path = None if args.no_cache else kb_path.with_name(kb_path.stem + "_embeddings.jsonl")

    print("Loading knowledge base...", end=" ", flush=True)
    try:
        docs = None
        if cache_path and not args.regenerate:
            docs = load_cached_embeddings(cache_path)
        if docs is not None and len(docs) > 0:
            meta = load_cache_meta(cache_path) if cache_path else None
            if meta and meta.get("generated_at"):
                when = _format_generated_at(meta["generated_at"])
            elif cache_path and cache_path.exists():
                when = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            else:
                when = "unknown"
            if sys.stdin.isatty() and cache_path:
                print(f"done ({len(docs)} docs).")
                print(f"Found embeddings (generated at {when}).")
                while True:
                    choice = input("  [L]oad existing or [G]enerate again? [L]: ").strip().upper() or "L"
                    if choice in ("L", "LOAD"):
                        break
                    if choice in ("G", "GENERATE"):
                        docs = load_kb(kb_path)
                        print(f"Regenerating... loaded {len(docs)} docs from KB.")
                        break
                    print("  Enter L or G.")
            else:
                print(f"done (from cache, {len(docs)} docs, generated at {when}).")
        else:
            docs = load_kb(kb_path)
            print(f"done ({len(docs)} docs).")
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error loading KB: {e}", file=sys.stderr)
        sys.exit(1)

    if not docs:
        print("Error: No documents in knowledge base.", file=sys.stderr)
        sys.exit(1)

    need_embed = not all(d.get("embedding") for d in docs)
    if need_embed:
        print("Building embeddings...", end=" ", flush=True)
        try:
            ensure_embeddings(docs, args.embed_model)
            if cache_path:
                save_cached_embeddings(docs, cache_path, args.embed_model)
            print("done.")
        except Exception as e:
            print(f"Error: {e}. Is Ollama running? ollama pull {args.embed_model}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Embeddings present, skipping embed step.")

    # Quick Ollama chat check
    try:
        ollama.chat(model=args.model, messages=[{"role": "user", "content": "Hi"}])
    except Exception as e:
        print(f"Error: Could not reach Ollama (chat). Is it running? ollama pull {args.model}", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDevtac RAG Chat. Ask a question or 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        print("Searching...", end=" ", flush=True)
        try:
            chunks = retrieve(docs, query, args.embed_model, args.top_k)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
        print("done.")

        context = build_context(chunks) if chunks else "(No relevant context found.)"
        system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        print("Assistant: ", end="", flush=True)
        try:
            if args.no_stream:
                chat_no_stream(args.model, system_content, query)
            else:
                chat_stream(args.model, system_content, query)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            continue

        if chunks:
            print_sources(chunks)
        print()


if __name__ == "__main__":
    main()
