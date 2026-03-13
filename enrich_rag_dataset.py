#!/usr/bin/env python3
"""
Enrich a Q&A CSV into a RAG-ready JSONL dataset using Ollama (Gemma 3:1b + embeddinggemma).

With enrichment (default), each document is crm.json-style:
  - id, title, category_path, text_for_embedding
  - metadata: intent_class, product_scope, support_level, answer_type, solution_summary,
    key_entities, question_variants, problem_signals, confidence_keywords, source_url, etc.

Use --no-enrich for flat output (text + id, question, answer, ...) without LLM metadata.

Usage:
  pip install -r requirements.txt
  ollama pull gemma3:1b
  ollama pull embeddinggemma:latest

  python enrich_rag_dataset.py --input devtac_info_full_v2.csv --output devtac_rag_enriched.jsonl
  python enrich_rag_dataset.py --input devtac_info_full_v2.csv --output devtac_rag_enriched.jsonl --embed
  python enrich_rag_dataset.py --no-enrich --output devtac_rag_fast.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import ollama
except ImportError:
    print("Error: Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


ENRICHMENT_PROMPT = """Turn this Q&A into a single short passage (2-4 sentences) that a retrieval system can use to match user questions. Write only the passage: no bullet points, no "Question:" or "Answer:" labels.

Question: {question}
Answer: {answer}
Category: {category}
Notes: {notes}

Passage:"""

METADATA_ENRICHMENT_PROMPT = """You are enriching a knowledge-base chunk for RAG retrieval. Given the Q&A below, respond with ONLY a valid JSON object (no markdown, no extra text) with these exact keys:
- "title": short 2-6 word title for this chunk
- "solution_summary": 2-4 sentence summary suitable for semantic search
- "intent_class": one of: factual, identity, contact, location, history, services, products, partnerships, configuration, feature_explanation, integration, other
- "answer_type": one of: explanation, step_by_step, factual, list
- "key_entities": array of 3-6 important names/terms (companies, products, concepts)
- "question_variants": array of 4-5 alternative phrasings of the question users might ask
- "problem_signals": array of 4-5 short search-like phrases (lowercase, no punctuation) that signal someone needs this answer
- "confidence_keywords": array of 4-5 terms that boost match confidence when present

Question: {question}
Answer: {answer}
Category: {category}
Notes: {notes}

JSON:"""


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with robust encoding (replace bad bytes if not UTF-8)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return pd.read_csv(f, on_bad_lines="skip")


def fallback_passage(row: pd.Series) -> str:
    """Build a deterministic passage when LLM is skipped or fails."""
    parts = [
        f"Question: {row.get('question', '')}.",
        f"Answer: {row.get('answer', '')}.",
    ]
    notes = row.get("notes", "")
    if pd.notna(notes) and str(notes).strip():
        parts.append(str(notes).strip())
    return " ".join(parts)


def check_ollama_connection(model: str) -> None:
    """Verify Ollama is reachable and model is available. Exit with clear message on failure."""
    try:
        ollama.chat(model=model, messages=[{"role": "user", "content": "Hi"}])
    except Exception as e:
        print(
            f"Error: Could not reach Ollama (is it running?). Pull the model with: ollama pull {model}",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)


def _default_metadata(row: pd.Series) -> dict:
    """Fallback metadata when LLM is skipped or fails."""
    q = str(row.get("question", "") or "")
    return {
        "title": (q[:50] + "..." if len(q) > 50 else q) or "Untitled",
        "solution_summary": fallback_passage(row),
        "intent_class": str(row.get("category", "other") or "other"),
        "answer_type": "factual",
        "key_entities": [],
        "question_variants": [q] if q else [],
        "problem_signals": [],
        "confidence_keywords": [],
    }


def _parse_metadata_json(raw: str, row: pd.Series) -> dict:
    """Parse LLM JSON response into metadata dict; merge with defaults on failure."""
    default = _default_metadata(row)
    raw = (raw or "").strip()
    # Strip markdown code fence if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    if not raw:
        return default
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return default
    if not isinstance(data, dict):
        return default
    return {
        "title": data.get("title") or default["title"],
        "solution_summary": data.get("solution_summary") or default["solution_summary"],
        "intent_class": data.get("intent_class") or default["intent_class"],
        "answer_type": data.get("answer_type") or default["answer_type"],
        "key_entities": data.get("key_entities") if isinstance(data.get("key_entities"), list) else default["key_entities"],
        "question_variants": data.get("question_variants") if isinstance(data.get("question_variants"), list) else default["question_variants"],
        "problem_signals": data.get("problem_signals") if isinstance(data.get("problem_signals"), list) else default["problem_signals"],
        "confidence_keywords": data.get("confidence_keywords") if isinstance(data.get("confidence_keywords"), list) else default["confidence_keywords"],
    }


def enrich_metadata(row: pd.Series, model: str) -> dict:
    """Use Ollama to produce full crm-style metadata (title, summary, question_variants, keywords, etc.)."""
    prompt = METADATA_ENRICHMENT_PROMPT.format(
        question=row.get("question", ""),
        answer=row.get("answer", ""),
        category=row.get("category", ""),
        notes=row.get("notes", ""),
    )
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (response.message.content or "").strip()
            if content:
                return _parse_metadata_json(content, row)
        except Exception:
            if attempt == 1:
                break
            continue
    return _default_metadata(row)


def enrich_passage(row: pd.Series, model: str) -> str:
    """Use Ollama chat to produce a retrieval-oriented passage. Fallback on error or empty."""
    prompt = ENRICHMENT_PROMPT.format(
        question=row.get("question", ""),
        answer=row.get("answer", ""),
        category=row.get("category", ""),
        notes=row.get("notes", ""),
    )
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (response.message.content or "").strip()
            if content:
                return content
        except Exception:
            if attempt == 1:
                pass
            continue
    return fallback_passage(row)


def build_text_for_embedding(meta: dict, category: str) -> str:
    """Build retrieval-rich text for embedding (crm.json-style: title + summary + question variants)."""
    parts = [
        meta.get("title", ""),
        category,
        meta.get("solution_summary", ""),
    ]
    variants = meta.get("question_variants") or []
    if variants:
        parts.append(" ".join(variants))
    return " ".join(p for p in parts if p).strip()


def row_to_doc(row: pd.Series, text: str, rich_metadata: dict | None = None) -> dict:
    """Build one JSONL document: crm.json-style with text_for_embedding and metadata, or flat text + fields."""
    category = str(row["category"]) if pd.notna(row.get("category")) else ""
    source_url = str(row["source_url(s)"]) if pd.notna(row.get("source_url(s)")) else ""
    confidence = str(row["confidence"]) if pd.notna(row.get("confidence")) else ""
    notes = str(row["notes"]) if pd.notna(row.get("notes")) else ""
    row_id = int(row["id"]) if pd.notna(row.get("id")) else None
    question = str(row["question"]) if pd.notna(row.get("question")) else ""
    answer = str(row["answer"]) if pd.notna(row.get("answer")) else ""

    if rich_metadata:
        text_for_embedding = build_text_for_embedding(rich_metadata, category)
        doc = {
            "id": f"devtac_{row_id}" if row_id is not None else None,
            "title": rich_metadata.get("title", ""),
            "category_path": category,
            "text_for_embedding": text_for_embedding,
            "metadata": {
                "intent_class": rich_metadata.get("intent_class", ""),
                "product_scope": "Devtac",
                "support_level": "L1",
                "answer_type": rich_metadata.get("answer_type", ""),
                "solution_summary": rich_metadata.get("solution_summary", ""),
                "key_entities": rich_metadata.get("key_entities", []),
                "question_variants": rich_metadata.get("question_variants", []),
                "problem_signals": rich_metadata.get("problem_signals", []),
                "confidence_keywords": rich_metadata.get("confidence_keywords", []),
                "source_url": source_url,
                "confidence": confidence,
                "notes": notes,
                "original_question": question,
                "original_answer": answer,
            },
        }
        return doc
    doc = {
        "text": text,
        "id": row_id,
        "question": question,
        "answer": answer,
        "source_url(s)": source_url,
        "confidence": confidence,
        "category": category,
        "notes": notes,
    }
    return doc


def check_ollama_embed(model: str) -> None:
    """Verify Ollama embed endpoint works. Exit with clear message on failure."""
    try:
        ollama.embed(model=model, input="test")
    except Exception as e:
        print(
            f"Error: Could not run embeddings. Ensure Ollama is running and run: ollama pull {model}",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """Return a simple text progress bar like [=========>     ] 45/90 50%"""
    if total <= 0:
        return "[]" + f" {current}/{total}"
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + ">" * (1 if filled < width else 0) + " " * (width - filled - 1)
    return f"[{bar}] {current}/{total} {int(pct * 100)}%"


def run_embeddings(docs: list[dict], model: str, batch_size: int) -> None:
    """Fill doc['embedding'] for each doc using Ollama embed (in-place)."""
    texts = [d.get("text_for_embedding") or d.get("text", "") for d in docs]
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            out = ollama.embed(model=model, input=batch)
            embs = out.get("embeddings", [])
            for j, doc in enumerate(docs[i : i + batch_size]):
                if j < len(embs):
                    doc["embedding"] = embs[j]
        except Exception as e:
            print(f"Embedding batch error at index {i}: {e}", file=sys.stderr)
            for doc in docs[i : i + batch_size]:
                doc["embedding"] = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich Q&A CSV into RAG JSONL using Gemma 3:1b; optional embeddings with embeddinggemma."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="devtac_info_full_v2.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="devtac_rag_enriched.jsonl",
        help="Path to output JSONL",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Run embedding step and add 'embedding' to each document",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:1b",
        help="Ollama chat model for enrichment",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="embeddinggemma:latest",
        help="Ollama embedding model",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip LLM enrichment; use question+answer+notes as passage",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for embedding API calls",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N rows (0 = all). Useful for testing enrichment.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not args.no_enrich:
        print("Checking Ollama connection (enrichment model)...", end=" ", flush=True)
        check_ollama_connection(args.model)
        print("done.")
    if args.embed:
        print("Checking Ollama embedding model...", end=" ", flush=True)
        check_ollama_embed(args.embed_model)
        print("done.")

    print("Loading CSV...", end=" ", flush=True)
    df = load_csv(str(input_path))
    if args.limit > 0:
        df = df.head(args.limit)
    n = len(df)
    print(f"done. ({n} rows)")

    docs: list[dict] = []
    step_label = "Building passages" if args.no_enrich else "Enriching"
    for i, (_, row) in enumerate(df.iterrows()):
        if args.no_enrich:
            text = fallback_passage(row)
            doc = row_to_doc(row, text, rich_metadata=None)
        else:
            rich_metadata = enrich_metadata(row, args.model)
            text = rich_metadata.get("solution_summary") or fallback_passage(row)
            doc = row_to_doc(row, text, rich_metadata=rich_metadata)
        docs.append(doc)
        bar = _progress_bar(i + 1, n)
        print(f"\r  {step_label}: {bar}", end="", flush=True)

    print(f"\r  {step_label}: {_progress_bar(n, n)} done.")

    if args.embed:
        num_batches = (len(docs) + args.batch_size - 1) // args.batch_size
        for i in range(0, len(docs), args.batch_size):
            batch_num = i // args.batch_size + 1
            print(f"\r  Embedding: batch {batch_num}/{num_batches}...", end="", flush=True)
            run_embeddings(docs[i : i + args.batch_size], args.embed_model, args.batch_size)
        print(f"\r  Embedding: {num_batches}/{num_batches} batches done.")

    print("Writing output...", end=" ", flush=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"done. -> {output_path} ({n} lines)")


if __name__ == "__main__":
    main()
