#!/usr/bin/env python3
"""
Read a JSONL file and write it as pretty-printed JSON (array of objects).

Usage:
  python prettify_jsonl.py
  python prettify_jsonl.py --input devtac_rag.jsonl --output devtac_rag_pretty.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skip line {i}: {e}", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prettify JSONL to indented JSON.")
    parser.add_argument(
        "--input",
        type=str,
        default="devtac_rag.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="devtac_rag_pretty.json",
        help="Output JSON file (use - for stdout)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent spaces (default 2)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    docs = load_jsonl(input_path)
    print(f"Loaded {len(docs)} records from {input_path}", file=sys.stderr)

    text = json.dumps(docs, indent=args.indent, ensure_ascii=False)

    if args.output == "-":
        print(text)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
