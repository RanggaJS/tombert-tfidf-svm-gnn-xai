# coding=utf-8
"""
Standalone GPT-based XAI generator.

Usage example:
  OPENAI_API_KEY=... python3 run_program/run_xai_gpt.py \
      --input ./output/tfidf_svm_ultra_optimized_YYYYMMDD_HHMMSS/predictions.jsonl \
      --output_dir ./output/xai_explanations

Expected input formats (detected by extension):
  - .jsonl: one JSON per line with keys: text, prediction, probabilities (list), true_label
  - .json: array of the same objects

Output:
  - explanations.jsonl in the chosen output_dir
"""
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from methods.xai.gpt_explainer import generate_gpt_explanations


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if isinstance(content, list):
        return content
    raise ValueError("JSON file must contain an array of prediction objects.")


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate GPT-based XAI explanations from prediction outputs.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to predictions file (.jsonl or .json) containing text, prediction, probabilities, true_label.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output/xai_explanations",
        help="Directory to store generated explanations (default: ./output/xai_explanations).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=320,
        help="Max tokens for each explanation (default: 320).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of samples to explain (default: all).",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".jsonl":
        samples = _read_jsonl(input_path)
    elif ext == ".json":
        samples = _read_json(input_path)
    else:
        raise ValueError("Unsupported input format. Use .jsonl or .json.")

    if args.limit is not None:
        samples = samples[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"explanations_{timestamp}.jsonl")

    explanations = generate_gpt_explanations(
        samples,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    _write_jsonl(output_file, explanations)

    print(f"✅ Generated {len(explanations)} explanations -> {output_file}")
    print("Pastikan OPENAI_API_KEY sudah diset di environment sebelum menjalankan.")


if __name__ == "__main__":
    main()

