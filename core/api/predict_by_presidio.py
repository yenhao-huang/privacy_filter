"""Predict benchmark masked text with Presidio recognizers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from core.service.presidio_privacy_predictor import build_presidio_analyzer, mask_text, predict_spans


DEFAULT_INPUT = Path("data/benchmark/tw-pii-openai-in-shcema.csv")
DEFAULT_OUTPUT = Path("results/tw-pii-presidio_masked.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict masked benchmark text with Presidio.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_rows(args.input, args.limit)
    analyzer = build_presidio_analyzer()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["masked_text", "presidio_spans_json"] if rows else []
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            spans = predict_spans(row["text"], analyzer)
            row["masked_text"] = mask_text(row["text"], spans)
            row["presidio_spans_json"] = json.dumps(spans, ensure_ascii=False)
            writer.writerow(row)
            print(f"{index}/{len(rows)} {row['id']}")

    print(f"wrote {args.output}")
    return 0


def read_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return rows[:limit] if limit is not None else rows


if __name__ == "__main__":
    raise SystemExit(main())
