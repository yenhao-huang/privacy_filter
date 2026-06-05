"""Evaluate masked predictions by whitespace-insensitive exact string match."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from transformers import AutoTokenizer

from core.service.openai_privacy_filter_model import DEFAULT_MODEL_PATH
from core.service.privacy_eval import labels_to_spans, load_gold_spans


DEFAULT_INPUT = Path("results/tw-pii-vllm-compatible_masked.csv")
DEFAULT_OVERALL = Path("results/tw-pii-vllm-compatible_masked_str_match_overall.csv")
DEFAULT_BY_LABEL = Path("results/tw-pii-vllm-compatible_masked_str_match_by_label.csv")
DEFAULT_DETAILS = Path("results/tw-pii-vllm-compatible_masked_str_match_details.csv")
WHITESPACE_RE = re.compile(r"\s+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate masked strings by exact match after removing whitespace. "
            "Each matched row gets score 1; each mismatched row gets score 0."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--source",
        choices=["masked_text", "openai_token_labels"],
        default="masked_text",
        help="Prediction format to evaluate.",
    )
    parser.add_argument("--prediction-column", default="masked_text")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Tokenizer path for OpenAI token labels.")
    parser.add_argument("--overall-output", type=Path, default=DEFAULT_OVERALL)
    parser.add_argument("--by-label-output", type=Path, default=DEFAULT_BY_LABEL)
    parser.add_argument("--details-output", type=Path, default=DEFAULT_DETAILS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    details: list[dict[str, object]] = []
    tokenizer = (
        AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        if args.source == "openai_token_labels"
        else None
    )

    with args.input.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if args.source == "masked_text" and args.prediction_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing prediction column: {args.prediction_column}")
        if args.source == "openai_token_labels" and "predicted_token_classes_json" not in (
            reader.fieldnames or []
        ):
            raise ValueError("Missing prediction column: predicted_token_classes_json")

        for row in reader:
            gold_masked = mask_text_with_gold_spans(row["text"], row["spans"])
            if args.source == "openai_token_labels":
                predicted_masked = mask_text_with_openai_token_labels(row, tokenizer)
            else:
                predicted_masked = row.get(args.prediction_column, "")
            normalized_gold = remove_whitespace(gold_masked)
            normalized_predicted = remove_whitespace(predicted_masked)
            score = int(normalized_gold == normalized_predicted)
            details.append(
                {
                    "id": row["id"],
                    "schema_name": row["schema_name"],
                    "score": score,
                    "gold_masked_text": gold_masked,
                    "predicted_masked_text": predicted_masked,
                    "normalized_gold": normalized_gold,
                    "normalized_predicted": normalized_predicted,
                }
            )

    total = len(details)
    correct = sum(int(row["score"]) for row in details)
    overall = {
        "total": total,
        "correct": correct,
        "score": round(correct / total, 6) if total else 0.0,
    }
    by_label = summarize_by_label(details)

    args.overall_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.overall_output, [overall])
    write_csv(args.by_label_output, by_label)
    write_csv(args.details_output, details)

    print(f"wrote {args.overall_output}")
    print(f"wrote {args.by_label_output}")
    print(f"wrote {args.details_output}")
    print(overall)
    return 0


def mask_text_with_gold_spans(text: str, raw_spans: str) -> str:
    return mask_text_with_spans(text, load_gold_spans(raw_spans))


def mask_text_with_openai_token_labels(row: dict[str, str], tokenizer) -> str:
    text = row["text"]
    labels = json.loads(row["predicted_token_classes_json"])
    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    spans = labels_to_spans(text, labels, offsets)
    return mask_text_with_spans(text, spans)


def mask_text_with_spans(text: str, spans) -> str:
    chunks: list[str] = []
    cursor = 0
    spans = sorted(spans, key=lambda span: (span.start, span.end, span.label))
    for span in spans:
        if cursor < span.start:
            chunks.append(text[cursor : span.start])
        chunks.append(f"[{span.label}]")
        cursor = max(cursor, span.end)
    if cursor < len(text):
        chunks.append(text[cursor:])
    return "".join(chunks)


def remove_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub("", text)


def summarize_by_label(details: list[dict[str, object]]) -> list[dict[str, object]]:
    labels = sorted({str(row["schema_name"]) for row in details})
    rows: list[dict[str, object]] = []
    for label in labels:
        label_rows = [row for row in details if row["schema_name"] == label]
        total = len(label_rows)
        correct = sum(int(row["score"]) for row in label_rows)
        rows.append(
            {
                "schema_name": label,
                "total": total,
                "correct": correct,
                "score": round(correct / total, 6) if total else 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
