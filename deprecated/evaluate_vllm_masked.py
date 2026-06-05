"""Deprecated placeholder-count evaluator for vLLM-compatible masked-text predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from core.service.privacy_eval import OPENAI_PRIVACY_LABELS


DEFAULT_INPUT = Path("results/tw-pii-vllm-compatible_masked.csv")
DEFAULT_OVERALL = Path("results/tw-pii-vllm-compatible_eval_overall.csv")
DEFAULT_BY_LABEL = Path("results/tw-pii-vllm-compatible_eval_by_label.csv")
DEFAULT_DETAILS = Path("results/tw-pii-vllm-compatible_eval_details.csv")

PLACEHOLDER_RE = re.compile(
    r"\[(account_number|private_address|private_date|private_email|private_person|private_phone|private_url|secret)\]"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate vLLM masked_text outputs by per-label placeholder counts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--overall-output", type=Path, default=DEFAULT_OVERALL)
    parser.add_argument("--by-label-output", type=Path, default=DEFAULT_BY_LABEL)
    parser.add_argument("--details-output", type=Path, default=DEFAULT_DETAILS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detail_rows: list[dict[str, object]] = []

    with args.input.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            gold = gold_label_counts(row["spans"])
            predicted = Counter(PLACEHOLDER_RE.findall(row.get("masked_text", "")))
            for label in sorted(OPENAI_PRIVACY_LABELS):
                tp = min(gold[label], predicted[label])
                fp = max(predicted[label] - gold[label], 0)
                fn = max(gold[label] - predicted[label], 0)
                if tp or fp or fn:
                    detail_rows.append(
                        {
                            "id": row["id"],
                            "schema_name": row["schema_name"],
                            "label": label,
                            "gold_count": gold[label],
                            "predicted_count": predicted[label],
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "masked_text": row.get("masked_text", ""),
                        }
                    )

    by_label_rows = summarize(detail_rows)
    total_tp = sum(int(row["tp"]) for row in by_label_rows)
    total_fp = sum(int(row["fp"]) for row in by_label_rows)
    total_fn = sum(int(row["fn"]) for row in by_label_rows)
    accuracy = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn else 0.0
    macro_f1 = (
        sum(float(row["f1"]) for row in by_label_rows) / len(by_label_rows)
        if by_label_rows
        else 0.0
    )
    overall_row = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
    }

    args.overall_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.overall_output, [overall_row])
    write_csv(args.by_label_output, by_label_rows)
    write_csv(args.details_output, detail_rows)
    print(f"wrote {args.overall_output}")
    print(f"wrote {args.by_label_output}")
    print(f"wrote {args.details_output}")
    print(overall_row)
    return 0


def gold_label_counts(raw_spans: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for span in json.loads(raw_spans):
        label = span.get("expected_model_label") or span.get("label")
        if label in OPENAI_PRIVACY_LABELS:
            counter[str(label)] += 1
    return counter


def summarize(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in sorted(OPENAI_PRIVACY_LABELS):
        tp = sum(int(row["tp"]) for row in detail_rows if row["label"] == label)
        fp = sum(int(row["fp"]) for row in detail_rows if row["label"] == label)
        fn = sum(int(row["fn"]) for row in detail_rows if row["label"] == label)
        accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "accuracy": round(accuracy, 6),
                "f1": round(f1, 6),
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
