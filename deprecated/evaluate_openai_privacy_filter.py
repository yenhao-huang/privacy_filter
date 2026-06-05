"""Deprecated exact span-match evaluator for local OpenAI Privacy Filter predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

from core.service.openai_privacy_filter_model import DEFAULT_MODEL_PATH
from core.service.privacy_eval import (
    exact_match_counts,
    format_metric,
    labels_to_spans,
    load_gold_spans,
    summarize_counts,
)


DEFAULT_INPUT = Path("results/tw-pii-openai-in-shcema_predictions.csv")
DEFAULT_PER_LABEL = Path("results/tw-pii-openai-in-shcema_eval_by_label.csv")
DEFAULT_OVERALL = Path("results/tw-pii-openai-in-shcema_eval_overall.csv")
DEFAULT_DETAILS = Path("results/tw-pii-openai-in-shcema_eval_details.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate span-level exact-match privacy predictions.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local tokenizer/model directory.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--per-label-output", type=Path, default=DEFAULT_PER_LABEL)
    parser.add_argument("--overall-output", type=Path, default=DEFAULT_OVERALL)
    parser.add_argument("--details-output", type=Path, default=DEFAULT_DETAILS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    detail_rows: list[dict[str, object]] = []
    with args.input.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            text = row["text"]
            gold_spans = load_gold_spans(row["spans"])
            predicted_labels = json.loads(row["predicted_token_classes_json"])
            offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
            predicted_spans = labels_to_spans(text, predicted_labels, offsets)

            true_positives, false_positives, false_negatives = exact_match_counts(
                gold_spans, predicted_spans
            )
            for outcome, spans in (
                ("tp", true_positives),
                ("fp", false_positives),
                ("fn", false_negatives),
            ):
                for span in spans:
                    detail_rows.append(
                        {
                            "id": row["id"],
                            "schema_name": row["schema_name"],
                            "outcome": outcome,
                            "label": span.label,
                            "start": span.start,
                            "end": span.end,
                            "text": span.text,
                        }
                    )

    per_label_rows = summarize_counts(detail_rows)
    total_tp = sum(int(row["tp"]) for row in per_label_rows)
    total_fp = sum(int(row["fp"]) for row in per_label_rows)
    total_fn = sum(int(row["fn"]) for row in per_label_rows)
    accuracy = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn else 0.0
    macro_f1 = (
        sum(float(row["f1"]) for row in per_label_rows) / len(per_label_rows)
        if per_label_rows
        else 0.0
    )
    overall_row = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "accuracy": format_metric(accuracy),
        "macro_f1": format_metric(macro_f1),
    }

    args.per_label_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.per_label_output, per_label_rows)
    _write_csv(args.overall_output, [overall_row])
    _write_csv(args.details_output, detail_rows)

    print(f"wrote {args.overall_output}")
    print(f"wrote {args.per_label_output}")
    print(f"wrote {args.details_output}")
    print(overall_row)
    return 0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
