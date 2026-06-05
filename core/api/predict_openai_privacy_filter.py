"""Run local OpenAI Privacy Filter predictions on benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from core.service.openai_privacy_filter_model import (
    DEFAULT_MODEL_PATH,
    LocalOpenAIPrivacyFilter,
)


DEFAULT_INPUT = Path("data/benchmark/tw-pii-openai-in-shcema.csv")
DEFAULT_OUTPUT = Path("results/tw-pii-openai-in-shcema_predictions.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict token classes for the benchmark with local openai/privacy-filter."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictor = LocalOpenAIPrivacyFilter(
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or []) + [
            "predicted_token_classes",
            "predicted_token_classes_json",
        ]
        rows = list(reader)

    if args.limit is not None:
        rows = rows[: args.limit]

    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            labels = predictor.predict_label_sequence(row["text"])
            row["predicted_token_classes"] = " ".join(labels)
            row["predicted_token_classes_json"] = json.dumps(labels, ensure_ascii=False)
            writer.writerow(row)

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

