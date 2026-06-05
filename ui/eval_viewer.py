"""Gradio UI for inspecting benchmark evaluation results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path

import gradio as gr
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.service.openai_privacy_filter_model import DEFAULT_MODEL_PATH  # noqa: E402
from core.service.privacy_eval import labels_to_spans, load_gold_spans  # noqa: E402


DEFAULT_PREDICTIONS = Path("results/tw-pii-openai-in-shcema_predictions.csv")
DEFAULT_QWEN_PREDICTIONS = Path("results/tw-pii-vllm-compatible_masked.csv")
PLACEHOLDER_RE = re.compile(
    r"\[(account_number|private_address|private_date|private_email|private_person|private_phone|private_url|secret)\]"
)


@dataclass(frozen=True)
class UiSpan:
    label: str
    start: int
    end: int
    text: str
    expected_model_label: str


def highlighted_segments(text: str, spans) -> list[tuple[str, str | None]]:
    if not spans:
        return [(text, None)]

    chunks: list[tuple[str, str | None]] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        if cursor < span.start:
            chunks.append((text[cursor : span.start], None))
        chunks.append((text[span.start : span.end], span.label))
        cursor = max(cursor, span.end)
    if cursor < len(text):
        chunks.append((text[cursor:], None))
    return chunks


def masked_text_segments(masked_text: str) -> list[tuple[str, str | None]]:
    chunks: list[tuple[str, str | None]] = []
    cursor = 0
    for match in PLACEHOLDER_RE.finditer(masked_text):
        if cursor < match.start():
            chunks.append((masked_text[cursor : match.start()], None))
        chunks.append((match.group(0), match.group(1)))
        cursor = match.end()
    if cursor < len(masked_text):
        chunks.append((masked_text[cursor:], None))
    return chunks or [(masked_text, None)]


def masked_segments_from_spans(text: str, spans) -> list[tuple[str, str | None]]:
    if not spans:
        return [(text, None)]

    chunks: list[tuple[str, str | None]] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        if cursor < span.start:
            chunks.append((text[cursor : span.start], None))
        chunks.append((f"[{span.label}]", span.label))
        cursor = max(cursor, span.end)
    if cursor < len(text):
        chunks.append((text[cursor:], None))
    return chunks


def load_records(predictions_path: Path, model_path: str) -> list[dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    records: list[dict[str, object]] = []
    with predictions_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            text = row["text"]
            gold_spans = load_raw_spans(row["spans"])
            predicted_labels = json.loads(row["predicted_token_classes_json"])
            offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
            predicted_spans = labels_to_spans(text, predicted_labels, offsets)
            records.append(
                {
                    "id": row["id"],
                    "schema_name": row["schema_name"],
                    "text": text,
                    "gold": masked_segments_from_spans(text, gold_spans),
                    "predicted": masked_segments_from_spans(text, predicted_spans),
                    "gold_rows": ideal_span_rows(gold_spans),
                    "predicted_rows": span_rows(predicted_spans),
                }
            )
    return records


def load_qwen_records(predictions_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with predictions_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            text = row["text"]
            gold_spans = load_raw_spans(row["spans"])
            masked_text = row.get("masked_text", "")
            records.append(
                {
                    "id": row["id"],
                    "schema_name": row["schema_name"],
                    "text": text,
                    "gold": masked_segments_from_spans(text, gold_spans),
                    "predicted": masked_text_segments(masked_text),
                    "gold_rows": ideal_span_rows(gold_spans),
                    "predicted_rows": placeholder_rows(masked_text),
                }
            )
    return records


def load_raw_spans(raw_spans: str) -> list[UiSpan]:
    spans = []
    for span in json.loads(raw_spans):
        label = str(span.get("label", ""))
        expected_model_label = str(span.get("expected_model_label", ""))
        spans.append(
            UiSpan(
                label=expected_model_label or label,
                start=int(span["start"]),
                end=int(span["end"]),
                text=str(span.get("text", "")),
                expected_model_label=expected_model_label,
            )
        )
    return sorted(spans, key=lambda span: (span.start, span.end, span.label))


def ideal_span_rows(spans: list[UiSpan]) -> list[list[object]]:
    return [
        [span.label, span.expected_model_label, span.start, span.end, span.text]
        for span in spans
    ]


def span_rows(spans) -> list[list[object]]:
    return [[span.label, span.start, span.end, span.text] for span in spans]


def placeholder_rows(masked_text: str) -> list[list[object]]:
    return [
        [match.group(1), match.start(), match.end(), match.group(0)]
        for match in PLACEHOLDER_RE.finditer(masked_text)
    ]


def build_app(
    predictions_path: Path,
    qwen_predictions_path: Path,
    model_path: str,
) -> gr.Blocks:
    openai_records = load_records(predictions_path, model_path)
    qwen_records = load_qwen_records(qwen_predictions_path)
    qwen_by_id = {str(record["id"]): record for record in qwen_records}
    records = []
    for openai_record in openai_records:
        qwen_record = qwen_by_id[str(openai_record["id"])]
        records.append(
            {
                "id": openai_record["id"],
                "schema_name": openai_record["schema_name"],
                "text": openai_record["text"],
                "gold": openai_record["gold"],
                "gold_rows": openai_record["gold_rows"],
                "openai_predicted": openai_record["predicted"],
                "openai_rows": openai_record["predicted_rows"],
                "qwen_predicted": qwen_record["predicted"],
                "qwen_rows": qwen_record["predicted_rows"],
            }
        )
    record_index = build_index(records)
    label_types = sorted(
        {"all"}
        | {str(record["schema_name"]) for record in records}
    )

    def select_record(label_type: str, sample_no: float):
        choice = choice_by_number(label_type, sample_no)
        record = record_index["by_id"][choice]
        return (
            choice,
            record["text"],
            record["gold"],
            record["gold_rows"],
            record["openai_predicted"],
            record["openai_rows"],
            record["qwen_predicted"],
            record["qwen_rows"],
        )

    def filter_samples(label_type: str):
        choices_by_label = record_index["choices_by_label"]
        filtered_choices = choices_by_label.get(label_type) or choices_by_label["all"]
        maximum = len(filtered_choices)
        return (gr.update(maximum=maximum, value=1), *select_record(label_type, 1))

    def move_sample(label_type: str, sample_no: float, delta: int):
        choices_by_label = record_index["choices_by_label"]
        filtered_choices = choices_by_label.get(label_type) or choices_by_label["all"]
        current_index = max(min(int(sample_no), len(filtered_choices)), 1) - 1
        next_number = (current_index + delta) % len(filtered_choices) + 1
        return (gr.update(value=next_number), *select_record(label_type, next_number))

    def previous_sample(label_type: str, sample_no: float):
        return move_sample(label_type, sample_no, -1)

    def next_sample(label_type: str, sample_no: float):
        return move_sample(label_type, sample_no, 1)

    def choice_by_number(label_type: str, sample_no: float) -> str:
        choices_by_label = record_index["choices_by_label"]
        filtered_choices = choices_by_label.get(label_type) or choices_by_label["all"]
        selected_index = max(min(int(sample_no), len(filtered_choices)), 1) - 1
        return filtered_choices[selected_index]

    with gr.Blocks(title="eval_viewer") as app:
        gr.Markdown("# eval_viewer")
        with gr.Row():
            label_type = gr.Dropdown(choices=label_types, value="all", label="label_type")
            sample_no = gr.Slider(
                minimum=1,
                maximum=len(record_index["choices"]),
                value=1,
                step=1,
                label="sample_no",
            )
            sample_id = gr.Textbox(label="sample_id", interactive=False)
            previous_button = gr.Button("Previous")
            next_button = gr.Button("Next", variant="primary")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_text = gr.Textbox(label="input text", lines=8, interactive=False)
                ideal_text = gr.HighlightedText(
                    label="ideal text",
                    combine_adjacent=True,
                    show_legend=True,
                )
                gold_table = gr.Dataframe(
                    label="ideal spans",
                    headers=["label", "expected_model_label", "start", "end", "text"],
                    datatype=["str", "str", "number", "number", "str"],
                    interactive=False,
                )
            with gr.Column(scale=1):
                openai_output = gr.HighlightedText(
                    label="openai output",
                    combine_adjacent=True,
                    show_legend=True,
                )
                openai_table = gr.Dataframe(
                    label="openai spans",
                    headers=["label", "start", "end", "text"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                )
            with gr.Column(scale=1):
                qwen_output = gr.HighlightedText(
                    label="qwen output",
                    combine_adjacent=True,
                    show_legend=True,
                )
                qwen_table = gr.Dataframe(
                    label="qwen placeholders",
                    headers=["label", "start", "end", "text"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                )

        sample_no.change(
            select_record,
            inputs=[label_type, sample_no],
            outputs=[
                sample_id,
                input_text,
                ideal_text,
                gold_table,
                openai_output,
                openai_table,
                qwen_output,
                qwen_table,
            ],
        )
        label_type.change(
            filter_samples,
            inputs=[label_type],
            outputs=[
                sample_no,
                sample_id,
                input_text,
                ideal_text,
                gold_table,
                openai_output,
                openai_table,
                qwen_output,
                qwen_table,
            ],
        )
        previous_button.click(
            previous_sample,
            inputs=[label_type, sample_no],
            outputs=[
                sample_no,
                sample_id,
                input_text,
                ideal_text,
                gold_table,
                openai_output,
                openai_table,
                qwen_output,
                qwen_table,
            ],
        )
        next_button.click(
            next_sample,
            inputs=[label_type, sample_no],
            outputs=[
                sample_no,
                sample_id,
                input_text,
                ideal_text,
                gold_table,
                openai_output,
                openai_table,
                qwen_output,
                qwen_table,
            ],
        )
        app.load(
            select_record,
            inputs=[label_type, sample_no],
            outputs=[
                sample_id,
                input_text,
                ideal_text,
                gold_table,
                openai_output,
                openai_table,
                qwen_output,
                qwen_table,
            ],
        )
    return app


def build_index(records: list[dict[str, object]]) -> dict[str, object]:
    choices = [f"{record['id']} | {record['schema_name']}" for record in records]
    choices_by_label: dict[str, list[str]] = {"all": choices}
    for choice, record in zip(choices, records, strict=True):
        choices_by_label.setdefault(str(record["schema_name"]), []).append(choice)
    return {
        "choices": choices,
        "by_id": {choice: record for choice, record in zip(choices, records, strict=True)},
        "choices_by_label": choices_by_label,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--qwen-predictions", type=Path, default=DEFAULT_QWEN_PREDICTIONS)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args.predictions, args.qwen_predictions, args.model_path)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
