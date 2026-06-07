"""Gradio UI for comparing all four privacy-masking methods side by side."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.service.openai_privacy_filter_model import DEFAULT_MODEL_PATH  # noqa: E402
from core.service.privacy_eval import labels_to_spans, load_gold_spans  # noqa: E402

DEFAULT_OPENAI_PREDICTIONS = Path("results/tw-pii-openai-in-shcema_predictions.csv")
DEFAULT_PRESIDIO_PREDICTIONS = Path("results/tw-pii-presidio_masked.csv")
DEFAULT_QWEN_PREDICTIONS = Path("results/tw-pii-vllm-compatible_masked.csv")
DEFAULT_GEMMA4_PREDICTIONS = Path("results/tw-pii-gemma4_masked.csv")

PLACEHOLDER_RE = re.compile(
    r"\[(account_number|private_address|private_date|private_email"
    r"|private_person|private_phone|private_url|secret)\]"
)


@dataclass(frozen=True)
class UiSpan:
    label: str
    start: int
    end: int


def masked_segments_from_spans(text: str, spans: list[UiSpan]) -> list[tuple[str, str | None]]:
    chunks: list[tuple[str, str | None]] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        if cursor < span.start:
            chunks.append((text[cursor:span.start], None))
        chunks.append((f"[{span.label}]", span.label))
        cursor = max(cursor, span.end)
    if cursor < len(text):
        chunks.append((text[cursor:], None))
    return chunks or [(text, None)]


def masked_text_segments(masked_text: str) -> list[tuple[str, str | None]]:
    chunks: list[tuple[str, str | None]] = []
    cursor = 0
    for m in PLACEHOLDER_RE.finditer(masked_text):
        if cursor < m.start():
            chunks.append((masked_text[cursor:m.start()], None))
        chunks.append((m.group(0), m.group(1)))
        cursor = m.end()
    if cursor < len(masked_text):
        chunks.append((masked_text[cursor:], None))
    return chunks or [(masked_text, None)]


def load_gold_spans_from_row(row: dict[str, str]) -> list[UiSpan]:
    spans = []
    for span in json.loads(row["spans"]):
        label = str(span.get("expected_model_label") or span.get("label", ""))
        spans.append(UiSpan(label=label, start=int(span["start"]), end=int(span["end"])))
    return sorted(spans, key=lambda s: (s.start, s.end))


def load_openai_records(path: Path, model_path: str) -> dict[str, dict]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    records = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row["text"]
            labels = json.loads(row["predicted_token_classes_json"])
            offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
            predicted_spans = labels_to_spans(text, labels, offsets)
            predicted = masked_segments_from_spans(
                text, [UiSpan(s.label, s.start, s.end) for s in predicted_spans]
            )
            records[row["id"]] = {"predicted": predicted}
    return records


def load_masked_text_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            records[row["id"]] = {"predicted": masked_text_segments(row.get("masked_text", ""))}
    return records


def load_all_records(
    openai_path: Path,
    presidio_path: Path,
    qwen_path: Path,
    gemma4_path: Path,
    model_path: str,
) -> list[dict]:
    openai_by_id = load_openai_records(openai_path, model_path)
    presidio_by_id = load_masked_text_records(presidio_path)
    qwen_by_id = load_masked_text_records(qwen_path)
    gemma4_by_id = load_masked_text_records(gemma4_path)

    records = []
    with openai_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rid = row["id"]
            gold_spans = load_gold_spans_from_row(row)
            records.append(
                {
                    "id": rid,
                    "schema_name": row["schema_name"],
                    "text": row["text"],
                    "gold": masked_segments_from_spans(row["text"], gold_spans),
                    "openai": openai_by_id[rid]["predicted"],
                    "presidio": presidio_by_id[rid]["predicted"],
                    "qwen": qwen_by_id[rid]["predicted"],
                    "gemma4": gemma4_by_id[rid]["predicted"],
                }
            )
    return records


def build_index(records: list[dict]) -> dict:
    choices = [f"{r['id']} | {r['schema_name']}" for r in records]
    choices_by_label: dict[str, list[str]] = {"all": list(choices)}
    for choice, record in zip(choices, records):
        choices_by_label.setdefault(str(record["schema_name"]), []).append(choice)
    return {
        "choices": choices,
        "by_id": {c: r for c, r in zip(choices, records)},
        "choices_by_label": choices_by_label,
    }


def build_app(
    openai_path: Path,
    presidio_path: Path,
    qwen_path: Path,
    gemma4_path: Path,
    model_path: str,
) -> gr.Blocks:
    records = load_all_records(openai_path, presidio_path, qwen_path, gemma4_path, model_path)
    index = build_index(records)
    label_types = sorted({"all"} | {str(r["schema_name"]) for r in records})

    def get_outputs(record: dict):
        return (
            record["id"],
            record["text"],
            record["gold"],
            record["qwen"],
            record["gemma4"],
            record["openai"],
            record["presidio"],
        )

    def select_record(label_type: str, sample_no: float):
        choices = index["choices_by_label"].get(label_type) or index["choices_by_label"]["all"]
        choice = choices[max(min(int(sample_no), len(choices)), 1) - 1]
        return get_outputs(index["by_id"][choice])

    def filter_samples(label_type: str):
        choices = index["choices_by_label"].get(label_type) or index["choices_by_label"]["all"]
        return (gr.update(maximum=len(choices), value=1), *select_record(label_type, 1))

    def move_sample(label_type: str, sample_no: float, delta: int):
        choices = index["choices_by_label"].get(label_type) or index["choices_by_label"]["all"]
        next_no = (max(min(int(sample_no), len(choices)), 1) - 1 + delta) % len(choices) + 1
        return (gr.update(value=next_no), *select_record(label_type, next_no))

    outputs_def = lambda: [sample_id, input_text, gold_text, openai_text, presidio_text, qwen_text, gemma4_text]  # noqa: E731

    with gr.Blocks(title="compare_viewer") as app:
        gr.Markdown("# Privacy Filter — 4-Method Compare Viewer")

        with gr.Row():
            label_type = gr.Dropdown(choices=label_types, value="all", label="label")
            sample_no = gr.Slider(minimum=1, maximum=len(index["choices"]), value=1, step=1, label="sample")
            sample_id = gr.Textbox(label="id", interactive=False, scale=2)
            prev_btn = gr.Button("◀ Prev")
            next_btn = gr.Button("Next ▶", variant="primary")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_text = gr.Textbox(label="input text", lines=6, interactive=False)
            with gr.Column(scale=1):
                gold_text = gr.HighlightedText(label="gold", combine_adjacent=True, show_legend=True)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                qwen_text = gr.HighlightedText(label="Qwen3.6-35B-A3B", combine_adjacent=True, show_legend=True)
            with gr.Column(scale=1):
                gemma4_text = gr.HighlightedText(label="Gemma4-26B-A4B", combine_adjacent=True, show_legend=True)
            with gr.Column(scale=1):
                openai_text = gr.HighlightedText(label="OpenAI privacy-filter", combine_adjacent=True, show_legend=True)
            with gr.Column(scale=1):
                presidio_text = gr.HighlightedText(label="Presidio", combine_adjacent=True, show_legend=True)

        all_outputs = [sample_id, input_text, gold_text, qwen_text, gemma4_text, openai_text, presidio_text]

        sample_no.change(select_record, inputs=[label_type, sample_no], outputs=all_outputs)
        label_type.change(filter_samples, inputs=[label_type], outputs=[sample_no] + all_outputs)
        prev_btn.click(lambda lt, sn: move_sample(lt, sn, -1), inputs=[label_type, sample_no], outputs=[sample_no] + all_outputs)
        next_btn.click(lambda lt, sn: move_sample(lt, sn, 1), inputs=[label_type, sample_no], outputs=[sample_no] + all_outputs)
        app.load(select_record, inputs=[label_type, sample_no], outputs=all_outputs)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all 4 privacy-masking methods.")
    parser.add_argument("--openai-predictions", type=Path, default=DEFAULT_OPENAI_PREDICTIONS)
    parser.add_argument("--presidio-predictions", type=Path, default=DEFAULT_PRESIDIO_PREDICTIONS)
    parser.add_argument("--qwen-predictions", type=Path, default=DEFAULT_QWEN_PREDICTIONS)
    parser.add_argument("--gemma4-predictions", type=Path, default=DEFAULT_GEMMA4_PREDICTIONS)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        args.openai_predictions,
        args.presidio_predictions,
        args.qwen_predictions,
        args.gemma4_predictions,
        args.model_path,
    )
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
