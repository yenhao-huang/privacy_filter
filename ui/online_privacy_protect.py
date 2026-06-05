"""Online Gradio UI for privacy protection with OpenAI or Qwen backends."""

from __future__ import annotations

import argparse
from functools import lru_cache
import re
import sys
from pathlib import Path

import gradio as gr
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.service.openai_privacy_filter_model import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    LocalOpenAIPrivacyFilter,
)
from core.service.privacy_eval import labels_to_spans  # noqa: E402
from core.api.predict_by_vllm_compatible import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL as DEFAULT_QWEN_MODEL,
    DEFAULT_PROMPT_FILE,
    extract_message_content,
    fetch_default_model,
    mask_text,
)


DEFAULT_TEXT = "您好，我是陳俊宏，電話是0912-345-678，email 是 chun@example.com。"
PLACEHOLDER_RE = re.compile(
    r"\[(account_number|private_address|private_date|private_email|private_person|private_phone|private_url|secret)\]"
)


@lru_cache(maxsize=1)
def get_predictor(model_path: str, device_map: str, torch_dtype: str) -> LocalOpenAIPrivacyFilter:
    return LocalOpenAIPrivacyFilter(
        model_path=model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )


def predict(
    text: str,
    engine: str,
    model_path: str,
    device_map: str,
    torch_dtype: str,
    qwen_base_url: str,
    qwen_model: str,
    prompt_file: str,
):
    if not text.strip():
        return [], [], "", []

    if engine == "qwen":
        return predict_qwen(text, qwen_base_url, qwen_model, prompt_file)

    predictor = get_predictor(model_path, device_map, torch_dtype)
    token_predictions = predictor.predict_token_classes(text)
    labels = [prediction.label for prediction in token_predictions]
    offsets = predictor.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    spans = labels_to_spans(text, labels, offsets)

    redacted = redact_text(text, spans)
    highlighted = masked_text_segments(redacted)
    span_rows = [[span.label, span.start, span.end, span.text] for span in spans]
    token_rows = [
        [prediction.token_index, prediction.token, prediction.label]
        for prediction in token_predictions
        if prediction.label != "O"
    ]

    return highlighted, span_rows, token_rows, redacted


def predict_qwen(text: str, base_url: str, model: str, prompt_file: str):
    normalized_base_url = base_url.rstrip("/")
    system_prompt = Path(prompt_file).read_text(encoding="utf-8")
    with httpx.Client(timeout=120.0) as client:
        resolved_model = model.strip() or fetch_default_model(client, normalized_base_url)
        response = mask_text(
            client=client,
            base_url=normalized_base_url,
            model=resolved_model,
            text=text,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=512,
        )
    masked = extract_message_content(response)
    highlighted = masked_text_segments(masked)
    placeholders = placeholder_rows(masked)
    usage_rows = [["model", resolved_model]]
    for key, value in response.get("usage", {}).items():
        usage_rows.append([key, value])
    return highlighted, placeholders, usage_rows, masked


def redact_text(text: str, spans) -> str:
    if not spans:
        return text

    chunks: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        chunks.append(text[cursor : span.start])
        chunks.append(f"[{span.label}]")
        cursor = span.end
    chunks.append(text[cursor:])
    return "".join(chunks)


def highlighted_segments(text: str, spans) -> list[tuple[str, str | None]]:
    if not spans:
        return [(text, None)]

    chunks: list[tuple[str, str | None]] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        if cursor < span.start:
            chunks.append((text[cursor : span.start], None))
        chunks.append((text[span.start : span.end], span.label))
        cursor = span.end
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


def placeholder_rows(masked_text: str) -> list[list[object]]:
    return [
        [match.group(1), match.start(), match.end(), match.group(0)]
        for match in PLACEHOLDER_RE.finditer(masked_text)
    ]


def build_app(model_path: str, device_map: str, torch_dtype: str) -> gr.Blocks:
    with gr.Blocks(title="online_privacy_protect") as app:
        gr.Markdown("# online_privacy_protect")
        with gr.Row():
            text = gr.Textbox(
                label="Text",
                value=DEFAULT_TEXT,
                lines=8,
                scale=2,
            )
            with gr.Column(scale=1):
                engine = gr.Dropdown(
                    choices=["openai", "qwen"],
                    value="openai",
                    label="engine",
                )
                model = gr.Textbox(label="Model path", value=model_path)
                device = gr.Textbox(label="Device map", value=device_map)
                dtype = gr.Textbox(label="Torch dtype", value=torch_dtype)
                qwen_base = gr.Textbox(label="Qwen base URL", value=DEFAULT_BASE_URL)
                qwen_model = gr.Textbox(label="Qwen model", value=DEFAULT_QWEN_MODEL)
                prompt = gr.Textbox(label="Prompt file", value=str(DEFAULT_PROMPT_FILE))
                run = gr.Button("Predict", variant="primary")

        highlighted = gr.HighlightedText(
            label="Protected text",
            combine_adjacent=True,
            show_legend=True,
        )
        redacted = gr.Textbox(label="Masked text", lines=4)
        spans = gr.Dataframe(
            label="Predicted spans / placeholders",
            headers=["label", "start", "end", "text"],
            datatype=["str", "number", "number", "str"],
        )
        tokens = gr.Dataframe(
            label="OpenAI token labels / Qwen usage",
            headers=["key", "value", "label"],
            datatype=["str", "str", "str"],
        )

        run.click(
            predict,
            inputs=[text, engine, model, device, dtype, qwen_base, qwen_model, prompt],
            outputs=[highlighted, spans, tokens, redacted],
        )
        text.submit(
            predict,
            inputs=[text, engine, model, device, dtype, qwen_base, qwen_model, prompt],
            outputs=[highlighted, spans, tokens, redacted],
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args.model_path, args.device_map, args.torch_dtype)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
