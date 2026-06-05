"""Predict privacy spans through a vLLM/OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import httpx

from core.api.predict_by_vllm_compatible import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    fetch_default_model,
    extract_message_content,
)


DEFAULT_INPUT = Path("data/benchmark/tw-pii-openai-in-shcema.csv")
DEFAULT_OUTPUT = Path("results/tw-pii-vllm-compatible_span_predictions.csv")
DEFAULT_PROMPT_FILE = Path("configs/span_predict.txt")
ALLOWED_LABELS = {
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict privacy span JSON with a vLLM/OpenAI-compatible chat completions server."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model id. Defaults to first /v1/models id, then falls back to {DEFAULT_MODEL}.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_url = args.base_url.rstrip("/")

    with httpx.Client(timeout=args.timeout) as client:
        model = args.model or fetch_default_model(client, base_url)
        rows = read_rows(args.input, args.limit)
        system_prompt = args.prompt_file.read_text(encoding="utf-8")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        input_fieldnames = list(rows[0].keys()) if rows else []
        fieldnames = input_fieldnames + [
            "predicted_spans_json",
            "span_parse_error",
            "vllm_model",
            "vllm_usage_json",
            "vllm_raw_response_json",
        ]
        with args.output.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                response = predict_spans(
                    client=client,
                    base_url=base_url,
                    model=model,
                    text=row["text"],
                    system_prompt=system_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                content = extract_message_content(response)
                spans_json, parse_error = normalize_span_response(content, row["text"])
                row["predicted_spans_json"] = json.dumps(spans_json, ensure_ascii=False)
                row["span_parse_error"] = parse_error
                row["vllm_model"] = model
                row["vllm_usage_json"] = json.dumps(response.get("usage", {}), ensure_ascii=False)
                row["vllm_raw_response_json"] = json.dumps(response, ensure_ascii=False)
                writer.writerow(row)
                print(f"{index}/{len(rows)} {row['id']}")
                if args.sleep:
                    time.sleep(args.sleep)

    print(f"wrote {args.output}")
    return 0


def read_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return rows[:limit] if limit is not None else rows


def predict_spans(
    client: httpx.Client,
    base_url: str,
    model: str,
    text: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    response = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    response.raise_for_status()
    return response.json()


def normalize_span_response(content: str, text: str) -> tuple[dict[str, list[dict[str, int | str]]], str]:
    try:
        payload = json.loads(extract_json_text(content))
    except (json.JSONDecodeError, ValueError) as exc:
        return {"spans": []}, str(exc)

    if isinstance(payload, list):
        candidate_spans = payload
    elif isinstance(payload, dict):
        candidate_spans = payload.get("spans", [])
    else:
        return {"spans": []}, "JSON payload must be an object or span list"

    if not isinstance(candidate_spans, list):
        return {"spans": []}, "spans must be a list"

    spans: list[dict[str, int | str]] = []
    errors: list[str] = []
    for index, span in enumerate(candidate_spans):
        normalized, error = normalize_span(span, text)
        if error:
            errors.append(f"span {index}: {error}")
            continue
        spans.append(normalized)

    spans.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
    return {"spans": spans}, "; ".join(errors)


def extract_json_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        return stripped[object_start : object_end + 1]

    list_start = stripped.find("[")
    list_end = stripped.rfind("]")
    if list_start != -1 and list_end != -1 and list_end > list_start:
        return stripped[list_start : list_end + 1]

    raise ValueError("No JSON object or list found in model response")


def normalize_span(span: Any, text: str) -> tuple[dict[str, int | str], str]:
    if not isinstance(span, dict):
        return {}, "span must be an object"

    try:
        start = int(span["start"])
        end = int(span["end"])
        label = str(span["label"])
    except KeyError as exc:
        return {}, f"missing key {exc.args[0]}"
    except (TypeError, ValueError):
        return {}, "start and end must be integers"

    if label not in ALLOWED_LABELS:
        return {}, f"invalid label {label}"
    if start < 0 or end <= start or end > len(text):
        return {}, f"invalid offsets start={start} end={end}"

    return {"start": start, "end": end, "label": label}, ""


if __name__ == "__main__":
    raise SystemExit(main())
