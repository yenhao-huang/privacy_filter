"""Mask benchmark text through a vLLM/OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://192.168.1.78:3132"
DEFAULT_MODEL = "/workspace/llm_model"
DEFAULT_INPUT = Path("data/benchmark/tw-pii-openai-in-shcema.csv")
DEFAULT_OUTPUT = Path("results/tw-pii-vllm-compatible_masked.csv")
DEFAULT_PROMPT_FILE = Path("configs/direct_replace.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mask benchmark text with a vLLM/OpenAI-compatible chat completions server."
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
        fieldnames = list(rows[0].keys()) + [
            "masked_text",
            "vllm_model",
            "vllm_usage_json",
            "vllm_raw_response_json",
        ]
        with args.output.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                response = mask_text(
                    client=client,
                    base_url=base_url,
                    model=model,
                    text=row["text"],
                    system_prompt=system_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                row["masked_text"] = extract_message_content(response)
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


def fetch_default_model(client: httpx.Client, base_url: str) -> str:
    try:
        response = client.get(f"{base_url}/v1/models")
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError:
        return DEFAULT_MODEL
    data = payload.get("data") or payload.get("models") or []
    if not data:
        return DEFAULT_MODEL
    first = data[0]
    model = first.get("id") or first.get("model") or first.get("name")
    if not model:
        return DEFAULT_MODEL
    return str(model)


def mask_text(
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


def extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content", "")).strip()


if __name__ == "__main__":
    raise SystemExit(main())
