"""Command-line interface for privacy filtering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.service.privacy_filter import PrivacyFilter, load_pattern_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Redact common private values from text.")
    parser.add_argument("text", nargs="*", help="Text to redact. Reads stdin when omitted.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/privacy_patterns.json"),
        help="Path to a JSON pattern config.",
    )
    parser.add_argument(
        "--list-matches",
        action="store_true",
        help="Print detected matches instead of redacted text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = " ".join(args.text) if args.text else sys.stdin.read()

    config = load_pattern_config(args.config)
    filterer = PrivacyFilter(patterns=config.patterns, replacement=config.replacement)

    if args.list_matches:
        for match in filterer.find(text):
            print(f"{match.kind}\t{match.start}:{match.end}\t{match.value}")
        return 0

    print(filterer.redact(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

