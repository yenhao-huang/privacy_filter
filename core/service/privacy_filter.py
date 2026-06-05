"""Detection and redaction primitives for sensitive text."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable


DEFAULT_REPLACEMENT = "[REDACTED:{kind}]"


@dataclass(frozen=True)
class PrivacyPattern:
    kind: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class PrivacyMatch:
    kind: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class PatternConfig:
    patterns: tuple[PrivacyPattern, ...]
    replacement: str = DEFAULT_REPLACEMENT


def _compile_patterns(raw_patterns: Iterable[dict[str, str]]) -> tuple[PrivacyPattern, ...]:
    patterns: list[PrivacyPattern] = []
    for raw in raw_patterns:
        kind = raw.get("kind")
        pattern = raw.get("pattern")
        if not kind or not pattern:
            raise ValueError("Each privacy pattern requires 'kind' and 'pattern'.")
        patterns.append(PrivacyPattern(kind=kind, regex=re.compile(pattern)))
    return tuple(patterns)


def default_pattern_config() -> PatternConfig:
    return PatternConfig(
        patterns=_compile_patterns(
            [
                {
                    "kind": "email",
                    "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                },
                {
                    "kind": "phone",
                    "pattern": r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
                },
                {"kind": "ssn", "pattern": r"\b\d{3}-\d{2}-\d{4}\b"},
                {"kind": "credit_card", "pattern": r"\b(?:\d[ -]*?){13,19}\b"},
                {
                    "kind": "api_key",
                    "pattern": r"\b(?:api[_-]?key|token|secret)\s*[:=]\s*[A-Za-z0-9_\-]{16,}\b",
                },
            ]
        )
    )


def load_pattern_config(path: Path) -> PatternConfig:
    if not path.exists():
        return default_pattern_config()

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    replacement = payload.get("replacement", DEFAULT_REPLACEMENT)
    raw_patterns = payload.get("patterns", [])
    if not isinstance(raw_patterns, list):
        raise ValueError("'patterns' must be a list.")

    return PatternConfig(patterns=_compile_patterns(raw_patterns), replacement=replacement)


class PrivacyFilter:
    def __init__(
        self,
        patterns: Iterable[PrivacyPattern] | None = None,
        replacement: str = DEFAULT_REPLACEMENT,
    ) -> None:
        config = default_pattern_config()
        self._patterns = tuple(patterns) if patterns is not None else config.patterns
        self._replacement = replacement

    def find(self, text: str) -> list[PrivacyMatch]:
        matches: list[PrivacyMatch] = []
        for pattern in self._patterns:
            for match in pattern.regex.finditer(text):
                matches.append(
                    PrivacyMatch(
                        kind=pattern.kind,
                        value=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return _without_overlaps(matches)

    def redact(self, text: str) -> str:
        matches = self.find(text)
        if not matches:
            return text

        redacted: list[str] = []
        cursor = 0
        for match in matches:
            redacted.append(text[cursor : match.start])
            redacted.append(self._replacement.format(kind=match.kind))
            cursor = match.end
        redacted.append(text[cursor:])
        return "".join(redacted)


def _without_overlaps(matches: list[PrivacyMatch]) -> list[PrivacyMatch]:
    ordered = sorted(matches, key=lambda item: (item.start, -(item.end - item.start)))
    filtered: list[PrivacyMatch] = []
    occupied_until = -1

    for match in ordered:
        if match.start < occupied_until:
            continue
        filtered.append(match)
        occupied_until = match.end

    return filtered

