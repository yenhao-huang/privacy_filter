"""Presidio-based privacy recognizers for Taiwan benchmark text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator
import re

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine


SUPPORTED_LANGUAGE = "en"
SUPPORTED_LABELS = {
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
}


@dataclass(frozen=True)
class RegexSpec:
    label: str
    pattern: str
    score: float
    group: int = 1


class NoOpNlpEngine(NlpEngine):
    """Minimal NLP engine for regex-only Presidio recognizers."""

    def load(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        return NlpArtifacts([], None, [], [], self, language)

    def process_batch(
        self,
        texts: Iterable[str],
        language: str,
        batch_size: int = 1,
        n_process: int = 1,
        **kwargs,
    ) -> Iterator[tuple[str, NlpArtifacts]]:
        for text in texts:
            yield text, self.process_text(text, language)

    def is_stopword(self, word: str, language: str) -> bool:
        return False

    def is_punct(self, word: str, language: str) -> bool:
        return False

    def get_supported_entities(self) -> list[str]:
        return []

    def get_supported_languages(self) -> list[str]:
        return [SUPPORTED_LANGUAGE]


class CaptureGroupRecognizer(EntityRecognizer):
    """Regex recognizer which returns the selected capture group span."""

    def __init__(self, spec: RegexSpec) -> None:
        super().__init__(
            supported_entities=[spec.label],
            name=f"tw_{spec.label}_capture_group",
            supported_language=SUPPORTED_LANGUAGE,
        )
        self.spec = spec
        self.regex = re.compile(spec.pattern, flags=re.IGNORECASE)

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts,
    ) -> list[RecognizerResult]:
        if self.spec.label not in entities:
            return []

        results: list[RecognizerResult] = []
        for match in self.regex.finditer(text):
            try:
                start = match.start(self.spec.group)
                end = match.end(self.spec.group)
            except IndexError:
                start = match.start()
                end = match.end()
            if start < 0 or end <= start:
                continue
            results.append(
                RecognizerResult(
                    entity_type=self.spec.label,
                    start=start,
                    end=end,
                    score=self.spec.score,
                )
            )
        return results


def build_presidio_analyzer() -> AnalyzerEngine:
    registry = RecognizerRegistry(supported_languages=[SUPPORTED_LANGUAGE])
    for spec in default_regex_specs():
        registry.add_recognizer(CaptureGroupRecognizer(spec))
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=NoOpNlpEngine(),
        supported_languages=[SUPPORTED_LANGUAGE],
        context_aware_enhancer=None,
    )


def default_regex_specs() -> list[RegexSpec]:
    return [
        RegexSpec("private_email", r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", 0.9),
        RegexSpec("private_phone", r"(?<!\d)(09\d{2}[-\s]?\d{3}[-\s]?\d{3})(?!\d)", 0.9),
        RegexSpec("private_phone", r"(?<!\d)(0800[-\s]?\d{3}[-\s]?\d{3})(?!\d)", 0.9),
        RegexSpec("private_phone", r"(?<!\d)((?:0[2-8]|0\d{2,3})[-\s]?\d{3,4}[-\s]?\d{4})(?!\d)", 0.82),
        RegexSpec(
            "private_url",
            r"((?:https?://)?(?:[A-Za-z0-9-]+\.)+(?:tw|com|net|org|edu|gov)(?:/[^\s，。；、)）]*)?)",
            0.85,
        ),
        RegexSpec("account_number", r"\b([A-Z]\d{8,10})\b", 0.86),
        RegexSpec("account_number", r"(?<!\d)(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})(?!\d)", 0.88),
        RegexSpec(
            "account_number",
            r"(?:帳號|卡號|學號|員工編號|會員編號|末五碼|代碼)[^\dA-Za-z]{0,6}([A-Z]?\d[\dA-Za-z-]{4,24})",
            0.78,
        ),
        RegexSpec(
            "secret",
            r"\b(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b",
            0.95,
        ),
        RegexSpec("secret", r"\b((?:sk|pk|api|key|token|secret)[-_A-Za-z0-9]{12,})\b", 0.9),
        RegexSpec("secret", r"(?:密碼|password|passwd|pwd)[：:=\s]*([^\s，。；、]{4,})", 0.86),
        RegexSpec("secret", r"(?:OTP|驗證碼|CVV)[：:=\s]*(\d{3,8})", 0.86),
        RegexSpec("private_date", r"((?:民國)?\d{2,3}年\d{1,2}月\d{1,2}日)", 0.82),
        RegexSpec("private_date", r"(\d{4}年\d{1,2}月(?:\d{1,2}日)?)", 0.82),
        RegexSpec("private_date", r"(\d{1,2}月\d{1,2}日)", 0.72),
        RegexSpec("private_date", r"(農曆[^\s，。；、]{1,10})", 0.78),
        RegexSpec(
            "private_address",
            r"((?:\d{3}(?:-\d{3})?\s*)?[\u4e00-\u9fff]{2,3}[縣市][\u4e00-\u9fff]{1,4}[鄉鎮市區][^，。；、\n]{0,45}(?:號|樓))",
            0.86,
        ),
        RegexSpec(
            "private_person",
            r"我是[^，。；、\n]{0,12}的([\u4e00-\u9fff·]{2,4})",
            0.9,
        ),
        RegexSpec(
            "private_person",
            r"我是([\u4e00-\u9fff·]{2,4})(?:，|,|。|、|\s|$)",
            0.82,
        ),
        RegexSpec(
            "private_person",
            r"(?:戶名是|姓名是|名字是|聯絡人是|收件人是|寄件人是|病患是|學生是)([\u4e00-\u9fff·]{2,6})",
            0.72,
        ),
        RegexSpec(
            "private_person",
            r"(?:name is|my name is|contact is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
            0.72,
        ),
    ]


def predict_spans(text: str, analyzer: AnalyzerEngine | None = None) -> list[dict[str, object]]:
    engine = analyzer or build_presidio_analyzer()
    results = engine.analyze(
        text=text,
        entities=sorted(SUPPORTED_LABELS),
        language=SUPPORTED_LANGUAGE,
    )
    spans = [
        {
            "label": result.entity_type,
            "start": int(result.start),
            "end": int(result.end),
            "text": text[result.start : result.end],
            "score": round(float(result.score), 3),
        }
        for result in results
        if result.entity_type in SUPPORTED_LABELS and result.end > result.start
    ]
    return without_overlaps(spans)


def without_overlaps(spans: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(
        spans,
        key=lambda span: (
            int(span["start"]),
            -float(span["score"]),
            -(int(span["end"]) - int(span["start"])),
        ),
    )
    kept: list[dict[str, object]] = []
    occupied_until = -1
    for span in ordered:
        start = int(span["start"])
        end = int(span["end"])
        if start < occupied_until:
            continue
        kept.append(span)
        occupied_until = end
    return sorted(kept, key=lambda span: (int(span["start"]), int(span["end"]), str(span["label"])))


def mask_text(text: str, spans: list[dict[str, object]]) -> str:
    chunks: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: (int(item["start"]), int(item["end"]))):
        start = int(span["start"])
        end = int(span["end"])
        if cursor < start:
            chunks.append(text[cursor:start])
        chunks.append(f"[{span['label']}]")
        cursor = max(cursor, end)
    if cursor < len(text):
        chunks.append(text[cursor:])
    return "".join(chunks)
