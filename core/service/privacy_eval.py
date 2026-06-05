"""Span-level evaluation helpers for privacy-filter predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json


OPENAI_PRIVACY_LABELS = {
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
}


@dataclass(frozen=True, order=True)
class Span:
    label: str
    start: int
    end: int
    text: str

    def key(self) -> tuple[str, int, int]:
        return (self.label, self.start, self.end)


def load_gold_spans(raw_spans: str) -> list[Span]:
    spans = json.loads(raw_spans)
    normalized: list[Span] = []
    for span in spans:
        label = span.get("expected_model_label") or span.get("label")
        if label not in OPENAI_PRIVACY_LABELS:
            continue
        normalized.append(
            Span(
                label=label,
                start=int(span["start"]),
                end=int(span["end"]),
                text=str(span.get("text", "")),
            )
        )
    return sorted(normalized)


def labels_to_spans(text: str, labels: list[str], offsets: list[tuple[int, int]]) -> list[Span]:
    spans: list[Span] = []
    active_label: str | None = None
    active_start: int | None = None
    active_end: int | None = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is not None and active_start is not None and active_end is not None:
            spans.append(
                Span(
                    label=active_label,
                    start=active_start,
                    end=active_end,
                    text=text[active_start:active_end],
                )
            )
        active_label = None
        active_start = None
        active_end = None

    for label, (start, end) in zip(labels, offsets, strict=False):
        if start == end:
            continue
        if label == "O" or "-" not in label:
            close_active()
            continue

        prefix, entity = label.split("-", 1)
        if entity not in OPENAI_PRIVACY_LABELS:
            close_active()
            continue

        if prefix == "S":
            close_active()
            spans.append(Span(label=entity, start=start, end=end, text=text[start:end]))
        elif prefix == "B":
            close_active()
            active_label = entity
            active_start = start
            active_end = end
        elif prefix == "I":
            if active_label == entity:
                active_end = end
            else:
                close_active()
                active_label = entity
                active_start = start
                active_end = end
        elif prefix == "E":
            if active_label == entity:
                active_end = end
                close_active()
            else:
                spans.append(Span(label=entity, start=start, end=end, text=text[start:end]))
        else:
            close_active()

    close_active()
    return sorted(spans, key=lambda span: (span.start, span.end, span.label))


def exact_match_counts(gold: list[Span], predicted: list[Span]) -> tuple[list[Span], list[Span], list[Span]]:
    gold_by_key = {span.key(): span for span in gold}
    predicted_by_key = {span.key(): span for span in predicted}

    true_positive_keys = set(gold_by_key) & set(predicted_by_key)
    false_negative_keys = set(gold_by_key) - set(predicted_by_key)
    false_positive_keys = set(predicted_by_key) - set(gold_by_key)

    true_positives = [gold_by_key[key] for key in sorted(true_positive_keys)]
    false_negatives = [gold_by_key[key] for key in sorted(false_negative_keys)]
    false_positives = [predicted_by_key[key] for key in sorted(false_positive_keys)]
    return true_positives, false_positives, false_negatives


def summarize_counts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    labels = sorted(OPENAI_PRIVACY_LABELS)
    counters: dict[str, Counter[str]] = {label: Counter() for label in labels}

    for row in rows:
        label = str(row["label"])
        outcome = str(row["outcome"])
        counters[label][outcome] += 1

    summary = []
    for label in labels:
        counter = counters[label]
        tp = counter["tp"]
        fp = counter["fp"]
        fn = counter["fn"]
        accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "accuracy": round(accuracy, 6),
                "f1": round(f1, 6),
            }
        )
    return summary
