"""Local OpenAI Privacy Filter model inference via Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL_PATH = "/models/openai-privacy-filter"


@dataclass(frozen=True)
class TokenPrediction:
    token_index: int
    token_id: int
    token: str
    label: str


class LocalOpenAIPrivacyFilter:
    """Thin wrapper around the local model.safetensors checkpoint."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device_map: str | None = "auto",
        torch_dtype: str | None = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers, and accelerate before running local model inference."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        kwargs: dict[str, Any] = {"local_files_only": True}
        if device_map:
            kwargs["device_map"] = device_map
        if torch_dtype:
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
        self.model.eval()

    @property
    def device(self) -> Any:
        return self.model.device

    def predict_token_classes(self, text: str) -> list[TokenPrediction]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        predicted_token_class_ids = outputs.logits.argmax(dim=-1)[0]
        input_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        predictions: list[TokenPrediction] = []
        for index, token_id in enumerate(predicted_token_class_ids):
            predictions.append(
                TokenPrediction(
                    token_index=index,
                    token_id=int(input_ids[index].item()),
                    token=tokens[index],
                    label=self.model.config.id2label[int(token_id.item())],
                )
            )
        return predictions

    def predict_label_sequence(self, text: str) -> list[str]:
        return [prediction.label for prediction in self.predict_token_classes(text)]

