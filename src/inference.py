from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .config import MODEL_DIR, TrainConfig
from .data import Vocabulary, decode_label
from .model import BiLSTMSentimentClassifier
from .text_utils import english_ratio, mostly_english, tokenize
from .trainer import get_device


@dataclass
class PredictionResult:
    text: str
    label: str
    positive_probability: float
    negative_probability: float
    label_confidence: float
    truncated: bool
    english_ratio: float
    decision_threshold: float


@dataclass
class TextValidation:
    text: str
    errors: list[str]
    warnings: list[str]
    english_ratio: float
    token_count: int
    truncated: bool

    @property
    def is_valid(self) -> bool:
        return not self.errors


@lru_cache(maxsize=4)
def _load_checkpoint_cached(cache_key: str, checkpoint_str: str) -> tuple[BiLSTMSentimentClassifier, Vocabulary, TrainConfig, torch.device, dict]:
    del cache_key
    checkpoint_path = Path(checkpoint_str)
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**payload["config"])
    vocab = Vocabulary.from_dict(payload["vocab"])
    device = get_device()
    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab.stoi),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_index=vocab.pad_index,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, vocab, config, device, payload


def load_checkpoint(checkpoint_path: Path | None = None) -> tuple[BiLSTMSentimentClassifier, Vocabulary, TrainConfig, torch.device, dict]:
    checkpoint_path = checkpoint_path or (MODEL_DIR / "best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    resolved = checkpoint_path.resolve()
    cache_key = f"{resolved}:{resolved.stat().st_mtime_ns}"
    return _load_checkpoint_cached(cache_key, str(resolved))


def predict_texts(texts: list[str], checkpoint_path: Path | None = None) -> list[PredictionResult]:
    if not texts:
        return []
    model, vocab, config, device, _ = load_checkpoint(checkpoint_path)
    decision_threshold = float(_.get("decision_threshold", 0.5))
    results: list[PredictionResult] = []
    encoded_batch = []
    lengths = []
    metadata = []

    for text in texts:
        tokens = tokenize(text)
        encoded, length, truncated = vocab.encode(tokens, config.max_length)
        encoded_batch.append(encoded)
        lengths.append(max(length, 1))
        metadata.append(
            {
                "text": text,
                "truncated": truncated,
                "english_ratio": english_ratio(text),
            }
        )

    input_ids = torch.tensor(encoded_batch, dtype=torch.long, device=device)
    input_lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids, input_lengths)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()

    for meta, probability in zip(metadata, probabilities):
        positive_probability = float(probability)
        negative_probability = float(1.0 - probability)
        label_id = int(positive_probability >= decision_threshold)
        label_confidence = positive_probability if label_id == 1 else negative_probability
        results.append(
            PredictionResult(
                text=meta["text"],
                label=decode_label(label_id),
                positive_probability=positive_probability,
                negative_probability=negative_probability,
                label_confidence=float(label_confidence),
                truncated=bool(meta["truncated"]),
                english_ratio=float(meta["english_ratio"]),
                decision_threshold=decision_threshold,
            )
        )
    return results


def validate_text_for_prediction(text: str, max_length: int, english_threshold: float = 0.7) -> TextValidation:
    errors: list[str] = []
    warnings: list[str] = []
    tokens = tokenize(text)
    ratio = english_ratio(text)
    if not text.strip():
        errors.append("输入为空，请先输入英文影评。")
    elif not mostly_english(text, threshold=english_threshold):
        errors.append("当前输入英文占比过低，系统仅支持英文影评预测。")
    if text.strip() and not tokens:
        errors.append("未识别到有效英文 tokens，请输入完整的英文影评句子。")
    truncated = len(tokens) > max_length
    if truncated:
        warnings.append(f"输入过长，系统将自动截断至前 {max_length} 个 tokens。")
    return TextValidation(
        text=text,
        errors=errors,
        warnings=warnings,
        english_ratio=ratio,
        token_count=len(tokens),
        truncated=truncated,
    )


def warning_for_text(text: str, max_length: int) -> list[str]:
    validation = validate_text_for_prediction(text, max_length)
    return [*validation.errors, *validation.warnings]


def prediction_to_row(result: PredictionResult) -> dict:
    return {
        "text": result.text,
        "label": result.label,
        "positive_probability": round(result.positive_probability, 6),
        "negative_probability": round(result.negative_probability, 6),
        "label_confidence": round(result.label_confidence, 6),
        "truncated": result.truncated,
        "english_ratio": round(result.english_ratio, 4),
        "decision_threshold": round(result.decision_threshold, 4),
    }
