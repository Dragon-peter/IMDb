from __future__ import annotations

from dataclasses import dataclass
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
    truncated: bool
    english_ratio: float


def load_checkpoint(checkpoint_path: Path | None = None) -> tuple[BiLSTMSentimentClassifier, Vocabulary, TrainConfig, torch.device, dict]:
    checkpoint_path = checkpoint_path or (MODEL_DIR / "best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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


def predict_texts(texts: list[str], checkpoint_path: Path | None = None) -> list[PredictionResult]:
    model, vocab, config, device, _ = load_checkpoint(checkpoint_path)
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
        label_id = int(probability >= 0.5)
        results.append(
            PredictionResult(
                text=meta["text"],
                label=decode_label(label_id),
                positive_probability=float(probability),
                negative_probability=float(1.0 - probability),
                truncated=bool(meta["truncated"]),
                english_ratio=float(meta["english_ratio"]),
            )
        )
    return results


def warning_for_text(text: str, max_length: int) -> list[str]:
    warnings: list[str] = []
    tokens = tokenize(text)
    if not text.strip():
        warnings.append("输入为空，请先输入英文影评。")
    if text.strip() and not mostly_english(text):
        warnings.append("该模型基于 IMDb 英文影评训练，建议输入英文评论。")
    if len(tokens) > max_length:
        warnings.append(f"输入过长，系统将自动截断至前 {max_length} 个 tokens。")
    return warnings


def prediction_to_row(result: PredictionResult) -> dict:
    label_confidence = (
        result.positive_probability if result.label == "positive" else result.negative_probability
    )
    return {
        "text": result.text,
        "label": result.label,
        "positive_probability": round(result.positive_probability, 6),
        "negative_probability": round(result.negative_probability, 6),
        "label_confidence": round(label_confidence, 6),
        "truncated": result.truncated,
        "english_ratio": round(result.english_ratio, 4),
    }
