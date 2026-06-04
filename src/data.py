from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import IMDB_EXTRACTED_DIR, PROCESSED_DIR, TrainConfig, ensure_directories
from .text_utils import build_counter, tokenize


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class Example:
    text: str
    label: int


class Vocabulary:
    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = {index: token for token, index in stoi.items()}
        self.pad_index = stoi[PAD_TOKEN]
        self.unk_index = stoi[UNK_TOKEN]

    @classmethod
    def build(cls, tokenized_texts: list[list[str]], min_freq: int) -> "Vocabulary":
        stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, count in build_counter(tokenized_texts).most_common():
            if count >= min_freq and token not in stoi:
                stoi[token] = len(stoi)
        return cls(stoi)

    def encode(self, tokens: list[str], max_length: int) -> tuple[list[int], int, bool]:
        truncated = len(tokens) > max_length
        trimmed = tokens[:max_length]
        ids = [self.stoi.get(token, self.unk_index) for token in trimmed]
        original_length = len(ids)
        if len(ids) < max_length:
            ids.extend([self.pad_index] * (max_length - len(ids)))
        return ids, original_length, truncated

    def to_dict(self) -> dict[str, int]:
        return self.stoi

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> "Vocabulary":
        return cls(payload)


class SentimentDataset(Dataset):
    def __init__(self, examples: list[Example], vocab: Vocabulary, max_length: int) -> None:
        self.examples = examples
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        tokens = tokenize(example.text)
        encoded, length, truncated = self.vocab.encode(tokens, self.max_length)
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "length": torch.tensor(max(length, 1), dtype=torch.long),
            "label": torch.tensor(example.label, dtype=torch.float32),
            "truncated": torch.tensor(int(truncated), dtype=torch.long),
        }


def _read_reviews(base_dir: Path) -> list[dict]:
    records: list[dict] = []
    for label_name, label_value in (("pos", 1), ("neg", 0)):
        label_dir = base_dir / label_name
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing directory: {label_dir}")
        for file_path in sorted(label_dir.glob("*.txt")):
            records.append(
                {
                    "path": str(file_path),
                    "label": label_value,
                }
            )
    return records


def _manifest_path() -> Path:
    return PROCESSED_DIR / "split_manifest.json"


def ensure_split_manifest(seed: int, val_ratio: float) -> Path:
    ensure_directories()
    manifest_path = _manifest_path()
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        same_seed = int(payload.get("seed", seed)) == seed
        same_ratio = float(payload.get("val_ratio", val_ratio)) == val_ratio
        if same_seed and same_ratio:
            return manifest_path

    if not IMDB_EXTRACTED_DIR.exists():
        raise FileNotFoundError(
            f"IMDb dataset not found at {IMDB_EXTRACTED_DIR}. Run scripts/download_imdb.py first."
        )

    train_records = _read_reviews(IMDB_EXTRACTED_DIR / "train")
    test_records = _read_reviews(IMDB_EXTRACTED_DIR / "test")
    train_labels = [record["label"] for record in train_records]
    train_split, val_split = train_test_split(
        train_records,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_labels,
        shuffle=True,
    )
    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train": train_split,
        "val": val_split,
        "test": test_records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _load_manifest(seed: int, val_ratio: float) -> dict:
    manifest_path = ensure_split_manifest(seed, val_ratio)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _materialize_examples(records: Iterable[dict], sample_limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    for index, record in enumerate(records):
        if sample_limit is not None and index >= sample_limit:
            break
        text = Path(record["path"]).read_text(encoding="utf-8")
        examples.append(Example(text=text, label=int(record["label"])))
    return examples


def load_examples(
    config: TrainConfig,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[list[Example], list[Example], list[Example]]:
    manifest = _load_manifest(config.seed, config.val_ratio)
    train_examples = _materialize_examples(manifest["train"], max_train_samples)
    val_examples = _materialize_examples(manifest["val"], max_val_samples)
    test_examples = _materialize_examples(manifest["test"], max_test_samples)
    return train_examples, val_examples, test_examples


def build_vocabulary(examples: list[Example], min_freq: int) -> Vocabulary:
    tokenized_texts = [tokenize(example.text) for example in examples]
    return Vocabulary.build(tokenized_texts, min_freq=min_freq)


def decode_label(label: int) -> str:
    return "positive" if label == 1 else "negative"
