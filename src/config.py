from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE = RAW_DIR / "aclImdb_v1.tar.gz"
IMDB_EXTRACTED_DIR = RAW_DIR / "aclImdb"


@dataclass
class TrainConfig:
    seed: int = 42
    val_ratio: float = 0.1
    min_freq: int = 2
    max_length: int = 256
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    batch_size: int = 256
    epochs: int = 8
    learning_rate: float = 1e-3
    patience: int = 2
    num_workers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
