from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ARTIFACTS_DIR, MODEL_DIR, TrainConfig, ensure_directories
from .data import SentimentDataset, Vocabulary, build_vocabulary, load_examples
from .metrics import plot_training_curves, search_best_threshold
from .model import BiLSTMSentimentClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(config: TrainConfig, vocab: Vocabulary, device: torch.device) -> BiLSTMSentimentClassifier:
    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab.stoi),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_index=vocab.pad_index,
    )
    return model.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    collect_outputs: bool = False,
) -> tuple[float, float] | tuple[float, float, list[int], list[float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    labels_out: list[int] = []
    probabilities_out: list[float] = []

    iterator = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["label"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_probabilities = torch.sigmoid(logits)
        predictions = (batch_probabilities >= 0.5).float()
        total_loss += loss.item() * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_count += labels.size(0)
        if collect_outputs:
            labels_out.extend(labels.detach().cpu().numpy().astype(int).tolist())
            probabilities_out.extend(batch_probabilities.detach().cpu().numpy().tolist())

    epoch_loss = total_loss / max(total_count, 1)
    epoch_accuracy = total_correct / max(total_count, 1)
    if collect_outputs:
        return epoch_loss, epoch_accuracy, labels_out, probabilities_out
    return epoch_loss, epoch_accuracy


def save_checkpoint(
    model: nn.Module,
    vocab: Vocabulary,
    config: TrainConfig,
    history: dict[str, list[float]],
    best_val_accuracy: float,
    decision_threshold: float,
    threshold_metric: float,
    output_path: Path,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "vocab": vocab.to_dict(),
        "config": asdict(config),
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "decision_threshold": decision_threshold,
        "best_val_f1": threshold_metric,
    }
    torch.save(payload, output_path)


def save_training_metadata(config: TrainConfig, output_path: Path) -> None:
    output_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def train_model(
    config: TrainConfig,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> dict:
    ensure_directories()
    set_seed(config.seed)
    device = get_device()

    train_examples, val_examples, _ = load_examples(
        config,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    vocab = build_vocabulary(train_examples, min_freq=config.min_freq)
    train_dataset = SentimentDataset(train_examples, vocab, config.max_length)
    val_dataset = SentimentDataset(val_examples, vocab, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(config, vocab, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_threshold": [],
    }
    best_val_accuracy = 0.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    best_path = MODEL_DIR / "best.pt"
    save_training_metadata(config, ARTIFACTS_DIR / "train_config.json")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy, val_labels, val_probabilities = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            collect_outputs=True,
        )
        threshold_summary = search_best_threshold(np.asarray(val_probabilities), val_labels)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(threshold_summary["f1"])
        history["val_threshold"].append(threshold_summary["threshold"])

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"val_f1={threshold_summary['f1']:.4f} threshold={threshold_summary['threshold']:.2f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_threshold = threshold_summary["threshold"]
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                vocab,
                config,
                history,
                best_val_accuracy,
                best_threshold,
                threshold_summary["f1"],
                best_path,
            )
            plot_training_curves(history, ARTIFACTS_DIR)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print("Early stopping triggered.")
                break

    plot_training_curves(history, ARTIFACTS_DIR)

    return {
        "device": str(device),
        "vocab_size": len(vocab.stoi),
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "best_val_accuracy": best_val_accuracy,
        "decision_threshold": best_threshold,
        "history": history,
        "checkpoint_path": str(best_path),
    }
