from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_training_curves(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrix(labels: list[int], predictions: list[int], output_path: Path) -> None:
    matrix = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["negative", "positive"])
    ax.set_yticks([0, 1], labels=["negative", "positive"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def probabilities_to_labels(probabilities: np.ndarray, threshold: float = 0.5) -> list[int]:
    return [int(value >= threshold) for value in probabilities]


def search_best_threshold(
    probabilities: np.ndarray,
    labels: list[int],
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    if thresholds is None:
        thresholds = np.arange(0.30, 0.701, 0.02)

    labels_array = np.asarray(labels)
    best_threshold = 0.5
    best_metrics = compute_classification_metrics(labels, probabilities_to_labels(probabilities, threshold=0.5))
    best_score = best_metrics[metric]

    for threshold in thresholds:
        predictions = probabilities_to_labels(probabilities, threshold=float(threshold))
        current_metrics = compute_classification_metrics(labels, predictions)
        current_score = current_metrics[metric]
        if current_score > best_score:
            best_threshold = float(threshold)
            best_metrics = current_metrics
            best_score = current_score

    return {
        "threshold": float(best_threshold),
        "accuracy": float(best_metrics["accuracy"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "f1": float(best_metrics["f1"]),
    }
