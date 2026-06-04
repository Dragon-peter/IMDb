from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import ARTIFACTS_DIR, TrainConfig, ensure_directories
from src.data import SentimentDataset, load_examples
from src.inference import load_checkpoint
from src.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    probabilities_to_labels,
    save_metrics,
)
from src.trainer import save_training_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained BiLSTM model.")
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    model, vocab, config, device, payload = load_checkpoint()
    _, _, test_examples = load_examples(
        TrainConfig(**payload["config"]),
        max_test_samples=args.max_test_samples,
    )
    dataset = SentimentDataset(test_examples, vocab, config.max_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    labels: list[int] = []
    probabilities: list[float] = []
    rows: list[dict] = []
    decision_threshold = float(payload.get("decision_threshold", 0.5))

    model.eval()
    with torch.no_grad():
        for batch, example_batch in zip(loader, _batched_examples(test_examples, config.batch_size)):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            logits = model(input_ids, lengths)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            batch_labels = batch["label"].cpu().numpy().astype(int).tolist()

            probabilities.extend(probs)
            labels.extend(batch_labels)

            for example, prob, label in zip(example_batch, probs, batch_labels):
                predicted = int(prob >= decision_threshold)
                confidence = prob if predicted == 1 else 1.0 - prob
                rows.append(
                    {
                        "text": example.text,
                        "true_label": "positive" if label == 1 else "negative",
                        "predicted_label": "positive" if predicted == 1 else "negative",
                        "probability": round(prob, 6),
                        "label_confidence": round(confidence, 6),
                        "margin_to_threshold": round(abs(prob - decision_threshold), 6),
                    }
                )

    predictions = probabilities_to_labels(torch.tensor(probabilities).numpy(), threshold=decision_threshold)
    metrics = compute_classification_metrics(labels, predictions)
    metrics["test_size"] = len(test_examples)
    metrics["best_val_accuracy"] = float(payload.get("best_val_accuracy", 0.0))
    metrics["best_val_f1"] = float(payload.get("best_val_f1", 0.0))
    metrics["decision_threshold"] = decision_threshold

    if "history" in payload:
        plot_training_curves(payload["history"], ARTIFACTS_DIR)
    save_training_metadata(TrainConfig(**payload["config"]), ARTIFACTS_DIR / "train_config.json")
    save_metrics(metrics, ARTIFACTS_DIR / "metrics.json")
    plot_confusion_matrix(labels, predictions, ARTIFACTS_DIR / "confusion_matrix.png")
    result_frame = pd.DataFrame(rows)
    result_frame.head(200).to_csv(ARTIFACTS_DIR / "sample_predictions.csv", index=False)
    misclassified = result_frame[result_frame["true_label"] != result_frame["predicted_label"]].copy()
    if not misclassified.empty:
        misclassified = misclassified.sort_values(by="label_confidence", ascending=False)
    misclassified.head(100).to_csv(ARTIFACTS_DIR / "misclassified_examples.csv", index=False)
    uncertain = result_frame.copy().sort_values(by="margin_to_threshold", ascending=True)
    uncertain.head(100).to_csv(ARTIFACTS_DIR / "uncertain_examples.csv", index=False)

    print("Evaluation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def _batched_examples(examples, batch_size):
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


if __name__ == "__main__":
    main()
