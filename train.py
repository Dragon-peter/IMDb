from __future__ import annotations

import argparse

from src.config import TrainConfig, ensure_directories
from src.trainer import train_model


# Author info placeholder for report screenshots:
# Name: YOUR_NAME
# Student ID: YOUR_STUDENT_ID
# Class: YOUR_CLASS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM sentiment classifier on IMDb.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    result = train_model(
        config,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    print("Training finished.")
    for key, value in result.items():
        if key != "history":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
