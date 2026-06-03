from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, TrainConfig
from src.inference import predict_texts, prediction_to_row, warning_for_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict IMDb sentiment from text or file.")
    parser.add_argument("--text", type=str, default=None, help="Single text to classify.")
    parser.add_argument("--input", type=str, default=None, help="CSV or TXT file for batch prediction.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ARTIFACTS_DIR / "sample_predictions.csv"),
        help="Output CSV path for batch prediction.",
    )
    return parser.parse_args()


def _load_batch_texts(file_path: Path) -> list[str]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(file_path)
        if "text" not in frame.columns:
            raise ValueError("CSV 文件必须包含 text 列，例如：text")
        return frame["text"].fillna("").astype(str).tolist()
    if suffix == ".txt":
        return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError("仅支持 csv 或 txt 文件。")


def _print_text_warnings(texts: list[str]) -> None:
    max_length = TrainConfig().max_length
    for index, text in enumerate(texts, start=1):
        warnings = warning_for_text(text, max_length)
        for message in warnings:
            print(f"[warning][{index}] {message}")


def main() -> None:
    args = parse_args()
    if not args.text and not args.input:
        raise ValueError("请使用 --text 或 --input 提供待预测内容。")

    if args.text:
        warnings = warning_for_text(args.text, TrainConfig().max_length)
        for message in warnings:
            print(f"[warning] {message}")
        result = predict_texts([args.text])[0]
        print(prediction_to_row(result))
        return

    input_path = Path(args.input)
    texts = _load_batch_texts(input_path)
    _print_text_warnings(texts)
    results = predict_texts(texts)
    frame = pd.DataFrame([prediction_to_row(result) for result in results])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
