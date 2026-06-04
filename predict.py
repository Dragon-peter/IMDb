from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, TrainConfig
from src.inference import predict_texts, prediction_to_row, validate_text_for_prediction


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
        validation = validate_text_for_prediction(text, max_length)
        for message in validation.errors:
            print(f"[error][{index}] {message}")
        for message in validation.warnings:
            print(f"[warning][{index}] {message}")


def main() -> None:
    args = parse_args()
    if not args.text and not args.input:
        raise ValueError("请使用 --text 或 --input 提供待预测内容。")

    if args.text:
        validation = validate_text_for_prediction(args.text, TrainConfig().max_length)
        for message in validation.errors:
            print(f"[error] {message}")
        for message in validation.warnings:
            print(f"[warning] {message}")
        if not validation.is_valid:
            raise ValueError("输入未通过校验，未执行预测。")
        result = predict_texts([args.text])[0]
        print(prediction_to_row(result))
        return

    input_path = Path(args.input)
    texts = _load_batch_texts(input_path)
    _print_text_warnings(texts)
    valid_texts = []
    invalid_rows = []
    for index, text in enumerate(texts, start=1):
        validation = validate_text_for_prediction(text, TrainConfig().max_length)
        if validation.is_valid:
            valid_texts.append(text)
        else:
            invalid_rows.append(
                {
                    "line_no": index,
                    "text": text,
                    "status": "invalid",
                    "reason": " | ".join(validation.errors),
                }
            )
    results = predict_texts(valid_texts)
    rows = [prediction_to_row(result) for result in results]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.insert(0, "status", "predicted")
    if invalid_rows:
        frame = pd.concat([frame, pd.DataFrame(invalid_rows)], ignore_index=True, sort=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
