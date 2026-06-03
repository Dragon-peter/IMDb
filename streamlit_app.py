from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, MODEL_DIR, TrainConfig
from src.inference import predict_texts, prediction_to_row, warning_for_text


st.set_page_config(page_title="IMDb 情感分析系统", page_icon="🎬", layout="wide")

EXAMPLES = {
    "正面示例 1": "this movie is heartfelt, funny, and far better than i expected",
    "正面示例 2": "a beautifully written story with strong performances and a satisfying ending",
    "负面示例 1": "the acting is wooden and the story drags on without any emotional payoff",
    "负面示例 2": "i regret watching this because the plot is messy and incredibly boring",
}


def init_history() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def source_label(source: str) -> str:
    if source == "single_text":
        return "单条预测"
    if source == "multiline_text":
        return "多行文本"
    if source.startswith("file:"):
        return f"文件上传 ({source.removeprefix('file:')})"
    return source


def build_display_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame

    rename_map = {
        "created_at": "提交时间",
        "source": "来源",
        "text": "影评内容",
        "sentiment_cn": "情感结果",
        "label": "英文标签",
        "label_confidence": "最终类别置信度",
        "positive_probability": "正面概率",
        "negative_probability": "负面概率",
        "truncated": "是否截断",
        "english_ratio": "英文占比",
    }
    frame = frame.rename(columns=rename_map)

    for column in ("最终类别置信度", "正面概率", "负面概率", "英文占比"):
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: f"{float(value):.2%}")

    if "影评内容" in frame.columns:
        frame["影评内容"] = frame["影评内容"].map(
            lambda text: text if len(text) <= 120 else f"{text[:117]}..."
        )

    preferred_columns = [
        "提交时间",
        "来源",
        "影评内容",
        "情感结果",
        "英文标签",
        "最终类别置信度",
        "正面概率",
        "负面概率",
        "是否截断",
        "英文占比",
    ]
    available_columns = [column for column in preferred_columns if column in frame.columns]
    return frame[available_columns]


def append_history(source: str, results: list) -> None:
    for result in results:
        row = prediction_to_row(result)
        row["created_at"] = datetime.now().strftime("%H:%M:%S")
        row["source"] = source_label(source)
        row["sentiment_cn"] = "正面" if result.label == "positive" else "负面"
        st.session_state.prediction_history.insert(0, row)


def render_history() -> None:
    st.subheader("历史查询记录")
    st.caption("仅保存在当前页面会话中，重启程序后会自动清空。")
    history = st.session_state.prediction_history
    if not history:
        st.info("当前还没有历史记录。")
        return

    st.caption(f"当前共保存 {len(history)} 条预测记录。")
    if st.button("清空历史记录"):
        st.session_state.prediction_history = []
        st.rerun()

    history_frame = build_display_frame(history)
    st.dataframe(history_frame, use_container_width=True)


def render_overview() -> None:
    st.title("基于 BiLSTM 的电影评论情感分析系统")
    st.markdown(
        """
        本系统基于 **IMDb Large Movie Review Dataset** 训练，任务是对英文影评进行二分类：
        `positive` 或 `negative`。网页界面为中文说明，便于课程设计演示，但模型训练语料是英文影评。
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("数据集", "IMDb aclImdb v1.0")
    col2.metric("模型", "BiLSTM")
    col3.metric("输入口径", "英文影评")

    st.subheader("使用说明")
    st.markdown(
        """
        - 在“单条预测”页面输入英文影评，查看情感类别与概率。
        - 在“批量预测”页面上传 `csv/txt` 文件，或直接粘贴多行英文影评，批量得到分类结果。
        - 在“模型效果”页面查看准确率、精确率、召回率、F1、混淆矩阵和训练曲线。
        """
    )


def render_single_prediction() -> None:
    st.header("单条预测")
    selected = st.selectbox("快速填充示例", ["自定义输入", *EXAMPLES.keys()])
    default_text = "" if selected == "自定义输入" else EXAMPLES[selected]
    text = st.text_area("请输入英文影评", value=default_text, height=180, placeholder="type an english movie review here...")

    if st.button("开始预测", type="primary"):
        warnings = warning_for_text(text, TrainConfig().max_length)
        for message in warnings:
            if "输入为空" in message:
                st.error(message)
            else:
                st.warning(message)
        if not text.strip():
            return
        result = predict_texts([text])[0]
        append_history("single_text", [result])
        sentiment_cn = "正面" if result.label == "positive" else "负面"
        probability = (
            result.positive_probability if result.label == "positive" else result.negative_probability
        )

        col1, col2 = st.columns(2)
        col1.metric("预测类别", f"{sentiment_cn} ({result.label})")
        col2.metric("置信度", f"{probability:.2%}")

        detail_col1, detail_col2 = st.columns(2)
        detail_col1.metric("正面概率", f"{result.positive_probability:.2%}")
        detail_col2.metric("负面概率", f"{result.negative_probability:.2%}")

        explanation = (
            "模型判断该评论整体倾向积极，通常意味着文本中包含更多正向评价词。"
            if result.label == "positive"
            else "模型判断该评论整体倾向消极，通常意味着文本中包含更多负向评价词。"
        )
        st.info(explanation)
    render_history()


def render_batch_prediction() -> None:
    st.header("批量预测")
    st.caption("支持 CSV 或 TXT。CSV 必须包含 text 列；TXT 每行一条英文影评。也支持直接粘贴多行文本。")
    pasted_texts = st.text_area(
        "直接粘贴多行英文影评",
        height=180,
        placeholder="each line is one english review...",
    )
    if st.button("分析粘贴的多行文本"):
        lines = [line.strip() for line in pasted_texts.splitlines() if line.strip()]
        if not lines:
            st.error("请至少输入一行英文影评。")
        else:
            _render_batch_results(lines, source="multiline_text")

    st.divider()
    uploaded = st.file_uploader("上传批量预测文件", type=["csv", "txt"])
    if not uploaded:
        render_history()
        return

    suffix = Path(uploaded.name).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(uploaded)
            if "text" not in frame.columns:
                st.error("CSV 文件必须包含 text 列，例如：text")
                st.code("text\nthis movie is wonderful\nthis film is painfully dull")
                return
            texts = frame["text"].fillna("").astype(str).tolist()
        else:
            content = StringIO(uploaded.getvalue().decode("utf-8"))
            texts = [line.strip() for line in content.readlines() if line.strip()]
    except Exception as exc:
        st.error(f"文件读取失败：{exc}")
        return

    if st.button("执行批量预测"):
        _render_batch_results(texts, source=f"file:{uploaded.name}")
    render_history()


def _render_batch_results(texts: list[str], source: str) -> None:
    for index, text in enumerate(texts, start=1):
        for message in warning_for_text(text, TrainConfig().max_length):
            if "输入为空" not in message:
                st.warning(f"第 {index} 条：{message}")
    results = predict_texts(texts)
    append_history(source, results)
    result_frame = pd.DataFrame([prediction_to_row(result) for result in results])
    result_frame.insert(0, "line_no", list(range(1, len(result_frame) + 1)))
    result_frame.insert(3, "sentiment_cn", ["正面" if row == "positive" else "负面" for row in result_frame["label"]])

    display_frame = result_frame.rename(
        columns={
            "line_no": "序号",
            "text": "影评内容",
            "sentiment_cn": "情感结果",
            "label": "英文标签",
            "label_confidence": "最终类别置信度",
            "positive_probability": "正面概率",
            "negative_probability": "负面概率",
            "truncated": "是否截断",
            "english_ratio": "英文占比",
        }
    ).copy()
    for column in ("最终类别置信度", "正面概率", "负面概率", "英文占比"):
        display_frame[column] = display_frame[column].map(lambda value: f"{float(value):.2%}")

    st.dataframe(display_frame, use_container_width=True)
    csv_bytes = result_frame.to_csv(index=False).encode("utf-8")
    st.download_button("下载预测结果 CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")


def render_model_metrics() -> None:
    st.header("模型效果")
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    checkpoint_path = MODEL_DIR / "best.pt"
    if not checkpoint_path.exists():
        st.warning("尚未检测到 models/best.pt，请先完成训练。")
        return
    if not metrics_path.exists():
        st.warning("尚未检测到评估结果，请先运行 evaluate.py。")
        return

    metrics = pd.read_json(metrics_path, typ="series")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1", f"{metrics['f1']:.2%}")

    for image_name, title in (
        ("confusion_matrix.png", "混淆矩阵"),
        ("loss_curve.png", "训练/验证损失曲线"),
        ("accuracy_curve.png", "训练/验证准确率曲线"),
    ):
        image_path = ARTIFACTS_DIR / image_name
        if image_path.exists():
            st.subheader(title)
            st.image(str(image_path))


def render_examples_page() -> None:
    st.header("示例案例")
    rows = []
    for name, text in EXAMPLES.items():
        rows.append({"示例名称": name, "英文影评": text})
    st.table(pd.DataFrame(rows))


def main() -> None:
    init_history()
    page = st.sidebar.radio(
        "页面导航",
        ["项目说明", "单条预测", "批量预测", "模型效果", "示例案例"],
    )
    if page == "项目说明":
        render_overview()
    elif page == "单条预测":
        render_single_prediction()
    elif page == "批量预测":
        render_batch_prediction()
    elif page == "模型效果":
        render_model_metrics()
    else:
        render_examples_page()


if __name__ == "__main__":
    main()
