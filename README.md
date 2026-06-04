# 基于 IMDb 数据集的 BiLSTM 影评情感分析系统

本项目基于 Stanford 发布的 IMDb Large Movie Review Dataset (`aclImdb v1.0`) 实现二分类情感分析，模型采用 `BiLSTM`，界面采用 `Streamlit`，演示口径为“中文界面 + 英文影评输入”。

当前版本额外包含：

- 英文输入硬校验，避免中文或空输入产生误导性结果
- 基于验证集的决策阈值优化，不再固定使用 `0.5`
- 历史记录、批量预测、正负概率详情、误判案例展示

## 技术栈

- Python 3.11
- PyTorch
- Streamlit
- pandas
- matplotlib
- scikit-learn
- tqdm

## 目录结构

```text
imdb-bilstm-sentiment-system/
├── artifacts/
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── scripts/
├── src/
├── evaluate.py
├── predict.py
├── requirements.txt
├── streamlit_app.py
└── train.py
```

## 环境准备

建议使用 Python 3.11 虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使用方式

1. 下载 IMDb 数据集

```bash
python scripts/download_imdb.py
```

2. 训练模型

```bash
python train.py --batch-size 256
```

3. 评估模型并生成图表

```bash
python evaluate.py
```

4. 命令行单条预测

```bash
python predict.py --text "this movie was surprisingly moving and beautifully acted"
```

5. 命令行批量预测

```bash
python predict.py --input sample_reviews.csv --output artifacts/sample_predictions.csv
```

6. 启动网页演示

```bash
streamlit run streamlit_app.py
```

## 批量预测输入格式

- `csv` 文件必须包含 `text` 列
- `txt` 文件每行一条英文影评

CSV 示例：

```csv
text
this movie is fantastic and emotionally rich
the plot is dull and the acting is terrible
```

## 主要产物

- `models/best.pt`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/loss_curve.png`
- `artifacts/accuracy_curve.png`
- `artifacts/sample_predictions.csv`
- `artifacts/misclassified_examples.csv`
- `artifacts/uncertain_examples.csv`

## 说明

- 本模型基于 IMDb 英文影评训练，仅支持英文评论输入。
- 当输入过长时，系统会自动截断到前 `256` 个 tokens。
- 训练阶段会根据验证集自动搜索更优的分类阈值，并在推理与评估阶段复用该阈值。
