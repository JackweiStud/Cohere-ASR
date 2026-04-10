# Cohere 本地 PoC — 音频转写 + LLM 清洗 + 结构化总结

> **Language / 语言：** [English](README.md) | 中文

在苹果芯片（已在 Mac mini M4 16 GB 上测试）本地运行
[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)，
并将输出接入可选的 LLM 清洗，以及两个结构化后处理产物：
分析包。

本工作空间只回答一个具体问题：

> Mac 本地运行 Cohere 模型的效果是否足以支撑后续集成 `summary-only-fast` 管线？

---

## 管线概览

```
input/audio.wav
    │
    ▼
poc_cohere_local_transcribe.py   ← 本地 ASR（transformers + MPS）
    │  output/transcript.txt
    ▼
transcript_cleanup.py            ← 可选：LLM 去重 / 标点补全 / 自动换行
    │  output/transcript_cleaned.txt
    ▼
transcript_summary.py            ← LLM 分析包（可直接读 transcript.txt 或 transcript_cleaned.txt）
       output/transcript_cleaned_analysis.md
```

每个阶段都是独立脚本，可单独运行。

---

## 目录结构

```text
.
├── .env                   ← 本地密钥（已 git-ignore）
├── .env.example           ← 复制此文件，填入你的密钥
├── .venv/                 ← 由 setup_venv.sh 创建（已 git-ignore）
├── input/                 ← 放置音频文件（已 git-ignore）
├── logs/                  ← 运行日志（已 git-ignore）
├── models/                ← 下载的模型权重（已 git-ignore）
│   └── CohereLabs/
├── output/                ← 生成的转写与总结（已 git-ignore）
├── requirements.txt
├── scripts/
│   ├── autoFull.py
│   ├── autoFull.sh
│   ├── download_model.sh
│   ├── setup_venv.sh
│   ├── poc_cohere_local_transcribe.py
│   ├── transcript_cleanup.py
│   └── transcript_summary.py
├── README.md
└── README.zh.md
```

---

## 依赖要求

- **Python 3.11** — 本地 ASR 技术栈在 3.11 上兼容性最好。
- LLM API Key，用于清洗和总结步骤（见[本地环境配置](#本地环境配置)）。

---

## 初始化环境

```bash
./scripts/setup_venv.sh
```

脚本会自动检测 `python3.11`（或回退到 `python3`）。可以手动指定解释器：

```bash
PYTHON=/opt/homebrew/bin/python3.11 ./scripts/setup_venv.sh
```

---

## 下载模型

```bash
./scripts/download_model.sh
```

模型保存至 `./models/CohereLabs/cohere-transcribe-03-2026`，PoC 脚本会自动识别该路径。

国内镜像加速：

```bash
HF_ENDPOINT=https://hf-mirror.com ./scripts/download_model.sh
```

---

## 本地环境配置

复制模板并填入 API Key：

```bash
cp .env.example .env
```

`.env.example` 内容：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=deepseek-ai/DeepSeek-V3
```

`LLM_BASE_URL` 和 `LLM_MODEL` 为可选项，不填则使用上述默认值。

---

## 运行

### 完整管线（转写 → 可选清洗 → 总结）

一条命令跑完全部流程，优先推荐 Python CLI：

```bash
./scripts/autoFull.py
```

shell 包装器仍然保留，兼容旧入口：

```bash
./scripts/autoFull.sh
```

默认会跳过清洗步骤；如需先清洗再总结，可显式开启：

```bash
./scripts/autoFull.sh --enCleanUp 1
```

所有参数均为可选，不填则使用默认值：

```bash
./scripts/autoFull.sh \
  --input    ./input/extracted_audio.wav \
  --output   ./output \
  --language en \
  --model    ./models/CohereLabs/cohere-transcribe-03-2026 \
  --env      ./.env
```

`-h` / `--help` 查看完整用法说明。

### 分步运行

需要更细粒度控制时，可单独执行各阶段脚本。

#### 仅转写

```bash
./.venv/bin/python ./scripts/poc_cohere_local_transcribe.py \
  --input ./input/extracted_audio.wav \
  --output-dir ./output
```

#### 仅清洗

```bash
./.venv/bin/python ./scripts/transcript_cleanup.py \
  --input ./output/transcript.txt \
  --output ./output/transcript_cleaned.txt
```

### 仅总结

```bash
./.venv/bin/python ./scripts/transcript_summary.py \
  --input ./output/transcript_cleaned.txt
```

如需显式指定分析包路径，可传 `--output`：

```bash
./.venv/bin/python ./scripts/transcript_summary.py \
  --input ./output/transcript_cleaned.txt \
  --output ./output/transcript_cleaned_analysis.md
```

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `output/transcript.txt` | ASR 原始输出 |
| `output/transcript_cleaned.txt` | LLM 去重 + 标点补全后的转写 |
| `output/transcript_analysis.md` | 直接基于原始转写生成的分析包 |
| `output/transcript_cleaned_analysis.md` | 主张 / 证据 / 风险提示分析包 + X 串帖草稿 |
| `output/report.json` | 运行基准报告：模型路径、Python/torch 版本、设备、音频时长、加载时间、转写时间、RSS 内存 |
| `logs/poc_cohere_local_transcribe.log` | ASR 运行日志 |
| `logs/transcript_cleanup.log` | 清洗运行日志 |
| `logs/transcript_summary.log` | 总结运行日志 |

---

## 注意事项

- 转写输出出现多语言乱码时，最常见原因是 `transformers` 版本不在支持范围内，检查 `requirements.txt` 中的版本约束。
- LLM 清洗和总结步骤是独立模块，可以不依赖 ASR 步骤单独调用或导入。
- 总结阶段现在默认输出一个 Markdown：用于 `主张 / 证据 / 风险` 分析。
- 最终 Markdown 会做可读性后处理，尽量在 `. ? !` 与 `。？！` 后优先换行，避免大段文本堆叠。
