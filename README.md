# SemEval-2026 Task 13: Machine-Generated Code Detection

A comprehensive collection of approaches for detecting machine-generated code, developed for the SemEval-2026 Task 13 competition.

## Task Description

Given a code snippet, classify whether it was:
- **Human-written (0)**: Code authored by a human developer
- **Machine-generated (1)**: Code produced by an AI/LLM

## Project Structure

```
DL_Project/
├── approaches/
│   ├── codebert/           # CodeBERT full fine-tuning
│   ├── unixcoder_lora/     # UnixCoder with LoRA adapters
│   ├── gpt_oss_qlora/      # GPT-OSS 20B with QLoRA
│   └── qwen3_zeroshot/     # Qwen3 Coder few-shot prompting
├── data/
│   ├── train.parquet       # Training dataset
│   ├── validation.parquet  # Validation dataset
│   ├── test.parquet        # Test dataset
│   ├── final_test.parquet  # Final evaluation test set
│   └── sample_submission.csv
├── predictions/
│   ├── unixcoder_lora_predictions.csv
│   ├── gpt_oss_predictions.csv
│   └── qwen_predictions.csv
└── README.md
```

## Approaches

| Approach | Model | Method | Training Time | Hardware |
|----------|-------|--------|---------------|----------|
| [CodeBERT](approaches/codebert/) | microsoft/codebert-base | Full fine-tuning | ~2-3 hours | GPU (8GB+) |
| [UnixCoder LoRA](approaches/unixcoder_lora/) | microsoft/unixcoder-base | LoRA + Binary head | ~1-2 hours | GPU (8GB+) |
| [GPT-OSS QLoRA](approaches/gpt_oss_qlora/) | unsloth/gpt-oss-20b | 4-bit QLoRA | ~4-6 hours | GPU (15GB+) |
| [Qwen3 Zero-Shot](approaches/qwen3_zeroshot/) | qwen3-coder:30b | Few-shot prompting | N/A | CPU/GPU (32GB RAM) |

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/oa07610/CodeCop-Machine-Generated-Code-Detection.git
cd CodeCop-Machine-Generated-Code-Detection
```

### 2. Choose an Approach

Each approach has its own directory with:
- Detailed README with usage instructions
- Specific `requirements.txt`
- Training/inference scripts or notebooks

### 3. Install Dependencies

```bash
cd approaches/codebert  # or any other approach
pip install -r requirements.txt
```

## Dataset

The dataset is from the [SemEval-2026 Task 13](https://semeval.github.io/) shared task and is available on HuggingFace.

### Download Data

The parquet files are too large for Git. Download them using one of these methods:

**Option 1: HuggingFace Datasets (Recommended)**
```python
from datasets import load_dataset
ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")
```

**Option 2: Direct Download**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Download to data/ folder
huggingface-cli download DaniilOr/SemEval-2026-Task13 --local-dir ./data
```

After downloading, your `data/` folder should contain:
- `train.parquet` (~200MB)
- `validation.parquet` (~40MB)  
- `test.parquet`
- `final_test.parquet`
- `sample_submission.csv`

### Data Format

| Column | Description |
|--------|-------------|
| `code` | Source code string |
| `label` | 0 (human) or 1 (machine) |
| `language` | Programming language |
| `generator` | Source of the code |

## Citation

If you use this code, please cite:

```bibtex
@misc{semeval2026task13,
  title={SemEval-2026 Task 13: Machine-Generated Code Detection},
  year={2026},
  url={https://semeval.github.io/}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Microsoft Research for CodeBERT and UniXcoder
- Unsloth team for efficient fine-tuning tools
- SemEval organizers for the shared task
