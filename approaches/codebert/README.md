# CodeBERT Fine-tuning Approach

Binary classification for machine-generated code detection using CodeBERT with full fine-tuning.

## Overview

This approach uses Microsoft's CodeBERT model fine-tuned for binary classification (human=0, AI=1) with support for:
- Class-weighted loss to handle imbalanced datasets
- Label smoothing for regularization
- Weights & Biases integration for experiment tracking

## Files

- `codebert_finetuning.py` - Training script with weighted loss and W&B logging
- `codebert_preds.py` - Inference script to generate Kaggle submission CSV

## Model Details

- **Base Model**: `microsoft/codebert-base`
- **Method**: Full fine-tuning with weighted cross-entropy loss
- **Max Sequence Length**: 512 tokens
- **Task**: Binary classification (human=0, machine=1)

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python codebert_finetuning.py \
    --model-name microsoft/codebert-base \
    --train-path ../../data/train.parquet \
    --dev-path ../../data/validation.parquet \
    --output-dir ./output \
    --batch-size 8 \
    --epochs 3 \
    --lr 1.5e-5
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `microsoft/codebert-base` | Base model to fine-tune |
| `--batch-size` | 8 | Training batch size |
| `--accum` | 2 | Gradient accumulation steps |
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 1.5e-5 | Learning rate |
| `--max-length` | 512 | Max token sequence length |
| `--label-smoothing` | 0.05 | Label smoothing factor |
| `--warmup` | 0.1 | Warmup ratio |

### Weights & Biases Integration

Enable experiment tracking with:
```bash
python codebert_finetuning.py \
    ... \
    --wandb-project your-project \
    --wandb-mode online
```

## Inference

```bash
python codebert_preds.py \
    --model-path ./output/best \
    --parquet-path ../../data/test.parquet \
    --output-csv ../../predictions/codebert_predictions.csv \
    --id-source ../../data/sample_submission.csv
```

## Performance

The model achieves competitive Macro F1 scores on the validation set. See W&B logs for detailed metrics and training curves.

