# UnixCoder LoRA Fine-tuning Approach

Binary classification for machine-generated code detection using UnixCoder with LoRA (Low-Rank Adaptation).

## Overview

This approach uses Microsoft's UniXcoder model with a custom binary classification head, fine-tuned using LoRA for parameter-efficient training. The model uses mean pooling over the last hidden states for classification.

## Files

- `unixcoder_lora.ipynb` - Complete training notebook (Google Colab compatible)
- `../../predictions/unixcoder_lora_predictions.csv` - Model predictions on test set

## Model Architecture

```
UniXcoder Base Encoder
    ↓
Mean Pooling (attention-masked)
    ↓
Dropout (0.1)
    ↓
Linear (hidden_size → 1)
    ↓
BCEWithLogitsLoss
```

## Model Details

- **Base Model**: `microsoft/unixcoder-base`
- **Method**: Full fine-tuning + LoRA adapters (in notebook)
- **Max Sequence Length**: 512 tokens
- **Special Token**: `<encoder-only>` prefix
- **Task**: Binary classification (human=0, machine=1)

## Installation

```bash
pip install -r requirements.txt
```

## Training (Google Colab)

1. Open `unixcoder_lora.ipynb` in Google Colab
2. Select a GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Update the data paths to point to your dataset location
4. Run all cells in order

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-5 |
| Batch Size | 8 |
| Epochs | 3 |
| Gradient Accumulation | 2 |
| Warmup Ratio | 0.06 |
| Weight Decay | 0.01 |
| FP16 | Enabled |

## Performance Metrics

The model is evaluated using:
- **Macro F1 Score** (primary metric)
- Accuracy
- Precision & Recall

## Notes

- The notebook includes both full fine-tuning and PEFT/LoRA sections
- LoRA significantly reduces trainable parameters while maintaining performance
- Uses BCEWithLogitsLoss for stable training

