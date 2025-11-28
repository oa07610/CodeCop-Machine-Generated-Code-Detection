# Qwen3 Zero-Shot Approach

Machine-generated code detection using Qwen3 Coder (30B) with few-shot prompting via Ollama.

## Overview

This approach uses a large language model in a zero/few-shot setting without fine-tuning. It leverages:
- Token-based similarity (Jaccard) for selecting relevant few-shot examples
- Detailed prompts with coding style indicators
- Local inference via Ollama

## Files

- `zeroshot_qwen3.py` - Complete inference script with few-shot prompting
- `../../predictions/qwen_predictions.csv` - Model predictions on test set

## Model Details

- **Model**: `qwen3-coder:30b` (via Ollama)
- **Method**: Few-shot prompting (5 examples)
- **Example Selection**: Jaccard similarity on code tokens
- **Task**: Binary classification (human=0, machine=1)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 3. Pull the Model

```bash
ollama pull qwen3-coder:30b
```

## Usage

```bash
# Start Ollama server (if not running)
ollama serve &

# Run inference
python zeroshot_qwen3.py
```

The script will:
1. Load training and test data from `../../data/`
2. For each test sample, select similar examples from training set
3. Generate predictions with checkpointing (every 10 samples)
4. Save final predictions to `final_predictions.csv`

## Configuration

Edit the constants at the top of `zeroshot_qwen3.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `qwen3-coder:30b` | Ollama model name |
| `FEW_SHOT_COUNT` | 5 | Number of examples in prompt |
| `BATCH_SIZE` | 1 | Processing batch size |
| `CHECKPOINT_FILE` | `predictions_checkpoint.csv` | Checkpoint file path |

## Prompt Strategy

The prompt includes key indicators for analysis:

1. **Commenting style**: AI uses verbose/template comments
2. **Variable naming**: AI uses generic names (result, temp, helper)
3. **Code structure**: AI follows standard patterns, humans have quirks
4. **Error handling**: AI includes comprehensive handling
5. **Whitespace**: AI has perfectly consistent spacing
6. **Algorithm efficiency**: AI chooses textbook solutions
7. **Edge cases**: AI handles edge cases explicitly
8. **Documentation**: AI over-documents with formal docstrings

## Few-Shot Example Selection

Examples are selected based on Jaccard similarity of code tokens:
1. Tokenize code (strip strings, comments)
2. Compute Jaccard similarity with all training samples
3. Select top similar samples, balanced by label (human/AI)
4. Shuffle and include in prompt

## Checkpointing

The script automatically saves progress every 10 samples. If interrupted:
- Resume by running the same command
- Already-processed samples are skipped
- Progress is loaded from `predictions_checkpoint.csv`

## Hardware Requirements

- **RAM**: 32GB+ recommended
- **GPU**: Optional but recommended (16GB+ VRAM)
- **Disk**: ~60GB for model weights

