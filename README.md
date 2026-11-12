# GPT-2

A from-scratch implementation of GPT-2 in PyTorch: data preparation, training, and evaluation.

## Files

| File | Purpose |
|------|---------|
| `fineweb.py` | Downloads and tokenizes the FineWeb-Edu 10B dataset into shards |
| `train.py` | Model definition and training loop |
| `hellaswag.py` | HellaSwag benchmark evaluation |
| `play.py` | Sampling / experimenting with the model |

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone <your-repo-url>
cd Gpt2
uv sync
```

## Usage

1. Prepare the dataset:
   ```bash
   uv run fineweb.py
   ```
2. Train the model:
   ```bash
   uv run train.py
   ```
3. Evaluate on HellaSwag:
   ```bash
   uv run hellaswag.py
   ```
4. Generate text / experiment:
   ```bash
   uv run play.py
   ```

## Notes

Datasets (`edu_fineweb10B/`), logs, and model checkpoints are not tracked in git.