# GPT-2 (124M) Replication in PyTorch

A from-scratch replication of **GPT-2 (124M)**, written entirely in **Python** with PyTorch. The project covers the full pipeline: dataset preparation, model definition, training, benchmark evaluation, and text generation.

The architecture follows the GPT-2 design and is checked against the **original OpenAI GPT-2 weights** (the model can load them directly). Training on FineWeb-Edu **matches the results of the original GPT-2 (124M) baseline**, evaluated with validation loss and the **HellaSwag** benchmark.

---

## Highlights

- **GPT-2 (124M) built from scratch**: 12 layers, 12 attention heads, 768-dim embeddings, 1024-token context.
- **Loads the original OpenAI GPT-2 weights** to verify correctness and to serve as a baseline.
- **Trained on FineWeb-Edu (10B tokens)** for roughly one epoch (19,073 steps at ~0.5M tokens per step).
- **HellaSwag evaluation** run periodically during training, plus validation loss and generated samples.
- **Trained on a single consumer GPU (RTX 4060)** because of limited GPU availability, using CUDA, Flash Attention, and **bfloat16 mixed precision**. This trades a little numerical precision for much faster training and lower memory use.
- **Light fine-tuning** applied after pretraining.
- **Multi-GPU ready**: the training script also supports DDP via `torchrun`, though all runs here used one GPU.

---

## Results

The reproduction reaches the same level as the original GPT-2 (124M) baseline.

| Model                           | Parameters | Train loss | Val loss |
| ------------------------------- | ---------: | ---------: | -------: |
| OpenAI GPT-2 (original weights) |       124M |       3.11 |     3.12 |
| **This replication**            |       124M |       ~3.1 |     ~3.1 |

HellaSwag accuracy (length-normalized) is logged every 250 steps to `log/log.txt`.

| Model                | HellaSwag accuracy |
| -------------------- | -----------------: |
| OpenAI GPT-2 (124M)  |             0.2967 |
| **This replication** |             0.3324 |

---

## Project Structure

| File           | Purpose                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| `fineweb.py`   | Downloads and tokenizes the FineWeb-Edu 10B dataset into shards         |
| `train.py`     | GPT-2 model, data loader, optimizer, training loop, periodic evaluation |
| `hellaswag.py` | HellaSwag benchmark evaluation                                          |
| `play.py`      | Text generation and experimenting with the model                        |

---

## Tech Stack

- **Language:** Python
- **Framework:** PyTorch
- **Tokenizer:** `tiktoken` (GPT-2 BPE)
- **Dependency management:** [uv](https://github.com/astral-sh/uv)

---

## Setup

```bash
git clone https://github.com/Garv-00/MY_GPT2
cd Gpt2
uv sync
```

An NVIDIA GPU with CUDA is recommended. Development and training were done on an RTX 4060.

---

## Usage

1. **Prepare the dataset**

   ```bash
   uv run fineweb.py
   ```

   Downloads FineWeb-Edu 10B and tokenizes it into shards in `edu_fineweb10B/`.

2. **Train the model**

   ```bash
   uv run train.py
   ```

   On multiple GPUs:

   ```bash
   torchrun --standalone --nproc_per_node=n train.py (n = number of GPU's/Parallel Processses)
   ```

3. **Evaluate on HellaSwag**

   ```bash
   uv run hellaswag.py
   ```

4. **Generate text / experiment**
   ```bash
   uv run play.py
   ```

---

## Implementation Details

**Model**

- Decoder-only Transformer with pre-LayerNorm blocks and residual connections.
- Causal self-attention using PyTorch's fused `scaled_dot_product_attention` (Flash Attention).
- GELU activation (tanh approximation), as in the original GPT-2.
- Token embedding and output head share weights.
- GPT-2 style initialization: std 0.02, with residual projections scaled by `(2 * n_layer)^-0.5`.
- Vocabulary padded from 50,257 to 50,304 for more efficient GPU kernels.

**Training**

- Total batch size of 524,288 tokens (2^19), built from micro-batches of 16 x 1024 tokens with gradient accumulation (32 steps).
- AdamW (fused on CUDA) with betas (0.9, 0.95), weight decay 0.1 applied only to 2D weights, and gradient clipping at 1.0.
- Learning rate: linear warmup for 715 steps to 6e-4, then cosine decay to 6e-5 over 19,073 steps.
- bfloat16 autocast and TF32 matmuls for speed on a consumer GPU. An optional compile flag is available in `train.py`.

**Evaluation and monitoring**

- Validation loss, HellaSwag accuracy, and sample generations (top-k = 50) every 250 steps.
- Checkpoints saved to `log/` every 5,000 steps and at the final step.

---

## References and Resources

- Vaswani et al., **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (2017)
- Radford et al., **[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** (GPT-2, 2019)
- Brown et al., **[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)** (GPT-3, 2020)
- Zellers et al., **[HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)** (2019)
- **[FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)** (Hugging Face)
- OpenAI GPT-2 pretrained weights (used for validation and as a baseline)
>>>>>>> 21c6415 (adding  bfloat32 precision and DPP code)
