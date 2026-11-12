
"""
Downloads and evaluates HellaSwag in Python.

Dataset source:
https://huggingface.co/datasets/Rowan/hellaswag

Original evaluation implementation based on:
https://github.com/rowanz/hellaswag
"""


import os
import json
import tiktoken
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """Downloads HellaSwag from Hugging Face and saves it as JSONL"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")

    if os.path.exists(data_filename):
        try:
            with open(data_filename, "r", encoding="utf-8") as f:
                json.loads(f.readline())
            print(f"Using cached HellaSwag {split} dataset.")
            return
        except (json.JSONDecodeError, OSError):
            print("Cached dataset is corrupted. Re-downloading...")
            os.remove(data_filename)

    hf_split = {"train": "train", "val": "validation", "test": "test"}[split]
    print(f"Downloading HellaSwag {hf_split} from Hugging Face...")

    dataset = load_dataset("Rowan/hellaswag", split=hf_split)

    with open(data_filename, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="Saving JSONL"):
            ctx = example.get("ctx")
            if ctx is None:
                ctx = (example.get("ctx_a", "") + " " + example.get("ctx_b", "")).strip()

            normalized_example = {"ind": example.get("ind"), "ctx": ctx, "label": int(example["label"]), "endings": example["endings"], "activity_label": example.get("activity_label", ""), "split": split, "source_id": example.get("source_id", "")}
            f.write(json.dumps(normalized_example, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} examples to {data_filename}")

def validate_dataset(split):
    """Validates the downloaded JSONL dataset"""
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    count = 0

    with open(data_filename, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                example = json.loads(line)
                assert "ctx" in example and "endings" in example and "label" in example
                assert len(example["endings"]) == 4 and example["label"] in range(4)
                count += 1
            except (json.JSONDecodeError, AssertionError, KeyError) as e:
                raise ValueError(f"Invalid HellaSwag data at line {line_number}: {e}")

    print(f"Dataset validation successful: {count} examples.")

    if split == "val" and count != 10042:
        raise ValueError(f"Expected 10042 validation examples, found {count}.")

def render_example(example):
    """
    Given an example, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN)
    - mask (1 in the region of the candidate completion)
    - label (the index of the correct completion)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {"label": label, "ctx_tokens": None, "ending_tokens": []}

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens

    tok_rows = []
    mask_rows = []

    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    validate_dataset(split)

    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

@torch.no_grad()
def evaluate(model_type, device):
    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    model.eval()
    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    for example in tqdm(iterate_examples("val"), total=10042, desc="Evaluating HellaSwag"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits

        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # now get the average loss just for the completion region (where mask == 1)
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        # debug: pretty print a few examples
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    print("\n" + "=" * 50)
    print("HELLASWAG EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total examples: {num_total}")
    print(f"Accuracy: {num_correct}/{num_total}={num_correct / num_total:.4f}")
    print(f"Normalized accuracy: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
