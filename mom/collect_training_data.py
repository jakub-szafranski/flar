#!/usr/bin/env python3
"""
Collect training data for the RL router by running the dense model on
CommonsenseQA and MMLU, keeping only correctly-answered samples.

Standardised output format (list of dicts saved as .pt):
    {"question": str, "answers": [str, ...], "correct": int}   # 0-indexed

Usage
-----
    python -m mom.collect_training_data \
        --model huggyllama/llama-7b \
        --batch_size 32 \
        --output data/router_train.pt
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from models.hf_llama.modeling_llama import LlamaForCausalLM


# ── model loading (same pattern as everywhere else) ──────────────
def get_llm(model_name: str, device: str = "cuda:0", cache_dir: str = "llm_weights"):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    return model


LABELS = ["A", "B", "C", "D", "E"]  # max 5; MMLU uses first 4, CSQA uses all 5


def format_prompt(question: str, choices: list[str]) -> str:
    """Build a standard MCQ prompt ending with 'Answer:'."""
    opts = "\n".join(f"{LABELS[i]}. {c}" for i, c in enumerate(choices))
    return f"Question: {question}\n{opts}\nAnswer:"


# ── normalise datasets into a unified iterator ───────────────────

def iter_commonsense_qa():
    """Yield (question, choices[5], correct_idx) from CommonsenseQA train."""
    ds = load_dataset("tau/commonsense_qa", split="train")
    label_map = {l: i for i, l in enumerate(LABELS)}
    for row in ds:
        choices = row["choices"]["text"]
        key = row["answerKey"]
        if key not in label_map or len(choices) != 5:
            continue
        yield row["question"], choices, label_map[key]


def iter_mmlu():
    """Yield (question, choices[4], correct_idx) from MMLU auxiliary_train."""
    raw = load_dataset("cais/mmlu", "auxiliary_train")
    ds = raw["train"] if hasattr(raw, "keys") else raw

    for row in ds:
        # this dataset wraps each record under a nested "train" key
        r = row["train"] if "train" in row else row
        choices = r["choices"]
        if len(choices) != 4:
            continue
        yield r["question"], choices, int(r["answer"])


# ── batched evaluation ───────────────────────────────────────────

@torch.no_grad()
def evaluate_batch(
    model,
    tokenizer,
    prompts: list[str],
    answer_token_ids: list[int],
    device: str,
) -> list[int]:
    """Return predicted answer index (0-3) for each prompt in the batch."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=1024)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # For each sample grab logits at the last real token position
    seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
    last_logits = logits[torch.arange(len(prompts), device=device), seq_lens]  # (B, V)

    answer_logits = last_logits[:, answer_token_ids]  # (B, N) where N=4 or 5
    preds = answer_logits.argmax(dim=1).tolist()
    return preds


# ── main collection loop ────────────────────────────────────────

def collect(
    model,
    tokenizer,
    data_iter,
    target: int,
    batch_size: int,
    device: str,
    answer_token_ids: list[int],
    tag: str,
    out_path: str,
    existing: list[dict],
    src: str,
    save_every: int = 500,
) -> list[dict]:
    """
    Iterate over *data_iter*, run batched eval, keep correct samples.
    Returns accumulated list, saves checkpoint every *save_every* new hits.
    """
    collected = list(existing)
    start_count = len(collected)
    batch_q, batch_c, batch_idx, batch_prompts = [], [], [], []

    pbar = tqdm(total=target, initial=0, desc=tag)

    for question, choices, correct in data_iter:
        if len(collected) - start_count >= target:
            break

        batch_q.append(question)
        batch_c.append(choices)
        batch_idx.append(correct)
        batch_prompts.append(format_prompt(question, choices))

        if len(batch_prompts) < batch_size:
            continue

        # run batch
        preds = evaluate_batch(model, tokenizer, batch_prompts,
                               answer_token_ids, device)
        for q, ch, gt, pred in zip(batch_q, batch_c, batch_idx, preds):
            if pred == gt:
                collected.append({"question": q, "answers": ch, "correct": gt, "_src": src})

        # periodic save
        new_hits = len(collected) - start_count
        pbar.n = new_hits
        pbar.refresh()
        if new_hits > 0 and new_hits % save_every < batch_size:
            torch.save(collected, out_path)
            tqdm.write(f"[{tag}] checkpoint – {len(collected)} total saved")

        batch_q, batch_c, batch_idx, batch_prompts = [], [], [], []

        if new_hits >= target:
            break

    # flush remaining batch
    if batch_prompts and (len(collected) - start_count) < target:
        preds = evaluate_batch(model, tokenizer, batch_prompts,
                               answer_token_ids, device)
        for q, ch, gt, pred in zip(batch_q, batch_c, batch_idx, preds):
            if pred == gt and (len(collected) - start_count) < target:
                collected.append({"question": q, "answers": ch, "correct": gt, "_src": src})

    pbar.n = len(collected) - start_count
    pbar.close()
    torch.save(collected, out_path)
    tqdm.write(f"[{tag}] done – {len(collected)} total saved")
    return collected


# ── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Collect router training data")
    ap.add_argument("--model", default="huggyllama/llama-7b")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output", default="data/router_train.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csqa_target", type=int, default=2000)
    ap.add_argument("--mmlu_target", type=int, default=3000)
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # resume if checkpoint exists
    if os.path.exists(args.output):
        collected = torch.load(args.output, weights_only=False)
        print(f"Resuming from checkpoint: {len(collected)} samples")
    else:
        collected = []

    # load model + tokenizer
    print(f"Loading {args.model} …")
    model = get_llm(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only: pad on left

    # resolve answer token ids per dataset (“ A”, “ B”, ... with leading space)
    all_token_ids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1]
                     for l in LABELS]
    csqa_token_ids = all_token_ids          # A-E (5 choices)
    mmlu_token_ids = all_token_ids[:4]      # A-D (4 choices)
    print(f"Answer token IDs: {dict(zip(LABELS, all_token_ids))}")

    # ── CommonsenseQA ──
    csqa_done = len([s for s in collected if s.get("_src") == "csqa"])
    if csqa_done < args.csqa_target:
        collected = collect(
            model, tokenizer, iter_commonsense_qa(),
            target=args.csqa_target - csqa_done,
            batch_size=args.batch_size, device=args.device,
            answer_token_ids=csqa_token_ids,
            tag="CSQA", out_path=args.output,
            existing=collected, src="csqa", save_every=args.save_every,
        )

    # ── MMLU ──
    mmlu_done = len([s for s in collected if s.get("_src") == "mmlu"])
    if mmlu_done < args.mmlu_target:
        collected = collect(
            model, tokenizer, iter_mmlu(),
            target=args.mmlu_target - mmlu_done,
            batch_size=args.batch_size, device=args.device,
            answer_token_ids=mmlu_token_ids,
            tag="MMLU", out_path=args.output,
            existing=collected, src="mmlu", save_every=args.save_every,
        )

    # strip internal bookkeeping key before final save
    for s in collected:
        s.pop("_src", None)
    torch.save(collected, args.output)
    print(f"\nDone. {len(collected)} total samples → {args.output}")


if __name__ == "__main__":
    main()
