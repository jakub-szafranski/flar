#!/usr/bin/env python3
"""
Correctness & timing test for PrunableLLM.

Tests both structured (unstr=False) and unstructured (unstr=True) modes.

For each mode:
  1. Report GPU memory after model load.
  2. Generate 30 tokens for two prompts (dense baseline).
  3. Prune → generate (pruned).
  4. Unprune → generate (restored dense).
  5. Assert restored outputs are identical to baseline.
  6. Time prune() and unprune() independently over 10 iterations.

Usage
-----
    python run_test.py
    python run_test.py --model huggyllama/llama-7b --expert experts/c4_p0.2_WIFV_ALAM_llama7b.pt
"""

import argparse
import time
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from mom.prunable_llm import PrunableLLM


# ─────────────────────────────────────────────────────────────────
PROMPTS = ["Hi", "This is a"]
MAX_NEW_TOKENS = 30
TIMING_ITERS = 10


# ─────────────────────────────────────────────────────────────────
def mem_summary(device: torch.device) -> str:
    alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    return f"  allocated: {alloc:.2f} GiB  |  reserved: {reserved:.2f} GiB"


def generate(wrapper: PrunableLLM, tokenizer, prompts, device: torch.device):
    """Return list of decoded strings (prompt + 30 new tokens)."""
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = wrapper.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        results.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return results


def time_op_cuda(fn, device: torch.device, n: int):
    """
    Time fn() for n iterations using CUDA events (wall time on GPU).
    Returns (mean_ms, std_ms).
    """
    timings = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record(torch.cuda.current_stream(device))
        fn()
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        timings.append(start.elapsed_time(end))   # milliseconds
    arr = np.array(timings)
    return float(arr.mean()), float(arr.std())


def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def run_mode(wrapper: PrunableLLM, tokenizer, expert: dict, device: torch.device,
             unstr: bool):
    mode = "UNSTRUCTURED (unstr=True)" if unstr else "STRUCTURED  (unstr=False)"
    section(f"MODE: {mode}")

    # ── 1. baseline dense generation ─────────────────────────────
    print("\n[1] Dense baseline generation:")
    dense_outputs = generate(wrapper, tokenizer, PROMPTS, device)
    for prompt, out in zip(PROMPTS, dense_outputs):
        print(f'  prompt: "{prompt}"')
        print(f'  output: "{out}"')
        print()

    # ── 2. prune → generate ──────────────────────────────────────
    print("[2] Applying expert (prune)...")
    wrapper.prune(expert, unstr=unstr)
    print(f"  Active expert: {wrapper.active_expert_info}")
    print(f"  GPU memory after prune:\n{mem_summary(device)}")

    print("\n  Pruned generation:")
    pruned_outputs = generate(wrapper, tokenizer, PROMPTS, device)
    for prompt, out in zip(PROMPTS, pruned_outputs):
        print(f'  prompt: "{prompt}"')
        print(f'  output: "{out}"')
        print()

    # ── 3. unprune → generate → correctness check ─────────────────
    print("[3] Restoring dense (unprune)...")
    wrapper.unprune()
    print(f"  GPU memory after unprune:\n{mem_summary(device)}")

    print("\n  Restored-dense generation:")
    restored_outputs = generate(wrapper, tokenizer, PROMPTS, device)
    for prompt, out in zip(PROMPTS, restored_outputs):
        print(f'  prompt: "{prompt}"')
        print(f'  output: "{out}"')
        print()

    # ── 4. correctness assertion ──────────────────────────────────
    print("[4] Correctness check (baseline == restored):")
    all_ok = True
    for i, (a, b, prompt) in enumerate(zip(dense_outputs, restored_outputs, PROMPTS)):
        ok = (a == b)
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f'  [{status}] prompt: "{prompt}"')
        if not ok:
            all_ok = False
            print(f"         baseline : {a}")
            print(f"         restored : {b}")
    if all_ok:
        print("  → All outputs match. Prune/unprune is lossless.")
    else:
        print("  → MISMATCH DETECTED. Check implementation!", file=sys.stderr)

    # ── 5. timing ────────────────────────────────────────────────
    # Ensure model is dense before timing loop
    if wrapper.is_pruned:
        wrapper.unprune()

    print(f"\n[5] Timing prune() and unprune() over {TIMING_ITERS} iterations...")

    def do_prune():
        wrapper.prune(expert, unstr=unstr)

    def do_unprune():
        wrapper.unprune()

    prune_times, unprune_times = [], []
    for _ in range(TIMING_ITERS):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        t0.record()
        wrapper.prune(expert, unstr=unstr)
        t1.record()
        wrapper.unprune()
        t2.record()
        torch.cuda.synchronize(device)
        prune_times.append(t0.elapsed_time(t1))
        unprune_times.append(t1.elapsed_time(t2))

    pt = np.array(prune_times)
    ut = np.array(unprune_times)
    print(f"  prune()   : {pt.mean():.1f} ± {pt.std():.1f} ms  "
          f"(min {pt.min():.1f}, max {pt.max():.1f})")
    print(f"  unprune() : {ut.mean():.1f} ± {ut.std():.1f} ms  "
          f"(min {ut.min():.1f}, max {ut.max():.1f})")
    print(f"  total switch (prune+unprune) : "
          f"{(pt + ut).mean():.1f} ± {(pt + ut).std():.1f} ms")

    return all_ok


# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PrunableLLM correctness & timing test")
    parser.add_argument(
        "--model", type=str, default="huggyllama/llama-7b",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--expert", type=str, default="experts/c4_p0.2_WIFV_ALAM_llama7b.pt",
        help="Path to .pt expert artifact",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Target device (default: cuda:0)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="llm_weights",
        help="Model cache directory",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── load model ────────────────────────────────────────────────
    section("LOADING MODEL")
    print(f"  model   : {args.model}")
    print(f"  device  : {device}")
    print(f"  expert  : {args.expert}")

    wrapper = PrunableLLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device=device
    )
    wrapper.model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # ── memory after load ─────────────────────────────────────────
    print(f"\n  GPU memory after model load:\n{mem_summary(device)}")
    total_params = sum(p.numel() for p in wrapper.model.parameters())
    print(f"  Parameters : {total_params / 1e9:.2f} B")

    # ── load expert ───────────────────────────────────────────────
    section("LOADING EXPERT")
    expert = torch.load(args.expert, map_location="cpu")
    print(f"  dataset        : {expert.get('calibration_dataset')}")
    print(f"  pruning_ratio  : {expert.get('pruning_ratio')}")
    print(f"  structure      : {expert.get('structure')}")
    print(f"  metrics        : {expert.get('metrics')}")
    print(f"  num_layers     : {expert.get('num_layers')}")

    # ── run both modes ────────────────────────────────────────────
    results = {}
    for unstr in (False, True):
        ok = run_mode(wrapper, tokenizer, expert, device, unstr=unstr)
        results["structured" if not unstr else "unstructured"] = ok

    # ── final summary ─────────────────────────────────────────────
    section("SUMMARY")
    all_passed = True
    for mode, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{status}] {mode}")
        if not ok:
            all_passed = False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
