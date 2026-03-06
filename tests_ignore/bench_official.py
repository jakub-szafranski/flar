#!/usr/bin/env python3
"""
Benchmark the *officially compressed* FLAP model (same pipeline as main.py).

This script replicates what main.py does *in-process* — load the dense model,
call prune_flap() to permanently reshape its weights, then benchmark generation
speed.  Doing it in-process avoids the save_pretrained → from_pretrained size-
mismatch that arises because save_pretrained writes pruned weight shapes while
config.json still says intermediate_size=11008.

Usage
-----
  # Compress & benchmark (same settings as main.py defaults):
  python bench_official.py --model huggyllama/llama-7b --pruning_ratio 0.2

  # Change token counts / runs:
  python bench_official.py --model huggyllama/llama-7b --tokens 50 250 500 --runs 5
"""

import argparse
import time

import numpy as np
import torch
from transformers import AutoTokenizer

# ── repo-local helpers ───────────────────────────────────────────
from mom.prunable_llm import load_llm
from lib.prune import prune_flap


# ─────────────────────────────────────────────────────────────────
PROMPTS = [
    "The best way to learn programming is",
    "Artificial intelligence will change the world by",
]


def section(title: str):
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")


def mem_str(device) -> str:
    alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
    res   = torch.cuda.memory_reserved(device)  / 1024 ** 3
    return f"  allocated: {alloc:.2f} GiB  |  reserved: {res:.2f} GiB"


def bench_model(model, tokenizer, device, prompts, n_tok_list, n_runs, tag):
    """Run generation benchmark for *model* at each token count."""
    results = {}
    for n_tok in n_tok_list:
        torch.cuda.empty_cache()
        times = []
        for _ in range(n_runs):
            t = 0.0
            for prompt in prompts:
                inp = tokenizer(prompt, return_tensors="pt").to(device)
                # warmup
                with torch.no_grad():
                    model.generate(**inp, max_new_tokens=10, do_sample=False,
                                   eos_token_id=[])
                torch.cuda.synchronize(device)
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                t0.record()
                with torch.no_grad():
                    model.generate(**inp, max_new_tokens=n_tok, do_sample=False,
                                   eos_token_id=[])   # never stop early
                t1.record()
                torch.cuda.synchronize(device)
                t += t0.elapsed_time(t1)
            times.append(t)
        arr = np.array(times)
        ms_per_run  = float(arr.mean())    # total ms for all prompts
        ms_per_tok  = ms_per_run / (n_tok * len(prompts))
        tok_per_sec = 1000.0 / ms_per_tok
        results[n_tok] = dict(ms=ms_per_run, std=float(arr.std()),
                              ms_per_tok=ms_per_tok, tok_per_sec=tok_per_sec)
        print(f"  [{tag}] {n_tok:>4} tokens │ "
              f"{ms_per_run:>8.1f} ± {float(arr.std()):>6.1f} ms │ "
              f"{ms_per_tok:>6.2f} ms/tok │ "
              f"{tok_per_sec:>6.1f} tok/s")
    return results


# ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         default="huggyllama/llama-7b")
    ap.add_argument("--pruning_ratio", type=float, default=0.2)
    ap.add_argument("--remove_heads",  type=int,   default=-1)
    ap.add_argument("--metrics",       default="WIFV",
                    choices=["IFV", "WIFV", "WIFN"])
    ap.add_argument("--structure",     default="AL-AM",
                    choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM"])
    ap.add_argument("--nsamples",      type=int, default=128)
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--device",        default="cuda:0")
    ap.add_argument("--cache_dir",     default="llm_weights")
    ap.add_argument("--tokens",        nargs="+", type=int, default=[50, 250, 500])
    ap.add_argument("--runs",          type=int, default=5)
    ap.add_argument("--unstr",         action="store_true",
                    help="Use unstructured pruning (structured by default)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── 1. Load dense model ────────────────────────────────────────
    section("STEP 1 – LOAD DENSE MODEL")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading: {args.model}")
    model = load_llm(args.model, cache_dir=args.cache_dir).to(device).eval()
    dense_params = sum(p.numel() for p in model.parameters())
    print(f"  Dense params : {dense_params/1e9:.2f} B")
    print(f"  GPU memory   :\n{mem_str(device)}")

    # ── 2. Benchmark dense ─────────────────────────────────────────
    section("STEP 2 – BENCHMARK DENSE")
    dense_res = bench_model(model, tokenizer, device,
                            PROMPTS, args.tokens, args.runs, "dense")

    # ── 3. Compress (same as main.py) ──────────────────────────────
    section("STEP 3 – COMPRESS (prune_flap in-process)")
    print(f"  pruning_ratio={args.pruning_ratio}  metrics={args.metrics}  "
          f"structure={args.structure}  nsamples={args.nsamples}")

    t0 = time.time()
    prune_flap(args, model, tokenizer, device)
    elapsed = time.time() - t0

    comp_params = sum(p.numel() for p in model.parameters())
    param_red   = (dense_params - comp_params) / dense_params * 100
    print(f"\n  Compression finished in {elapsed:.1f}s")
    print(f"  Compressed params : {comp_params/1e9:.2f} B  "
          f"({param_red:.1f}% reduction)")
    print(f"  GPU memory        :\n{mem_str(device)}")

    # ── 4. Per-layer shape printout ────────────────────────────────
    section("STEP 4 – COMPRESSED LAYER SHAPES")
    print(f"  {'Layer':>5}  {'Heads':>8}  {'Attn dim':>10}  {'MLP dim':>10}")
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.num_heads
        a = layer.self_attn.q_proj.weight.shape[0]
        m = layer.mlp.up_proj.weight.shape[0]
        print(f"  {i:>5}  {h:>8}  {a:>10}  {m:>10}")

    # ── 5. Benchmark compressed ────────────────────────────────────
    section("STEP 5 – BENCHMARK COMPRESSED")
    torch.cuda.empty_cache()
    comp_res = bench_model(model, tokenizer, device,
                           PROMPTS, args.tokens, args.runs, "compressed")

    # ── 6. Summary table ───────────────────────────────────────────
    section("STEP 6 – SUMMARY")
    col = [6, 14, 14, 14, 10]
    hdr = ["Tokens", "Dense ms/tok", "Comp ms/tok", "tok/s (comp)", "Speedup%"]
    sep = "─┼─".join("─" * c for c in col)
    fmt = " │ ".join(f"{{:^{c}}}" for c in col)
    print("  " + fmt.format(*hdr))
    print("  " + sep)
    for n_tok in args.tokens:
        d = dense_res[n_tok]
        c = comp_res[n_tok]
        spd = (d["ms_per_tok"] - c["ms_per_tok"]) / d["ms_per_tok"] * 100
        print("  " + fmt.format(
            str(n_tok),
            f"{d['ms_per_tok']:.2f}",
            f"{c['ms_per_tok']:.2f}",
            f"{c['tok_per_sec']:.1f}",
            f"{spd:+.1f}%",
        ))
    print(f"\n  Model params: {dense_params/1e9:.2f} B → {comp_params/1e9:.2f} B "
          f"({param_red:.1f}% reduction)")


if __name__ == "__main__":
    main()
