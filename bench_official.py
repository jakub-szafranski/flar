#!/usr/bin/env python3
"""
Benchmark the *officially compressed* FLAP model produced by main.py.

This script:
  1. Optionally runs main.py to produce a save_pretrained() checkpoint.
  2. Loads the checkpoint back from disk.
  3. Measures generation latency (tokens/s) with eos_token_id=[] so every
     run generates *exactly* max_new_tokens — an apples-to-apples comparison
     against the dense baseline and our PrunableLLM structured mode.

Why a separate script?
  main.py calls compress() which irreversibly mutates the nn.Linear shapes
  (same as our structured prune, but permanent).  Loading back from
  save_pretrained() gives us the "ground truth" speed for the compressed
  model without any PrunableLLM overhead.

Usage
-----
  # Run main.py first, then benchmark the saved model:
  python bench_official.py

  # Skip re-running main.py (model already saved):
  python bench_official.py --skip_prune

  # Custom paths / token counts:
  python bench_official.py --model huggyllama/llama-7b \\
      --save_dir llm_weights/flap_p0.2_WIFV_ALAM_llama_7b \\
      --pruning_ratio 0.2 --tokens 50 250 500
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

# ── repo-local patched model class ──────────────────────────────
from models.hf_llama.modeling_llama import LlamaForCausalLM


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


def cuda_ms(fn, device, n: int = 5) -> tuple[float, float]:
    """Return (mean_ms, std_ms) over *n* timed GPU runs (with warmup)."""
    for _ in range(2):          # warmup
        fn()
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(n):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        t0.record(); fn(); t1.record()
        torch.cuda.synchronize(device)
        ts.append(t0.elapsed_time(t1))
    a = np.array(ts)
    return float(a.mean()), float(a.std())


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
                def _gen(inp_=inp):
                    with torch.no_grad():
                        model.generate(
                            **inp_,
                            max_new_tokens=n_tok,
                            do_sample=False,
                            eos_token_id=[],   # never stop early
                        )
                t += cuda_ms(_gen, device, n=1)[0]
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
    ap.add_argument("--save_dir",      default="llm_weights/flap_p0.2_WIFV_ALAM_llama_7b")
    ap.add_argument("--pruning_ratio", type=float, default=0.2)
    ap.add_argument("--metrics",       default="WIFV")
    ap.add_argument("--structure",     default="AL-AM")
    ap.add_argument("--nsamples",      type=int, default=128)
    ap.add_argument("--device",        default="cuda:0")
    ap.add_argument("--cache_dir",     default="llm_weights")
    ap.add_argument("--tokens",        nargs="+", type=int, default=[50, 250, 500])
    ap.add_argument("--runs",          type=int, default=5)
    ap.add_argument("--skip_prune",    action="store_true",
                    help="Skip running main.py (use if model already saved)")
    args = ap.parse_args()
    device = torch.device(args.device)

    # ── 1. Run main.py to produce the compressed checkpoint ────────
    section("STEP 1 – COMPRESS WITH main.py")
    if args.skip_prune:
        print(f"  --skip_prune set.  Expecting saved model at: {args.save_dir}")
    else:
        cmd = [
            sys.executable, "main.py",
            "--model",          args.model,
            "--prune_method",   "flap",
            "--pruning_ratio",  str(args.pruning_ratio),
            "--remove_heads",   "-1",
            "--metrics",        args.metrics,
            "--structure",      args.structure,
            "--nsamples",       str(args.nsamples),
            "--save_model",     args.save_dir,
        ]
        print("  Running: " + " ".join(cmd))
        t0 = time.time()
        ret = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"\n  main.py finished in {elapsed:.1f}s")

    if not os.path.isdir(args.save_dir):
        print(f"ERROR: {args.save_dir} not found.  Run without --skip_prune first.",
              file=sys.stderr)
        sys.exit(1)

    # ── 2. Load dense baseline ─────────────────────────────────────
    section("STEP 2 – LOAD MODELS")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading dense model: {args.model}")
    dense = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        cache_dir=args.cache_dir, low_cpu_mem_usage=True,
    ).to(device).eval()
    # initialise biases (required by patched model)
    for layer in dense.model.layers:
        layer.self_attn.o_proj.bias = torch.nn.Parameter(
            torch.zeros(layer.self_attn.o_proj.out_features,
                        dtype=torch.float16, device=device))
        layer.mlp.down_proj.bias = torch.nn.Parameter(
            torch.zeros(layer.mlp.down_proj.out_features,
                        dtype=torch.float16, device=device))
    dense_params = sum(p.numel() for p in dense.parameters())
    print(f"  Dense params : {dense_params/1e9:.2f} B")
    print(f"  GPU memory   :\n{mem_str(device)}")

    print(f"\n  Loading compressed model from: {args.save_dir}")
    compressed = LlamaForCausalLM.from_pretrained(
        args.save_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    comp_params  = sum(p.numel() for p in compressed.parameters())
    param_reduction = (dense_params - comp_params) / dense_params * 100
    print(f"  Compressed params : {comp_params/1e9:.2f} B  "
          f"({param_reduction:.1f}% reduction)")
    print(f"  GPU memory        :\n{mem_str(device)}")

    # ── 3. Per-layer shape printout ────────────────────────────────
    section("STEP 3 – COMPRESSED LAYER SHAPES")
    print(f"  {'Layer':>5}  {'Heads':>8}  {'Attn dim':>10}  {'MLP dim':>10}")
    for i, layer in enumerate(compressed.model.layers):
        h = layer.self_attn.num_heads
        a = layer.self_attn.q_proj.weight.shape[0]
        m = layer.mlp.up_proj.weight.shape[0]
        print(f"  {i:>5}  {h:>8}  {a:>10}  {m:>10}")

    # ── 4. Benchmark ───────────────────────────────────────────────
    section("STEP 4 – GENERATION BENCHMARK")
    print(f"  Prompts : {PROMPTS}")
    print(f"  Runs    : {args.runs} per condition")
    print(f"  Tokens  : {args.tokens}")
    print()
    print(f"  {'Tag':<12} {'Tokens':>6} │ {'Total ms':>14} │ "
          f"{'ms/tok':>10} │ {'tok/s':>8}")
    print("  " + "─" * 58)

    dense_res = bench_model(dense, tokenizer, device,
                            PROMPTS, args.tokens, args.runs, "dense")

    del dense
    torch.cuda.empty_cache()

    comp_res  = bench_model(compressed, tokenizer, device,
                            PROMPTS, args.tokens, args.runs, "compressed")

    # ── 5. Summary table ───────────────────────────────────────────
    section("STEP 5 – SUMMARY")
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
          f"({param_reduction:.1f}% reduction)")


if __name__ == "__main__":
    main()
