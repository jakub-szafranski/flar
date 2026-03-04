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


def _cuda_time_ms(fn, device: torch.device) -> float:
    """Run fn() once and return GPU-side elapsed time in ms."""
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    t0.record()
    fn()
    t1.record()
    torch.cuda.synchronize(device)
    return t0.elapsed_time(t1)


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
    # print actual sparsity (uses lib.check_sparsity under the hood)
    try:
        print(f"  Sparsity (check_sparsity): {wrapper.sparsity()*100:.3f}%")
    except Exception:
        pass
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
GEN_TOKEN_COUNTS = (50, 150)
GEN_TIMING_RUNS = 3
GEN_BATCH_SIZES = (1, 2, 4, 8)


def generation_benchmark(
    wrapper: PrunableLLM,
    tokenizer,
    expert: dict,
    device: torch.device,
    prompts=PROMPTS,
    token_counts=GEN_TOKEN_COUNTS,
    n_runs=GEN_TIMING_RUNS,
    batch_sizes=GEN_BATCH_SIZES,
):
    """Compare greedy-decode throughput: dense vs structured-pruned.

    For each batch size and token count, runs n_runs each of:
      - dense generation
      - pruned generation  (unstr=False only – real structured pruning)

    Prints a table per batch size with columns:
      Tokens | Dense ms | Pruned ms | Speedup% | Net speedup%

    Net speedup accounts for the one-time prune+unprune switch overhead
    (relevant when the model is only temporarily pruned for a single request).
    """
    section("GENERATION SPEED BENCHMARK  (structured pruning only)")

    # ── clean CUDA state: release all stale cached blocks from
    #    previous prune/unprune cycles so new allocations are contiguous
    torch.cuda.empty_cache()
    print(f"  GPU memory (after empty_cache):\n{mem_summary(device)}")

    print(f"  {'Base prompts':>14}: {prompts}")
    print(f"  {'Runs':>14}: {n_runs} per condition")
    print(f"  {'Batch sizes':>14}: {list(batch_sizes)}")

    # ── print per-layer pruned shapes for diagnosis ─────────────
    wrapper.prune(expert, unstr=False)
    print(f"\n  Per-layer pruned shapes (structured):")
    print(f"  {'Layer':>5}  {'Heads':>10}  {'Attn dim':>10}  {'MLP dim':>10}")
    for idx in range(expert["num_layers"]):
        layer = wrapper.model.model.layers[idx]
        h = layer.self_attn.num_heads
        a = layer.self_attn.q_proj.weight.shape[0]
        m = layer.mlp.up_proj.weight.shape[0]
        print(f"  {idx:>5}  {h:>10}  {a:>10}  {m:>10}")
    total_pruned_params = sum(p.numel() for p in wrapper.model.parameters())
    print(f"  Total pruned params : {total_pruned_params / 1e9:.2f} B")
    print(f"  GPU memory (pruned) :\n{mem_summary(device)}")
    wrapper.unprune()

    # ── warmup generation (both dense and pruned) to trigger
    #    cuBLAS algorithm selection before timed runs ─────────────
    #    NOTE: eos_token_id=[] disables early stopping so both
    #    dense and pruned always generate *exactly* max_new_tokens.
    inputs_warmup = tokenizer(prompts[0], return_tensors="pt").to(device)
    print("\n  Running warmup generation (dense)...")
    with torch.no_grad():
        wrapper.model.generate(**inputs_warmup, max_new_tokens=10,
                               do_sample=False, eos_token_id=[])

    print("  Running warmup generation (pruned)...")
    wrapper.prune(expert, unstr=False)
    with torch.no_grad():
        wrapper.model.generate(**inputs_warmup, max_new_tokens=10,
                               do_sample=False, eos_token_id=[])
    wrapper.unprune()
    torch.cuda.empty_cache()

    # ── measure switch overhead (needed for net-speedup column) ──
    switch_ms_list = []
    for _ in range(n_runs):
        wrapper.prune(expert, unstr=False)
        t_un = _cuda_time_ms(wrapper.unprune, device)
        t_pr = _cuda_time_ms(lambda: wrapper.prune(expert, unstr=False), device)
        wrapper.unprune()
        switch_ms_list.append(t_pr + t_un)
    switch_arr = np.array(switch_ms_list)
    switch_mean = float(switch_arr.mean())
    switch_std  = float(switch_arr.std())
    print(f"\n  Switch overhead (prune+unprune): "
          f"{switch_mean:.1f} ± {switch_std:.1f} ms")

    # table layout
    col = [14, 22, 22, 11, 26]
    hdr = [
        "Tokens",
        "Dense (ms)",
        "Pruned (ms)",
        "Speedup",
        "Net speedup (w/ switch)",
    ]
    sep = "─┼─".join("─" * c for c in col)
    row_fmt = " │ ".join(f"{{:^{c}}}" for c in col)

    all_results = {}
    for bs in batch_sizes:
        print(f"\n{'─' * 60}")
        print(f"  Batch size = {bs}")
        print("  " + row_fmt.format(*hdr))
        print("  " + sep)

        # build a fixed batch of `bs` prompts (cycle through base prompts)
        batch_prompts = [prompts[i % len(prompts)] for i in range(bs)]
        batch_inputs_warmup = tokenizer(
            batch_prompts, return_tensors="pt", padding=True
        ).to(device)

        results = {}
        for n_tok in token_counts:
            # ── dense timing ─────────────────────────────────────
            torch.cuda.empty_cache()
            # per-token-count cuBLAS warmup (dense)
            with torch.no_grad():
                wrapper.model.generate(**batch_inputs_warmup,
                                       max_new_tokens=min(n_tok, 20),
                                       do_sample=False, eos_token_id=[])
            dense_times = []
            for _ in range(n_runs):
                batch_in = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True
                ).to(device)
                def _gen_dense(inp=batch_in):
                    with torch.no_grad():
                        wrapper.model.generate(
                            **inp,
                            max_new_tokens=n_tok,
                            do_sample=False,
                            eos_token_id=[],
                        )
                dense_times.append(_cuda_time_ms(_gen_dense, device))

            # ── pruned timing ─────────────────────────────────────
            torch.cuda.empty_cache()
            wrapper.prune(expert, unstr=False)
            torch.cuda.empty_cache()      # defrag after prune allocations

            # per-token-count cuBLAS warmup (pruned)
            with torch.no_grad():
                wrapper.model.generate(**batch_inputs_warmup,
                                       max_new_tokens=min(n_tok, 20),
                                       do_sample=False, eos_token_id=[])
            pruned_times = []
            for _ in range(n_runs):
                batch_in = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True
                ).to(device)
                def _gen_pruned(inp=batch_in):
                    with torch.no_grad():
                        wrapper.model.generate(
                            **inp,
                            max_new_tokens=n_tok,
                            do_sample=False,
                            eos_token_id=[],
                        )
                pruned_times.append(_cuda_time_ms(_gen_pruned, device))
            wrapper.unprune()
            torch.cuda.empty_cache()

            da = np.array(dense_times)
            pa = np.array(pruned_times)
            d_mean, d_std = float(da.mean()), float(da.std())
            p_mean, p_std = float(pa.mean()), float(pa.std())

            speedup_pct = (d_mean - p_mean) / d_mean * 100
            # net: compare dense gen vs (prune + pruned gen + unprune)
            net_pct = (d_mean - (p_mean + switch_mean)) / d_mean * 100

            results[n_tok] = dict(
                d_mean=d_mean, d_std=d_std,
                p_mean=p_mean, p_std=p_std,
                speedup_pct=speedup_pct, net_pct=net_pct,
            )

            print("  " + row_fmt.format(
                str(n_tok),
                f"{d_mean:.1f} ± {d_std:.1f}",
                f"{p_mean:.1f} ± {p_std:.1f}",
                f"{speedup_pct:+.1f}%",
                f"{net_pct:+.1f}%",
            ))

        all_results[bs] = results

    print()
    print("  Net speedup = (dense_gen - (pruned_gen + switch)) / dense_gen")
    print("  Negative net speedup = switch overhead dominates at this seq len.")
    return all_results


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
    parser.add_argument(
        "--batch_sizes", type=str, default="1,2,4,8",
        help="Comma-separated batch sizes for the generation benchmark (default: 1,2,4,8)",
    )
    args = parser.parse_args()
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

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
    correctness = {}
    for unstr in (False, True):
        ok = run_mode(wrapper, tokenizer, expert, device, unstr=unstr)
        correctness["structured" if not unstr else "unstructured"] = ok

    # ── generation speed benchmark (structured only) ──────────────
    generation_benchmark(wrapper, tokenizer, expert, device,
                         batch_sizes=args.batch_sizes)

    # ── final summary ─────────────────────────────────────────────
    section("SUMMARY")
    all_passed = True
    for mode, ok in correctness.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{status}] {mode}")
        if not ok:
            all_passed = False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
