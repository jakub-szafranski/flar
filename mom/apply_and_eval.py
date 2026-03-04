#!/usr/bin/env python3
"""
Load a pre-extracted expert (.pt), apply masks + bias to a fresh model,
evaluate perplexity, and report timings.

Example
-------
    python -m mom.apply_and_eval \
        --model huggyllama/llama-7b \
        --expert experts/wikitext2_20.pt \
        --eval
"""

import argparse
import time
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from models.hf_llama.modeling_llama import LlamaForCausalLM

from lib.prune import compress, check_sparsity
from lib.eval import eval_ppl


# ─────────────────────────────────────────────────────────────────
# Model loading (identical to run_extract / main.py)
# ─────────────────────────────────────────────────────────────────
def get_llm(model_name, cache_dir="llm_weights"):
    """Load model with zero-initialised biases on o_proj / down_proj."""
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        layer = model.model.layers[i]
        layer.self_attn.o_proj.bias = torch.nn.Parameter(
            torch.zeros_like(layer.self_attn.o_proj.bias, device="cpu")
        )
        layer.mlp.down_proj.bias = torch.nn.Parameter(
            torch.zeros_like(layer.mlp.down_proj.bias, device="cpu")
        )
        torch.nn.init.zeros_(layer.self_attn.o_proj.bias)
        torch.nn.init.zeros_(layer.mlp.down_proj.bias)

    model.seqlen = 128
    return model


# ─────────────────────────────────────────────────────────────────
# Apply pre-computed masks & biases
# ─────────────────────────────────────────────────────────────────
def apply_expert(model, expert_data, device, *, unstr=False):
    """
    Apply masks and bias-compensation from a loaded expert dict.

    Uses the original ``compress()`` from lib/prune.py for correctness.
    The bias vectors are **pre-computed** in the .pt file, so we pass
    ``bias=False`` to ``compress()`` and inject the biases ourselves.

    Parameters
    ----------
    model : nn.Module
        Fresh (un-pruned) model with zero-initialised biases.
    expert_data : dict
        Output of ``extract_flap_masks`` / ``torch.load('expert.pt')``.
    device : torch.device
        Target device.
    unstr : bool
        If True, mask weights without reshaping (unstructured style).

    Returns
    -------
    model  –  the mutated model (same object, modified in-place).
    """
    num_layers = expert_data["num_layers"]

    for idx in range(num_layers):
        ld = expert_data["layers"][idx]
        layer = model.model.layers[idx]

        attn_mask = ld["attn_mask"]          # (num_heads,) bool
        mlp_mask = ld["mlp_mask"]            # (intermediate,) bool
        attn_bias = ld["attn_bias"]          # (hidden_size,) half
        mlp_bias = ld["mlp_bias"]            # (hidden_size,) half
        attn_baseline = ld["attn_baseline_inp"]
        mlp_baseline = ld["mlp_baseline_inp"]

        if f"model.layers.{idx}" in getattr(model, "hf_device_map", {}):
            dev = model.hf_device_map[f"model.layers.{idx}"]
        else:
            dev = device

        # ── attention ──
        compress(
            layer,
            attn_mask.to(dev), None,
            attn_baseline.to(dev), None,
            dev,
            bias=True,
            unstr=unstr,
        )
        # Override the bias with our pre-computed version
        # (compress re-computes it from scratch, but ours was computed on
        #  the *un-pruned* weights before any layer was touched)
        layer.self_attn.o_proj.bias.data = attn_bias.to(dev)

        # ── mlp ──
        compress(
            layer,
            None, mlp_mask.to(dev),
            None, mlp_baseline.to(dev),
            dev,
            bias=True,
            unstr=unstr,
        )
        layer.mlp.down_proj.bias.data = mlp_bias.to(dev)

    torch.cuda.empty_cache()
    return model


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Apply a saved FLAP expert and evaluate the pruned model"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--expert", type=str, required=True,
        help="Path to the .pt expert file produced by run_extract",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="llm_weights",
    )
    parser.add_argument(
        "--unstr", action="store_true",
        help="Use unstructured masking (mask only, no real pruning)",
    )
    parser.add_argument(
        "--eval", action="store_true", dest="do_eval",
        help="Run wikitext2 perplexity evaluation after pruning",
    )
    parser.add_argument(
        "--eval_iters", type=int, default=None,
        help="Limit eval to first N iterations (batches) when computing ppl",
    )
    parser.add_argument(
        "--save_model", type=str, default=None,
        help="Path to save the pruned model",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # ── load expert data ──
    print(f"[mom] loading expert: {args.expert}")
    expert_data = torch.load(args.expert, map_location="cpu")
    print(f"[mom] expert config: dataset={expert_data.get('calibration_dataset')}, "
          f"ratio={expert_data.get('pruning_ratio')}, "
          f"structure={expert_data.get('structure')}, "
          f"metrics={expert_data.get('metrics')}")

    # ── load model ──
    print(f"[mom] loading model: {args.model}")
    model = get_llm(args.model, args.cache_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]

    # ── apply masks (timed) ──
    print("[mom] applying expert masks …")
    t0 = time.perf_counter()
    apply_expert(model, expert_data, device, unstr=args.unstr)
    t_prune = time.perf_counter() - t0
    print(f"[mom] pruning applied in {t_prune:.4f}s")

    # ── sparsity check ──
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameters {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print("*" * 30)

    # ── eval (timed) ──
    if args.do_eval:
        print("[mom] evaluating perplexity …")
        t0 = time.perf_counter()
        ppl = eval_ppl(model, tokenizer, device, max_iters=args.eval_iters)
        t_eval = time.perf_counter() - t0
        print(f"ppl on wikitext2: {ppl:.2f}")
        print(f"[mom] eval completed in {t_eval:.4f}s")

    # ── timing summary ──
    print("\n" + "=" * 40)
    print(f"  Pruning time : {t_prune:.4f}s")
    if args.do_eval:
        print(f"  Eval time    : {t_eval:.4f}s")
    print("=" * 40)

    # ── save ──
    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"[mom] pruned model saved → {args.save_model}")


if __name__ == "__main__":
    main()
