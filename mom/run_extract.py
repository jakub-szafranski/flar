#!/usr/bin/env python3
"""
CLI for extracting FLAP masks & bias-compensation vectors.

Outputs a .pt file (+ a human-readable _meta.json) that contains
per-layer attn/mlp masks, pre-computed biases, and baseline inputs.

Example
-------
    python -m mom.run_extract \
        --model meta-llama/Llama-2-7b-hf \
        --calibration_dataset wikitext2 \
        --pruning_ratio 0.2 \
        --output experts/wikitext2_20.pt

    python -m mom.run_extract \
        --model meta-llama/Llama-2-7b-hf \
        --calibration_dataset c4 \
        --pruning_ratio 0.4 \
        --output experts/c4_40.pt
"""

import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from models.hf_llama.modeling_llama import LlamaForCausalLM

from mom.extract import extract_flap_masks, save_expert


def get_llm(model_name, cache_dir="llm_weights"):
    """Load model with bias-initialised o_proj / down_proj (same as main.py)."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract FLAP pruning masks & biases for MoE routing"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or path (e.g. meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--calibration_dataset", type=str, default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="Calibration dataset used for metric collection",
    )
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.2,
        help="Target sparsity ratio (e.g. 0.2, 0.4)",
    )
    parser.add_argument(
        "--structure", type=str, default="AL-AM",
        choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM"],
    )
    parser.add_argument(
        "--metrics", type=str, default="WIFV",
        choices=["IFV", "WIFV", "WIFN"],
    )
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remove_heads", type=int, default=-1)
    parser.add_argument(
        "--cache_dir", type=str, default="llm_weights",
        help="HuggingFace cache directory for model weights",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for the output .pt file (e.g. experts/wikitext2_20.pt)",
    )
    args = parser.parse_args()

    # reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # load model
    print(f"[mom] loading model: {args.model}")
    model = get_llm(args.model, args.cache_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]

    # extract masks
    expert_data = extract_flap_masks(
        model,
        tokenizer,
        calibration_dataset=args.calibration_dataset,
        pruning_ratio=args.pruning_ratio,
        structure=args.structure,
        metrics=args.metrics,
        nsamples=args.nsamples,
        seed=args.seed,
        remove_heads=args.remove_heads,
        device=device,
    )

    # persist
    expert_data["model"] = args.model
    save_expert(expert_data, args.output)


if __name__ == "__main__":
    main()
