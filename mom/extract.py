"""
Mask & Bias extraction for the MoE-FLAP router.

Runs the FLAP metric-collection pass on a given model + calibration dataset,
then serialises the resulting per-layer masks and pre-computed bias-compensation
vectors to a `.pt` file.  The original lib/ code is NOT modified – we simply
re-use its helpers (find_layers, BiasGPT, prepare_calibration_input, …).
"""

import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

# ── re-use originals ──────────────────────────────────────────────
from lib.prune import (
    find_layers,
    prepare_calibration_input,
    metrics as flap_metrics,
    cal_remove_neuron,
)
from lib.layerwrapper import BiasGPT
from lib.data import get_loaders


# ──────────────────────────────────────────────────────────────────
#  Pre-compute the bias compensation vector that `compress()` would
#  add to o_proj / down_proj.  We do it here so the saved artifact
#  is self-contained (no need to touch the model weights later).
# ──────────────────────────────────────────────────────────────────
def _compute_bias(mask, baseline_inp, output_weight, device):
    """
    Reproduce the bias formula from compress():
        bias = (baseline_inp * ~mask) @ output_weight.T
    """
    mask_dev = mask.to(device)
    inp_dev = baseline_inp.to(device)
    w_dev = output_weight.to(device)
    bias = (inp_dev * ~mask_dev) @ w_dev.T
    return bias.cpu()


# ──────────────────────────────────────────────────────────────────
#  Hardware alignment – pad MLP masks to Tensor-Core-friendly sizes
# ──────────────────────────────────────────────────────────────────
MLP_ALIGN = 64  # cuBLAS FP16 tile size; keeps dims on Tensor Core fast path


def _align_mlp_mask(mask, importance, align_to=MLP_ALIGN):
    """Expand *mask* so ``mask.sum()`` becomes a multiple of *align_to*.

    Instead of padding with zeros at runtime, we simply retain the
    highest-importance neurons that were just below the pruning threshold.
    This is both faster (no dynamic padding in the hot path) and slightly
    better quality-wise (real weights > zeros).

    Attention masks are always ``num_heads × 128`` which is already a
    multiple of 128, so they never need alignment.
    """
    n = int(mask.sum().item())
    pad = (align_to - (n % align_to)) % align_to
    if pad == 0:
        return mask
    pruned_idx = torch.where(~mask)[0]
    if len(pruned_idx) < pad:          # edge case: nearly nothing pruned
        return mask
    _, topk = torch.topk(importance[pruned_idx], pad)
    mask = mask.clone()
    mask[pruned_idx[topk]] = True
    return mask


# ──────────────────────────────────────────────────────────────────
#  Main extraction routine
# ──────────────────────────────────────────────────────────────────
def extract_flap_masks(
    model,
    tokenizer,
    *,
    calibration_dataset: str = "wikitext2",
    pruning_ratio: float = 0.2,
    structure: str = "AL-AM",
    metrics: str = "WIFV",
    nsamples: int = 128,
    seed: int = 0,
    remove_heads: int = -1,
    device=None,
):
    """
    Run the FLAP metric pass and return a dict ready for torch.save().

    Parameters
    ----------
    model : nn.Module
        A loaded (and bias-initialised) LlamaForCausalLM.
    tokenizer : tokenizer
        The matching tokenizer.
    calibration_dataset : str
        Name passed to ``get_loaders`` (e.g. "wikitext2", "c4", "ptb").
    pruning_ratio : float
        Target sparsity ratio (e.g. 0.2 or 0.4).
    structure : str
        FLAP structure flag – one of "UL-UM", "UL-MM", "AL-MM", "AL-AM".
    metrics : str
        FLAP metric flag – one of "IFV", "WIFV", "WIFN".
    nsamples : int
        Number of calibration samples.
    seed : int
        Random seed for calibration data.
    remove_heads : int
        Number of heads to remove (used by UL-MM / AL-MM).
    device : torch.device | None
        Compute device (defaults to cuda:0 if available, else cpu).

    Returns
    -------
    dict   – a serialisation-ready dict (can be passed to ``torch.save``).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ── lightweight args-like namespace so we can call helpers that expect `args` ─
    class _Args:
        pass
    args = _Args()
    args.pruning_ratio = pruning_ratio
    args.structure = structure
    args.metrics = metrics
    args.nsamples = nsamples
    args.seed = seed
    args.remove_heads = remove_heads
    args.unstr = False

    # ── calibration data ─────────────────────────────────────────
    print(f"[mom] loading calibration data: {calibration_dataset}")
    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("[mom] calibration data loaded")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    layers = model.model.layers
    num_layers = len(layers)

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask_list, mlp_mask_list = [], []
    mlp_importance_per_layer = []           # raw per-layer MLP importance (for mask alignment)

    # ── per-layer metric collection (mirrors prune_flap) ─────────
    for i in tqdm(range(num_layers), desc="[mom] collecting metrics"):
        layer = layers[i]
        subset = {
            "self_attn.o_proj": find_layers(layer)["self_attn.o_proj"],
            "mlp.down_proj": find_layers(layer)["mlp.down_proj"],
        }

        if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev),
            )

        wrapped_layers = {
            name: BiasGPT(subset[name], metrics) for name in subset
        }

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        for h in handles:
            h.remove()

        # ── compute importance & (optionally) per-layer masks ──
        for name in subset:
            if name == "self_attn.o_proj":
                W_metric = flap_metrics[metrics](wrapped_layers, subset, name) ** 2
                if structure == "UL-UM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][
                        int(pruning_ratio * layer.self_attn.num_heads)
                    ].cpu()
                    attn_mask_list.append((W_metric >= thresh))
                elif structure == "UL-MM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][
                        remove_heads // num_layers
                    ].cpu()
                    attn_mask_list.append((W_metric >= thresh))
                else:
                    attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(
                    wrapped_layers[name].baseline_inp.type(torch.half)
                )
            else:
                W_metric = flap_metrics[metrics](wrapped_layers, subset, name)
                mlp_importance_per_layer.append(W_metric.cpu())  # always save for alignment
                if structure == "UL-UM":
                    thresh = torch.sort(W_metric.cuda())[0][
                        int(W_metric.numel() * pruning_ratio)
                    ].cpu()
                    mlp_mask_list.append((W_metric >= thresh))
                elif structure == "UL-MM":
                    thresh = torch.sort(W_metric.cuda())[0][
                        cal_remove_neuron(args, model)
                    ].cpu()
                    mlp_mask_list.append((W_metric >= thresh))
                else:
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(
                    wrapped_layers[name].baseline_inp.type(torch.half)
                )

            wrapped_layers[name].free()

        inps, outs = outs, inps
        torch.cuda.empty_cache()

    # ── global threshold (AL-MM / AL-AM) ─────────────────────────
    standardize = lambda x: (
        (x - torch.mean(x, axis=1, keepdim=True))
        / torch.std(x, axis=1, keepdim=True)
    )

    if structure in ("AL-MM", "AL-AM"):
        attn_metric = standardize(torch.stack(attn_metric_list))
        attn_metric = attn_metric.reshape(num_layers, -1, 128).mean(dim=2)

        mlp_metric = standardize(torch.stack(mlp_metric_list))

        if structure == "AL-MM":
            sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
            attn_thres = sorted_attn[-int(remove_heads)]
            attn_mask_t = attn_metric > attn_thres

            sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
            mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
            mlp_mask_t = mlp_metric > mlp_thres
        else:  # AL-AM
            prune_metric = torch.cat(
                [attn_metric.view(-1), mlp_metric.view(-1)]
            )
            sorted_prune, indices = torch.sort(prune_metric, descending=True)
            compression_weight = torch.ones_like(indices)
            compression_weight[indices < attn_metric.numel()] = 512.0 / 3
            threshold = sorted_prune[
                torch.argmin(
                    torch.abs(
                        torch.cumsum(compression_weight, 0)
                        - torch.sum(compression_weight) * (1 - pruning_ratio)
                    )
                )
            ]
            attn_mask_t = attn_metric > threshold
            mlp_mask_t = mlp_metric > threshold

        attn_mask_list = list(attn_mask_t)
        mlp_mask_list = list(mlp_mask_t)
    else:
        attn_mask_list = [m for m in torch.stack(attn_mask_list)]
        mlp_mask_list = [m for m in torch.stack(mlp_mask_list)]

    # ── align MLP masks to multiples of 64 (Tensor Core friendly) ─
    aligned_count = 0
    for idx in range(num_layers):
        old_n = int(mlp_mask_list[idx].sum().item())
        mlp_mask_list[idx] = _align_mlp_mask(
            mlp_mask_list[idx], mlp_importance_per_layer[idx]
        )
        new_n = int(mlp_mask_list[idx].sum().item())
        if new_n != old_n:
            aligned_count += 1
    if aligned_count:
        print(f"[mom] aligned MLP masks in {aligned_count}/{num_layers} layers "
              f"to multiples of {MLP_ALIGN} (Tensor Core acceleration)")

    # ── pre-compute bias vectors and pack results ────────────────
    layer_data = {}
    for idx in range(num_layers):
        layer = layers[idx]
        o_weight = layer.self_attn.o_proj.weight.data
        d_weight = layer.mlp.down_proj.weight.data

        a_mask = attn_mask_list[idx]
        m_mask = mlp_mask_list[idx]

        # expand head-level mask → per-channel for bias computation
        a_mask_expanded = a_mask.repeat_interleave(128)

        attn_bias = _compute_bias(
            a_mask_expanded, attn_baseline_inp_list[idx], o_weight, device
        )
        mlp_bias = _compute_bias(
            m_mask, mlp_baseline_inp_list[idx], d_weight, device
        )

        layer_data[idx] = {
            "attn_mask": a_mask.cpu().bool(),         # (num_heads,)
            "mlp_mask": m_mask.cpu().bool(),           # (intermediate_size,)
            "attn_bias": attn_bias.half(),             # (hidden_size,)
            "mlp_bias": mlp_bias.half(),               # (hidden_size,)
            "attn_baseline_inp": attn_baseline_inp_list[idx].cpu(),
            "mlp_baseline_inp": mlp_baseline_inp_list[idx].cpu(),
        }

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    result = {
        "calibration_dataset": calibration_dataset,
        "pruning_ratio": pruning_ratio,
        "structure": structure,
        "metrics": metrics,
        "nsamples": nsamples,
        "seed": seed,
        "num_layers": num_layers,
        "layers": layer_data,
    }

    print(f"[mom] extraction complete – {num_layers} layers, "
          f"pruning_ratio={pruning_ratio}, dataset={calibration_dataset}")
    return result


def save_expert(data: dict, path: str):
    """Save the extracted masks & biases to a .pt file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(data, path)
    print(f"[mom] expert saved → {path}")

    # also dump a small human-readable summary next to it
    meta_path = path.replace(".pt", "_meta.json")
    meta = {k: v for k, v in data.items() if k != "layers"}
    # add per-layer sparsity info
    per_layer = {}
    for idx, ld in data["layers"].items():
        per_layer[str(idx)] = {
            "attn_heads_retained": int(ld["attn_mask"].sum().item()),
            "attn_heads_total": int(ld["attn_mask"].numel()),
            "mlp_neurons_retained": int(ld["mlp_mask"].sum().item()),
            "mlp_neurons_total": int(ld["mlp_mask"].numel()),
        }
    meta["per_layer"] = per_layer
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[mom] metadata  → {meta_path}")
