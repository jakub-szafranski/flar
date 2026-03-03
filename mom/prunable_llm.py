"""
PrunableLLM – fast, reversible expert application for the MoE-FLAP router.

Wraps a dense LlamaForCausalLM and exposes ``prune()`` / ``unprune()``
so any pre-extracted expert can be applied and reverted without reloading
the model from disk.

Memory strategy: **delta snapshots on GPU**.  Before pruning, only the
weight rows/columns that will be *removed* are saved to a separate dict
(still on GPU).  Both the pruned model and deltas remain on GPU for
maximum switching speed.  Total VRAM ≈ dense model size.  On
``unprune()`` the full tensors are reconstructed in-place.

Pruning is inlined (no ``compress()`` call) to avoid redundant bias
computation and per-layer ``torch.cuda.empty_cache()`` overheads.

Usage
-----
    from mom.prunable_llm import PrunableLLM

    wrapper = PrunableLLM(model, device="cuda:0")
    expert  = torch.load("experts/wikitext2_20.pt", map_location="cpu")

    wrapper.prune(expert)          # apply masks + bias compensation
    # … run inference …
    wrapper.unprune()              # restore dense weights

    wrapper.prune(other_expert)    # switch to a different expert
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lib.prune import check_sparsity
from models.hf_llama.modeling_llama import LlamaForCausalLM


# ──────────────────────────────────────────────────────────────────
#  Model loader (shared with run_extract / apply_and_eval)
# ──────────────────────────────────────────────────────────────────
def _init_biases(model: nn.Module) -> None:
    """Zero-initialise o_proj / down_proj biases (required by FLAP)."""
    for layer in model.model.layers:
        layer.self_attn.o_proj.bias = nn.Parameter(
            torch.zeros_like(layer.self_attn.o_proj.bias, device="cpu")
        )
        layer.mlp.down_proj.bias = nn.Parameter(
            torch.zeros_like(layer.mlp.down_proj.bias, device="cpu")
        )


def load_llm(model_name: str, cache_dir: str = "llm_weights") -> LlamaForCausalLM:
    """Load a LlamaForCausalLM with bias-initialised projection layers."""
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    _init_biases(model)
    model.seqlen = 128
    return model


# ──────────────────────────────────────────────────────────────────
#  PrunableLLM
# ──────────────────────────────────────────────────────────────────
class PrunableLLM:
    """Reversible expert-pruning wrapper around a dense LlamaForCausalLM.

    Parameters
    ----------
    model : LlamaForCausalLM
        A loaded (bias-initialised) dense model.
    device : torch.device | str | None
        Target compute device.  Defaults to the device of the first param.
    """

    HEAD_DIM = 128  # fixed in patched modeling_llama.py

    def __init__(self, model: nn.Module, *, device=None):
        self.model = model
        self.device = (
            torch.device(device) if device else next(model.parameters()).device
        )
        self._num_layers: int = model.config.num_hidden_layers
        self._orig_num_heads: int = model.config.num_attention_heads
        self._orig_hidden: int = model.config.hidden_size
        self._orig_intermediate: int = model.config.intermediate_size
        self._deltas: dict | None = None
        self._pruned_unstr: bool = False
        self._active_expert: dict | None = None

    # ── public API ───────────────────────────────────────────────

    def prune(self, expert: dict, *, unstr: bool = False) -> None:
        """Apply an expert's masks and bias-compensation vectors.

        If the model is already pruned, it is automatically un-pruned first.
        Pruning logic is inlined (weight slicing + config updates) to skip
        the redundant bias computation and empty_cache() calls in compress().

        Parameters
        ----------
        expert : dict
            A loaded ``.pt`` expert artifact (output of ``extract_flap_masks``).
        unstr : bool
            ``True``  → mask weights in-place (shapes unchanged).
            ``False`` → real structured pruning (rows/cols removed).
        """
        if self._active_expert is not None:
            self._restore()

        self._pruned_unstr = unstr
        self._deltas = {}

        for idx in range(expert["num_layers"]):
            ld = expert["layers"][idx]
            layer = self.model.model.layers[idx]
            dev = layer.self_attn.q_proj.weight.device

            attn_mask = ld["attn_mask"].to(dev)
            mlp_mask = ld["mlp_mask"].to(dev)
            attn_mask_exp = attn_mask.repeat_interleave(self.HEAD_DIM)
            retained_attn = torch.where(attn_mask_exp)[0]
            removed_attn = torch.where(~attn_mask_exp)[0]
            retained_mlp = torch.where(mlp_mask)[0]
            removed_mlp = torch.where(~mlp_mask)[0]

            # ── snapshot removed slices (stay on GPU for speed) ──
            delta: dict = {
                "o_proj.bias": layer.self_attn.o_proj.bias.data.clone(),
                "down_proj.bias": layer.mlp.down_proj.bias.data.clone(),
                "q_removed": layer.self_attn.q_proj.weight.data[removed_attn],
                "k_removed": layer.self_attn.k_proj.weight.data[removed_attn],
                "v_removed": layer.self_attn.v_proj.weight.data[removed_attn],
                "up_removed": layer.mlp.up_proj.weight.data[removed_mlp],
                "gate_removed": layer.mlp.gate_proj.weight.data[removed_mlp],
            }
            if not unstr:
                delta["o_removed_cols"] = (
                    layer.self_attn.o_proj.weight.data[:, removed_attn]
                )
                delta["down_removed_cols"] = (
                    layer.mlp.down_proj.weight.data[:, removed_mlp]
                )
            self._deltas[idx] = delta

            # ── apply masking / pruning (inlined from compress) ──
            if unstr:
                mask_attn = attn_mask_exp.unsqueeze(-1)
                layer.self_attn.q_proj.weight.data *= mask_attn
                layer.self_attn.k_proj.weight.data *= mask_attn
                layer.self_attn.v_proj.weight.data *= mask_attn
                mask_mlp = mlp_mask.unsqueeze(-1)
                layer.mlp.up_proj.weight.data *= mask_mlp
                layer.mlp.gate_proj.weight.data *= mask_mlp
            else:
                # attention: rows from Q/K/V, columns from o_proj
                layer.self_attn.q_proj.weight.data = (
                    layer.self_attn.q_proj.weight.data[retained_attn])
                layer.self_attn.k_proj.weight.data = (
                    layer.self_attn.k_proj.weight.data[retained_attn])
                layer.self_attn.v_proj.weight.data = (
                    layer.self_attn.v_proj.weight.data[retained_attn])
                layer.self_attn.o_proj.weight.data = (
                    layer.self_attn.o_proj.weight.data[:, retained_attn])
                retain_heads = int(attn_mask.sum().item())
                n_attn = int(attn_mask_exp.sum().item())
                layer.self_attn.num_heads = retain_heads
                layer.self_attn.hidden_size = retain_heads * self.HEAD_DIM
                layer.self_attn.q_proj.out_features = n_attn
                layer.self_attn.k_proj.out_features = n_attn
                layer.self_attn.v_proj.out_features = n_attn
                layer.self_attn.o_proj.in_features = n_attn

                # MLP: rows from up/gate, columns from down_proj
                layer.mlp.up_proj.weight.data = (
                    layer.mlp.up_proj.weight.data[retained_mlp])
                layer.mlp.gate_proj.weight.data = (
                    layer.mlp.gate_proj.weight.data[retained_mlp])
                layer.mlp.down_proj.weight.data = (
                    layer.mlp.down_proj.weight.data[:, retained_mlp])
                n_mlp = int(mlp_mask.sum().item())
                layer.mlp.intermediate_size = n_mlp
                layer.mlp.up_proj.out_features = n_mlp
                layer.mlp.gate_proj.out_features = n_mlp
                layer.mlp.down_proj.in_features = n_mlp

            # ── set pre-computed bias compensation ───────────────
            layer.self_attn.o_proj.bias.data = ld["attn_bias"].to(dev)
            layer.mlp.down_proj.bias.data = ld["mlp_bias"].to(dev)

        self._active_expert = expert

    def unprune(self) -> None:
        """Restore the model to its original dense state."""
        if self._active_expert is None:
            return
        self._restore()
        self._active_expert = None
        self._deltas = None

    @property
    def is_pruned(self) -> bool:
        """Whether an expert is currently applied."""
        return self._active_expert is not None

    @property
    def active_expert_info(self) -> dict | None:
        """Metadata of the currently active expert, or ``None``."""
        if self._active_expert is None:
            return None
        return {
            k: self._active_expert.get(k)
            for k in ("calibration_dataset", "pruning_ratio", "structure", "metrics")
        }

    def sparsity(self) -> float:
        """Return the current sparsity ratio via ``check_sparsity()``."""
        return check_sparsity(self.model)

    # ── convenience forward ──────────────────────────────────────

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    # ── classmethod loader ───────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        cache_dir: str = "llm_weights",
        device=None,
    ) -> "PrunableLLM":
        """Load a model from HuggingFace and wrap it."""
        model = load_llm(model_name, cache_dir=cache_dir)
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(dev)
        model.eval()
        return cls(model, device=dev)

    # ── internals ────────────────────────────────────────────────

    def _restore(self) -> None:
        """Reconstruct the full dense weights from pruned + GPU delta."""
        expert = self._active_expert
        unstr = self._pruned_unstr

        H = self._orig_hidden
        I = self._orig_intermediate

        for i in range(self._num_layers):
            delta = self._deltas[i]
            ld = expert["layers"][i]
            layer = self.model.model.layers[i]
            dev = layer.self_attn.q_proj.weight.device

            attn_mask_exp = ld["attn_mask"].repeat_interleave(self.HEAD_DIM)
            retained_attn = torch.where(attn_mask_exp)[0].to(dev)
            removed_attn = torch.where(~attn_mask_exp)[0].to(dev)
            retained_mlp = torch.where(ld["mlp_mask"])[0].to(dev)
            removed_mlp = torch.where(~ld["mlp_mask"])[0].to(dev)

            if unstr:
                # shapes unchanged — write back zeroed values
                layer.self_attn.q_proj.weight.data[removed_attn] = delta["q_removed"]
                layer.self_attn.k_proj.weight.data[removed_attn] = delta["k_removed"]
                layer.self_attn.v_proj.weight.data[removed_attn] = delta["v_removed"]
                layer.mlp.up_proj.weight.data[removed_mlp] = delta["up_removed"]
                layer.mlp.gate_proj.weight.data[removed_mlp] = delta["gate_removed"]
            else:
                # structured: tensors were physically shrunk → reconstruct
                dtype = layer.self_attn.q_proj.weight.dtype
                in_qkv = layer.self_attn.q_proj.in_features
                in_mlp = layer.mlp.up_proj.in_features

                def _rebuild_rows(current, removed, out_dim, in_dim, ret, rem):
                    full = torch.empty(out_dim, in_dim, dtype=dtype, device=dev)
                    full[ret] = current
                    full[rem] = removed
                    return full

                def _rebuild_cols(current, removed, out_dim, orig_in, ret, rem):
                    full = torch.empty(out_dim, orig_in, dtype=dtype, device=dev)
                    full[:, ret] = current
                    full[:, rem] = removed
                    return full

                # q / k / v  – rows
                layer.self_attn.q_proj.weight.data = _rebuild_rows(
                    layer.self_attn.q_proj.weight.data, delta["q_removed"],
                    H, in_qkv, retained_attn, removed_attn)
                layer.self_attn.k_proj.weight.data = _rebuild_rows(
                    layer.self_attn.k_proj.weight.data, delta["k_removed"],
                    H, in_qkv, retained_attn, removed_attn)
                layer.self_attn.v_proj.weight.data = _rebuild_rows(
                    layer.self_attn.v_proj.weight.data, delta["v_removed"],
                    H, in_qkv, retained_attn, removed_attn)
                # o_proj  – columns
                layer.self_attn.o_proj.weight.data = _rebuild_cols(
                    layer.self_attn.o_proj.weight.data, delta["o_removed_cols"],
                    H, H, retained_attn, removed_attn)
                # up / gate  – rows
                layer.mlp.up_proj.weight.data = _rebuild_rows(
                    layer.mlp.up_proj.weight.data, delta["up_removed"],
                    I, in_mlp, retained_mlp, removed_mlp)
                layer.mlp.gate_proj.weight.data = _rebuild_rows(
                    layer.mlp.gate_proj.weight.data, delta["gate_removed"],
                    I, in_mlp, retained_mlp, removed_mlp)
                # down_proj  – columns
                layer.mlp.down_proj.weight.data = _rebuild_cols(
                    layer.mlp.down_proj.weight.data, delta["down_removed_cols"],
                    H, I, retained_mlp, removed_mlp)

                # restore config scalars from model.config originals
                layer.self_attn.num_heads = self._orig_num_heads
                layer.self_attn.hidden_size = H
                layer.self_attn.q_proj.out_features = H
                layer.self_attn.k_proj.out_features = H
                layer.self_attn.v_proj.out_features = H
                layer.self_attn.o_proj.in_features = H
                layer.mlp.intermediate_size = I
                layer.mlp.up_proj.out_features = I
                layer.mlp.gate_proj.out_features = I
                layer.mlp.down_proj.in_features = I

            # biases (always)
            layer.self_attn.o_proj.bias.data = delta["o_proj.bias"]
            layer.mlp.down_proj.bias.data = delta["down_proj.bias"]

    def __repr__(self) -> str:
        info = self.active_expert_info
        status = (
            f"pruned (dataset={info['calibration_dataset']}, "
            f"ratio={info['pruning_ratio']})"
            if info
            else "dense"
        )
        return f"PrunableLLM(layers={self._num_layers}, status={status})"
