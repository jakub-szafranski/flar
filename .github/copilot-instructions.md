### Project Overview: **Dynamic MoE-FLAP Router**

If you change something update in shortly in the copilot-instructions.md!

Treat folder lib as source of truth for the FLAP pruning method. Do not modify it, but feel free to read and understand the code there. The `mom` folder contains our custom implementation of the MoE router and related utilities.

**The Goal:** Build a high-throughput, training-free LLM inference system that uses an RL Agent to dynamically route user prompts to specialized, pruned sub-networks (Experts) based on task complexity and domain.

---

### Core Mechanism: The "Option B" Dense-to-MoE
Instead of a static global pruning threshold, the system treats a single dense LLM as a collection of $K$ virtual experts.

#### 1. Pruning Method (FLAP)
* **Fluctuation-based Adaptive Structured Pruning:** Use this for hardware-friendly (structured) reduction.
* **Why:** It requires zero retraining and uses bias compensation ($B_{\ell}^0$) to recover performance loss instantly by approximating pruned channels with their mean values.

#### 2. Expert Creation (Offline)
* **5 Diverse Calibration Datasets:**
    * *LogiQA* (Logic)
    * *MBPP* (Code)
    * *GSM8K* (Math)
    * *WikiText* (General Syntax)
    * *PubMed/arXiv* (Technical/Scientific)
* **Sparsity Tiers:** For each dataset, generate two pruning masks (20% and 40% sparsity).
* **Total Experts:** $K = 11$ (5 domains $\times$ 2 sparsity levels, plus 1 original Dense Baseline).

#### 3. The RL Agent (The Router)
* **Input (State):** A lightweight text embedding of the user prompt (e.g., from a frozen MiniLM).
* **Output (Action):** A discrete choice ($a \in \{1, \dots, 11\}$) selecting the optimal Expert.
* **Training:** RL-only (PPO/DQN). The LLM weights remain strictly frozen.
* **Reward Function:** $R = \text{Accuracy Score} - \lambda(\text{Compute Cost})$.
    * *Note:* This forces the agent to learn the "difficulty" of a prompt, using the 40% expert for simple tasks and the Dense model only when necessary.

---

### Technical Implementation Details
* **No Weight Swapping:** Keep the full model in VRAM at all times. Use **Index Slicing** (pointers to weight columns) to "activate" a sub-network instantly. This treats the pruned model as a "view" of the original weights, avoiding the latency of unpinning or reloading memory.
* **Bias Compensation:** Each expert utilizes its own unique, pre-computed $B_{\ell}^0$ vector, which is added during the forward pass to maintain signal integrity without fine-tuning or LoRA.
* **Sub-network Indexing:** The system maps the RL agent's discrete action to a pre-defined set of column indices for each layer's weight matrix.

---

### Codebase Layout

```
lib/          – original FLAP source (DO NOT MODIFY – source of truth)
  prune.py    – prune_flap(), compress(), find_layers(), cal_remove_neuron()
  layerwrapper.py – BiasGPT, WrappedGPT
  data.py     – get_loaders() → wikitext2 / c4 / ptb
  eval.py     – eval_ppl()
models/hf_llama/modeling_llama.py – patched LlamaForCausalLM

mom/          – MoE router implementation (our code)
  extract.py        – extract_flap_masks(model, tokenizer, **kwargs) → dict
                      save_expert(data, path)
  run_extract.py    – CLI: collect masks from a calibration dataset → .pt file
  apply_and_eval.py – CLI: load .pt, apply masks/biases, eval PPL + timing
  prunable_llm.py   – PrunableLLM class: reversible prune(expert)/unprune()
                      wrapper; delta snapshots kept on GPU for fast switching;
                      pruning inlined (no compress() call); also exports
                      load_llm() shared model loader
  __init__.py       – re-exports: extract_flap_masks, save_expert,
                      PrunableLLM, load_llm
```

#### Expert artifact format (`.pt`)
Each saved expert is a `dict` with top-level metadata keys (`calibration_dataset`, `pruning_ratio`, `structure`, `metrics`, `nsamples`, `seed`, `num_layers`, `model`) and a `layers` dict keyed by layer index, each containing:
- `attn_mask` `(num_heads,)` bool – heads to **retain**
- `mlp_mask` `(intermediate_size,)` bool – neurons to **retain**
- `attn_bias` / `mlp_bias` `(hidden_size,)` float16 – pre-computed $B_\ell^0$
- `attn_baseline_inp` / `mlp_baseline_inp` – raw channel means (for recomputation)

A human-readable `_meta.json` sidecar is also written next to every `.pt`.

#### Standard CLI defaults (match `huggyllama/llama-7b` experiments)
| param | value |
|---|---|
| `--nsamples` | 128 |
| `--remove_heads` | -1 (AL-AM doesn't use it) |
| `--metrics` | WIFV |
| `--structure` | AL-AM |
| `--seed` | 0 |

---

### Benchmarking & Development Guidelines

#### Prune / Unprune Switching (measured on LLaMA-7B, 20% AL-AM)
| Mode | prune() | unprune() | Total |
|------|---------|-----------|-------|
| Structured | ~141 ms | ~150 ms | **~291 ms** |
| Unstructured | ~107 ms | ~27 ms | **~134 ms** |

- **Use unstructured mode** (`unstr=True`) during RL agent training loops — 2× faster switching, and the forward pass quality difference is identical (same masks, same bias compensation).
- **Use structured mode** (`unstr=False`) for real deployment benchmarks — only structured pruning produces genuinely smaller tensors and memory savings for inference.
- Correctness verified: both modes are **lossless** (dense == unprune(prune(dense)) at token level).

#### Generation Speed Benchmarking
- Always call `torch.cuda.empty_cache()` before timing after any prune/unprune cycles to avoid CUDA memory fragmentation skewing results.
- Run a warmup generation (≥10 tokens) before timed runs so cuBLAS algorithm selection doesn't contaminate timing.
- Print per-layer pruned shapes — if MLP intermediate dims are not multiples of 8, cuBLAS will use slow fallback kernels.
- If structured pruning shows unexpected slowdowns, compare with `mom/apply_and_eval.py` (uses `compress()` directly) to isolate whether the issue is PrunableLLM or GPU/cuBLAS.


---

### Next Steps / Remaining Challenges
* **State Encoder Design:** Selecting a sub-5ms embedding model to ensure the routing overhead doesn't cancel out the pruning gains.
* **Reward Function Tuning:** Carefully balancing the penalty $\lambda$ so the agent doesn't become "lazy" (always picking 40%) or "paranoid" (always picking Dense).
* **Action Mapping:** Ensuring the agent effectively differentiates between domain-specific experts versus general-purpose ones.
* **Custom calibration datasets:** `lib/data.py` currently supports `wikitext2`, `c4`, `ptb`. LogiQA / MBPP / GSM8K / PubMed loaders need to be added there, then referenced via `--calibration_dataset`.