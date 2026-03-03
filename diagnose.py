#!/usr/bin/env python3
"""
Minimal diagnostic to isolate the structured-pruning slowdown.

Tests 3 levels independently:
  A) Raw F.linear() calls with dense vs pruned weight shapes
  B) Single-token model.forward() (incremental decode step)
  C) Full model.generate() for 50 tokens

Each is tested under: dense, unstructured-pruned, structured-pruned
This tells us whether the bottleneck is cuBLAS kernels, model forward,
or HuggingFace generate() orchestration.

Usage:
    python diagnose.py --model huggyllama/llama-7b \
                       --expert experts/c4_p0.2_WIFV_ALAM_llama7b.pt
"""
import argparse, torch, time, numpy as np
from transformers import AutoTokenizer
from mom.prunable_llm import PrunableLLM

# ── helpers ──────────────────────────────────────────────────────
def cuda_ms(fn, device, n=20):
    """Time *fn* on GPU, return (mean, std) in ms over *n* runs."""
    # warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(n):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize(device)
        ts.append(t0.elapsed_time(t1))
    a = np.array(ts)
    return float(a.mean()), float(a.std())


def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ── main ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="huggyllama/llama-7b")
    ap.add_argument("--expert", default="experts/c4_p0.2_WIFV_ALAM_llama7b.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    # load
    wrapper = PrunableLLM.from_pretrained(args.model, device=dev)
    wrapper.model.eval()
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir="llm_weights")
    tok.pad_token = tok.eos_token
    expert = torch.load(args.expert, map_location="cpu")

    # ─── 0) Sanity: print dtypes & use_cache ─────────────────────
    section("SANITY CHECKS")
    l0 = wrapper.model.model.layers[0]
    print(f"  use_cache (config)  : {wrapper.model.config.use_cache}")
    print(f"  q_proj weight dtype : {l0.self_attn.q_proj.weight.dtype}")
    print(f"  o_proj bias dtype   : {l0.self_attn.o_proj.bias.dtype}")
    print(f"  down_proj bias dtype: {l0.mlp.down_proj.bias.dtype}")

    # ─── A) Raw F.linear timing ──────────────────────────────────
    section("A) RAW F.linear TIMING  (layer 13 gate_proj)")
    # Layer 13 is heavily pruned (20 heads, 6848 MLP)

    # Dense gate_proj shape: (11008, 4096) -- weight layout
    x_dense = torch.randn(1, 1, 4096, dtype=torch.float16, device=dev)
    w_dense = torch.randn(11008, 4096, dtype=torch.float16, device=dev)
    b_none = None

    m_d, s_d = cuda_ms(lambda: torch.nn.functional.linear(x_dense, w_dense, b_none), dev, n=100)
    print(f"  Dense   (11008, 4096): {m_d:.3f} ± {s_d:.3f} ms")

    # Pruned gate_proj -- get the actual shape from pruned layer 13
    wrapper.prune(expert, unstr=False)
    l13 = wrapper.model.model.layers[13]
    w_shape = l13.mlp.gate_proj.weight.shape
    w_pruned = torch.randn(*w_shape, dtype=torch.float16, device=dev)
    x_pruned = torch.randn(1, 1, w_shape[1], dtype=torch.float16, device=dev)

    m_p, s_p = cuda_ms(lambda: torch.nn.functional.linear(x_pruned, w_pruned, b_none), dev, n=100)
    print(f"  Pruned  {tuple(w_shape)}: {m_p:.3f} ± {s_p:.3f} ms")
    print(f"  Speedup: {(m_d - m_p)/m_d*100:+.1f}%")

    # Also test with the ACTUAL weight tensor (not a fresh random one)
    w_actual = l13.mlp.gate_proj.weight.data
    print(f"\n  Actual pruned weight tensor properties:")
    print(f"    dtype       : {w_actual.dtype}")
    print(f"    is_contiguous: {w_actual.is_contiguous()}")
    print(f"    device      : {w_actual.device}")
    print(f"    stride      : {w_actual.stride()}")
    print(f"    shape       : {w_actual.shape}")
    m_a, s_a = cuda_ms(lambda: torch.nn.functional.linear(x_pruned, w_actual, b_none), dev, n=100)
    print(f"  Actual  {tuple(w_actual.shape)}: {m_a:.3f} ± {s_a:.3f} ms")
    wrapper.unprune()

    # ─── B) Single-token forward pass ────────────────────────────
    section("B) SINGLE-TOKEN FORWARD PASS  (incremental decode)")
    inp = tok("Hello world", return_tensors="pt").to(dev)

    # build KV cache with a prefill
    with torch.no_grad():
        out = wrapper.model(**inp, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(dim=-1)
    pos = torch.tensor([[inp.input_ids.shape[1]]], device=dev)

    # Dense: time single incremental forward
    def fwd_dense():
        with torch.no_grad():
            wrapper.model(next_id, position_ids=pos, past_key_values=past, use_cache=True)

    m_fd, s_fd = cuda_ms(fwd_dense, dev, n=50)
    print(f"  Dense forward  : {m_fd:.3f} ± {s_fd:.3f} ms")

    # Unstructured pruned
    wrapper.prune(expert, unstr=True)
    with torch.no_grad():
        out_u = wrapper.model(**inp, use_cache=True)
    past_u = out_u.past_key_values
    next_id_u = out_u.logits[:, -1:].argmax(dim=-1)

    def fwd_unstr():
        with torch.no_grad():
            wrapper.model(next_id_u, position_ids=pos, past_key_values=past_u, use_cache=True)

    m_fu, s_fu = cuda_ms(fwd_unstr, dev, n=50)
    print(f"  Unstr forward  : {m_fu:.3f} ± {s_fu:.3f} ms")
    wrapper.unprune()

    # Structured pruned
    wrapper.prune(expert, unstr=False)
    print(f"\n  Structured pruned config check:")
    print(f"    use_cache (config): {wrapper.model.config.use_cache}")
    print(f"    layer 13 num_heads: {wrapper.model.model.layers[13].self_attn.num_heads}")
    print(f"    layer 13 q_proj W : {wrapper.model.model.layers[13].self_attn.q_proj.weight.shape}")
    print(f"    layer 13 q_proj W dtype: {wrapper.model.model.layers[13].self_attn.q_proj.weight.dtype}")
    print(f"    layer 13 q_proj W contiguous: {wrapper.model.model.layers[13].self_attn.q_proj.weight.is_contiguous()}")

    with torch.no_grad():
        out_s = wrapper.model(**inp, use_cache=True)
    past_s = out_s.past_key_values
    next_id_s = out_s.logits[:, -1:].argmax(dim=-1)

    # Print KV cache shapes for first few layers
    print(f"\n  KV cache shapes (structured, first 5 layers):")
    for i in range(5):
        k_shape = past_s[i][0].shape
        print(f"    layer {i}: K={k_shape}")

    def fwd_struct():
        with torch.no_grad():
            wrapper.model(next_id_s, position_ids=pos, past_key_values=past_s, use_cache=True)

    m_fs, s_fs = cuda_ms(fwd_struct, dev, n=50)
    print(f"\n  Struct forward : {m_fs:.3f} ± {s_fs:.3f} ms")
    print(f"  Speedup vs dense: {(m_fd - m_fs)/m_fd*100:+.1f}%")
    wrapper.unprune()

    # ─── C) Full generate (50 tokens) ────────────────────────────
    section("C) GENERATE 50 TOKENS")
    gen_inp = tok("The meaning of life is", return_tensors="pt").to(dev)

    def gen(n=50):
        with torch.no_grad():
            wrapper.model.generate(**gen_inp, max_new_tokens=n, do_sample=False)

    # Dense
    torch.cuda.empty_cache()
    m_gd, s_gd = cuda_ms(lambda: gen(50), dev, n=5)
    print(f"  Dense generate    : {m_gd:.1f} ± {s_gd:.1f} ms  ({m_gd/50:.1f} ms/tok)")

    # Unstructured
    wrapper.prune(expert, unstr=True)
    torch.cuda.empty_cache()
    m_gu, s_gu = cuda_ms(lambda: gen(50), dev, n=5)
    print(f"  Unstr generate    : {m_gu:.1f} ± {s_gu:.1f} ms  ({m_gu/50:.1f} ms/tok)")
    wrapper.unprune()

    # Structured
    wrapper.prune(expert, unstr=False)
    torch.cuda.empty_cache()
    m_gs, s_gs = cuda_ms(lambda: gen(50), dev, n=5)
    print(f"  Struct generate   : {m_gs:.1f} ± {s_gs:.1f} ms  ({m_gs/50:.1f} ms/tok)")
    print(f"  Struct vs Dense   : {(m_gd - m_gs)/m_gd*100:+.1f}%")
    print(f"  Unstr  vs Dense   : {(m_gd - m_gu)/m_gd*100:+.1f}%")
    wrapper.unprune()

    # ─── D) Summary ──────────────────────────────────────────────
    section("SUMMARY")
    print(f"  {'Test':<25} {'Dense':>10} {'Unstr':>10} {'Struct':>10}")
    print(f"  {'F.linear (ms)':<25} {m_d:>10.3f} {'n/a':>10} {m_p:>10.3f}")
    print(f"  {'Single fwd (ms)':<25} {m_fd:>10.3f} {m_fu:>10.3f} {m_fs:>10.3f}")
    print(f"  {'Generate 50 tok (ms)':<25} {m_gd:>10.1f} {m_gu:>10.1f} {m_gs:>10.1f}")
    print(f"  {'ms/tok (generate)':<25} {m_gd/50:>10.1f} {m_gu/50:>10.1f} {m_gs/50:>10.1f}")


if __name__ == "__main__":
    main()
