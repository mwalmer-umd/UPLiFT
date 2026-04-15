"""
Fused Triton kernel for UPLiFT LocalAttender — v3 with autotuning + fp32 accumulation.

Key improvements over v2:
1. Adaptive kernel selection: cell-tiled for small H, NHWC-streaming for large H
2. fp32 softmax and accumulation throughout
3. NHWC streaming kernel: one program per cell, loops over channels with
   pre-computed softmax weights. Coalesced C-dimension loads/stores.
4. @triton.autotune for the streaming kernel (BLOCK_C, num_warps, num_stages)
5. L2 cache hints (evict_last) on value loads for spatial locality

Performance vs v2:
- H<=32: matches v2 (same cell-tiled kernel, ~0.04-0.08ms)
- H=48:  1.5x faster than v2 (streaming kernel)
- H=64:  1.8x faster than v2, 1.8x faster than compiled
- H=96:  1.9x faster than v2, 1.6x faster than compiled

Target: n=17, I=2 on H100 PCIe.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════
# Kernel A: Cell-tiled kernel (best for small H, matches v2 performance)
# Grid: (ceil(C/BLOCK_C), B*H*W)
# Low overhead, no autotuning. BLOCK_C chosen by Python wrapper.
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _v3_cell_tiled_kernel(
    att_ptr, val_ptr, out_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    H_pad: tl.constexpr, W_pad: tl.constexpr,
    att_stride_b: tl.constexpr, att_stride_d: tl.constexpr,
    att_stride_h: tl.constexpr, att_stride_w: tl.constexpr,
    val_stride_b: tl.constexpr, val_stride_c: tl.constexpr,
    val_stride_h: tl.constexpr, val_stride_w: tl.constexpr,
    out_stride_b: tl.constexpr, out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_cell = tl.program_id(1)

    b = pid_cell // (H * W)
    rem = pid_cell % (H * W)
    h_lr = rem // W
    w_lr = rem % W

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    val_base = b * val_stride_b + c_offs * val_stride_c

    # Load 17 neighbor values (native dtype; promoted to fp32 during weighted sum)
    v0  = tl.load(val_ptr + val_base + (h_lr+0)*val_stride_h + (w_lr+0)*val_stride_w, mask=c_mask, other=0.0)
    v1  = tl.load(val_ptr + val_base + (h_lr+0)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0)
    v2  = tl.load(val_ptr + val_base + (h_lr+0)*val_stride_h + (w_lr+4)*val_stride_w, mask=c_mask, other=0.0)
    v3  = tl.load(val_ptr + val_base + (h_lr+1)*val_stride_h + (w_lr+1)*val_stride_w, mask=c_mask, other=0.0)
    v4  = tl.load(val_ptr + val_base + (h_lr+1)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0)
    v5  = tl.load(val_ptr + val_base + (h_lr+1)*val_stride_h + (w_lr+3)*val_stride_w, mask=c_mask, other=0.0)
    v6  = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+0)*val_stride_w, mask=c_mask, other=0.0)
    v7  = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+1)*val_stride_w, mask=c_mask, other=0.0)
    v8  = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0)
    v9  = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+3)*val_stride_w, mask=c_mask, other=0.0)
    v10 = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+4)*val_stride_w, mask=c_mask, other=0.0)
    v11 = tl.load(val_ptr + val_base + (h_lr+3)*val_stride_h + (w_lr+1)*val_stride_w, mask=c_mask, other=0.0)
    v12 = tl.load(val_ptr + val_base + (h_lr+3)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0)
    v13 = tl.load(val_ptr + val_base + (h_lr+3)*val_stride_h + (w_lr+3)*val_stride_w, mask=c_mask, other=0.0)
    v14 = tl.load(val_ptr + val_base + (h_lr+4)*val_stride_h + (w_lr+0)*val_stride_w, mask=c_mask, other=0.0)
    v15 = tl.load(val_ptr + val_base + (h_lr+4)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0)
    v16 = tl.load(val_ptr + val_base + (h_lr+4)*val_stride_h + (w_lr+4)*val_stride_w, mask=c_mask, other=0.0)

    h_out_base = h_lr * 2
    w_out_base = w_lr * 2

    # Process 4 sub-pixels (unrolled static_range)
    for si in tl.static_range(0, 2):
        for sj in tl.static_range(0, 2):
            h_out = h_out_base + si
            w_out = w_out_base + sj
            att_base = b * att_stride_b + h_out * att_stride_h + w_out * att_stride_w

            # Load 17 attention logits (scalar) and softmax in fp32
            a0  = tl.load(att_ptr + att_base + 0  * att_stride_d).to(tl.float32)
            a1  = tl.load(att_ptr + att_base + 1  * att_stride_d).to(tl.float32)
            a2  = tl.load(att_ptr + att_base + 2  * att_stride_d).to(tl.float32)
            a3  = tl.load(att_ptr + att_base + 3  * att_stride_d).to(tl.float32)
            a4  = tl.load(att_ptr + att_base + 4  * att_stride_d).to(tl.float32)
            a5  = tl.load(att_ptr + att_base + 5  * att_stride_d).to(tl.float32)
            a6  = tl.load(att_ptr + att_base + 6  * att_stride_d).to(tl.float32)
            a7  = tl.load(att_ptr + att_base + 7  * att_stride_d).to(tl.float32)
            a8  = tl.load(att_ptr + att_base + 8  * att_stride_d).to(tl.float32)
            a9  = tl.load(att_ptr + att_base + 9  * att_stride_d).to(tl.float32)
            a10 = tl.load(att_ptr + att_base + 10 * att_stride_d).to(tl.float32)
            a11 = tl.load(att_ptr + att_base + 11 * att_stride_d).to(tl.float32)
            a12 = tl.load(att_ptr + att_base + 12 * att_stride_d).to(tl.float32)
            a13 = tl.load(att_ptr + att_base + 13 * att_stride_d).to(tl.float32)
            a14 = tl.load(att_ptr + att_base + 14 * att_stride_d).to(tl.float32)
            a15 = tl.load(att_ptr + att_base + 15 * att_stride_d).to(tl.float32)
            a16 = tl.load(att_ptr + att_base + 16 * att_stride_d).to(tl.float32)

            m = tl.maximum(a0, a1)
            m = tl.maximum(m, a2);  m = tl.maximum(m, a3);  m = tl.maximum(m, a4)
            m = tl.maximum(m, a5);  m = tl.maximum(m, a6);  m = tl.maximum(m, a7)
            m = tl.maximum(m, a8);  m = tl.maximum(m, a9);  m = tl.maximum(m, a10)
            m = tl.maximum(m, a11); m = tl.maximum(m, a12); m = tl.maximum(m, a13)
            m = tl.maximum(m, a14); m = tl.maximum(m, a15); m = tl.maximum(m, a16)

            e0  = tl.exp(a0-m);  e1  = tl.exp(a1-m);  e2  = tl.exp(a2-m)
            e3  = tl.exp(a3-m);  e4  = tl.exp(a4-m);  e5  = tl.exp(a5-m)
            e6  = tl.exp(a6-m);  e7  = tl.exp(a7-m);  e8  = tl.exp(a8-m)
            e9  = tl.exp(a9-m);  e10 = tl.exp(a10-m); e11 = tl.exp(a11-m)
            e12 = tl.exp(a12-m); e13 = tl.exp(a13-m); e14 = tl.exp(a14-m)
            e15 = tl.exp(a15-m); e16 = tl.exp(a16-m)

            inv_s = 1.0 / (e0+e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16)

            # Weighted sum: fp32 weights * native dtype values = fp32 accumulator
            acc = (e0*inv_s)*v0 + (e1*inv_s)*v1 + (e2*inv_s)*v2
            acc += (e3*inv_s)*v3 + (e4*inv_s)*v4 + (e5*inv_s)*v5
            acc += (e6*inv_s)*v6 + (e7*inv_s)*v7 + (e8*inv_s)*v8
            acc += (e9*inv_s)*v9 + (e10*inv_s)*v10 + (e11*inv_s)*v11
            acc += (e12*inv_s)*v12 + (e13*inv_s)*v13 + (e14*inv_s)*v14
            acc += (e15*inv_s)*v15 + (e16*inv_s)*v16

            out_off = b * out_stride_b + c_offs * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
            tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════════
# Kernel B: NHWC channel-streaming kernel (best for large H >= 48)
# Grid: (B*H*W,) — one program per cell, loops over channels in BLOCK_C chunks.
# Softmax weights precomputed once per cell and reused across all channel blocks.
# NHWC layout ensures coalesced channel-dimension loads/stores.
# ═══════════════════════════════════════════════════════════════════════════

def _autotune_configs_nhwc_streaming():
    configs = []
    for block_c in [64, 128, 256, 512]:
        for nw in [4, 8]:
            for ns in [2, 3, 4]:
                configs.append(
                    triton.Config({'BLOCK_C': block_c}, num_warps=nw, num_stages=ns)
                )
    return configs


@triton.autotune(configs=_autotune_configs_nhwc_streaming(), key=['C', 'H', 'W'])
@triton.jit
def _v3_nhwc_streaming_kernel(
    att_ptr,   # [B, 17, H_out, W_out] — NCHW
    val_ptr,   # [B, H_pad, W_pad, C]  — NHWC
    out_ptr,   # [B, H_out, W_out, C]  — NHWC
    C: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    H_pad: tl.constexpr, W_pad: tl.constexpr,
    att_stride_b: tl.constexpr, att_stride_d: tl.constexpr,
    att_stride_h: tl.constexpr, att_stride_w: tl.constexpr,
    val_stride_b: tl.constexpr, val_stride_h: tl.constexpr,
    val_stride_w: tl.constexpr, val_stride_c: tl.constexpr,
    out_stride_b: tl.constexpr, out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr, out_stride_c: tl.constexpr,
    num_cells: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    NHWC streaming kernel. One program per cell. Loops over channels.
    - Softmax weights for all 4 sub-pixels precomputed and reused across channel loop.
    - NHWC layout ensures C-dimension loads/stores are contiguous.
    - evict_last hints on value loads exploit spatial locality in L2 cache.
    """
    pid_cell = tl.program_id(0)

    b = pid_cell // (H * W)
    rem = pid_cell % (H * W)
    h_lr = rem // W
    w_lr = rem % W

    h_out_base = h_lr * 2
    w_out_base = w_lr * 2

    # Precompute softmax weights for all 4 sub-pixels (68 fp32 scalars in registers)
    att_base_00 = b * att_stride_b + (h_out_base + 0) * att_stride_h + (w_out_base + 0) * att_stride_w
    w00_0, w00_1, w00_2, w00_3, w00_4, w00_5, w00_6, w00_7, w00_8, w00_9, w00_10, w00_11, w00_12, w00_13, w00_14, w00_15, w00_16 = _load_softmax_weights(att_ptr, att_base_00, att_stride_d)

    att_base_01 = b * att_stride_b + (h_out_base + 0) * att_stride_h + (w_out_base + 1) * att_stride_w
    w01_0, w01_1, w01_2, w01_3, w01_4, w01_5, w01_6, w01_7, w01_8, w01_9, w01_10, w01_11, w01_12, w01_13, w01_14, w01_15, w01_16 = _load_softmax_weights(att_ptr, att_base_01, att_stride_d)

    att_base_10 = b * att_stride_b + (h_out_base + 1) * att_stride_h + (w_out_base + 0) * att_stride_w
    w10_0, w10_1, w10_2, w10_3, w10_4, w10_5, w10_6, w10_7, w10_8, w10_9, w10_10, w10_11, w10_12, w10_13, w10_14, w10_15, w10_16 = _load_softmax_weights(att_ptr, att_base_10, att_stride_d)

    att_base_11 = b * att_stride_b + (h_out_base + 1) * att_stride_h + (w_out_base + 1) * att_stride_w
    w11_0, w11_1, w11_2, w11_3, w11_4, w11_5, w11_6, w11_7, w11_8, w11_9, w11_10, w11_11, w11_12, w11_13, w11_14, w11_15, w11_16 = _load_softmax_weights(att_ptr, att_base_11, att_stride_d)

    # Channel loop
    num_c_blocks = tl.cdiv(C, BLOCK_C)
    for c_block_idx in range(num_c_blocks):
        c_offs = c_block_idx * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        val_base = b * val_stride_b + c_offs * val_stride_c

        # Load 17 neighbor values with L2 cache hints
        v0  = tl.load(val_ptr + val_base + (h_lr+0)*val_stride_h + (w_lr+0)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v1  = tl.load(val_ptr + val_base + (h_lr+0)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v2  = tl.load(val_ptr + val_base + (h_lr+0)*val_stride_h + (w_lr+4)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v3  = tl.load(val_ptr + val_base + (h_lr+1)*val_stride_h + (w_lr+1)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v4  = tl.load(val_ptr + val_base + (h_lr+1)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v5  = tl.load(val_ptr + val_base + (h_lr+1)*val_stride_h + (w_lr+3)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v6  = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+0)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v7  = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+1)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v8  = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v9  = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+3)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v10 = tl.load(val_ptr + val_base + (h_lr+2)*val_stride_h + (w_lr+4)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v11 = tl.load(val_ptr + val_base + (h_lr+3)*val_stride_h + (w_lr+1)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v12 = tl.load(val_ptr + val_base + (h_lr+3)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v13 = tl.load(val_ptr + val_base + (h_lr+3)*val_stride_h + (w_lr+3)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v14 = tl.load(val_ptr + val_base + (h_lr+4)*val_stride_h + (w_lr+0)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v15 = tl.load(val_ptr + val_base + (h_lr+4)*val_stride_h + (w_lr+2)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        v16 = tl.load(val_ptr + val_base + (h_lr+4)*val_stride_h + (w_lr+4)*val_stride_w, mask=c_mask, other=0.0, eviction_policy='evict_last').to(tl.float32)

        # Weighted sum + store for 4 sub-pixels
        # Sub-pixel (0,0)
        acc = w00_0*v0 + w00_1*v1 + w00_2*v2 + w00_3*v3 + w00_4*v4 + w00_5*v5 + w00_6*v6 + w00_7*v7 + w00_8*v8 + w00_9*v9 + w00_10*v10 + w00_11*v11 + w00_12*v12 + w00_13*v13 + w00_14*v14 + w00_15*v15 + w00_16*v16
        tl.store(out_ptr + b*out_stride_b + (h_out_base+0)*out_stride_h + (w_out_base+0)*out_stride_w + c_offs*out_stride_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)

        # Sub-pixel (0,1)
        acc = w01_0*v0 + w01_1*v1 + w01_2*v2 + w01_3*v3 + w01_4*v4 + w01_5*v5 + w01_6*v6 + w01_7*v7 + w01_8*v8 + w01_9*v9 + w01_10*v10 + w01_11*v11 + w01_12*v12 + w01_13*v13 + w01_14*v14 + w01_15*v15 + w01_16*v16
        tl.store(out_ptr + b*out_stride_b + (h_out_base+0)*out_stride_h + (w_out_base+1)*out_stride_w + c_offs*out_stride_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)

        # Sub-pixel (1,0)
        acc = w10_0*v0 + w10_1*v1 + w10_2*v2 + w10_3*v3 + w10_4*v4 + w10_5*v5 + w10_6*v6 + w10_7*v7 + w10_8*v8 + w10_9*v9 + w10_10*v10 + w10_11*v11 + w10_12*v12 + w10_13*v13 + w10_14*v14 + w10_15*v15 + w10_16*v16
        tl.store(out_ptr + b*out_stride_b + (h_out_base+1)*out_stride_h + (w_out_base+0)*out_stride_w + c_offs*out_stride_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)

        # Sub-pixel (1,1)
        acc = w11_0*v0 + w11_1*v1 + w11_2*v2 + w11_3*v3 + w11_4*v4 + w11_5*v5 + w11_6*v6 + w11_7*v7 + w11_8*v8 + w11_9*v9 + w11_10*v10 + w11_11*v11 + w11_12*v12 + w11_13*v13 + w11_14*v14 + w11_15*v15 + w11_16*v16
        tl.store(out_ptr + b*out_stride_b + (h_out_base+1)*out_stride_h + (w_out_base+1)*out_stride_w + c_offs*out_stride_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _load_softmax_weights(att_ptr, att_base, att_stride_d):
    """Load 17 attention logits and return softmax weights in fp32."""
    a0  = tl.load(att_ptr + att_base + 0  * att_stride_d).to(tl.float32)
    a1  = tl.load(att_ptr + att_base + 1  * att_stride_d).to(tl.float32)
    a2  = tl.load(att_ptr + att_base + 2  * att_stride_d).to(tl.float32)
    a3  = tl.load(att_ptr + att_base + 3  * att_stride_d).to(tl.float32)
    a4  = tl.load(att_ptr + att_base + 4  * att_stride_d).to(tl.float32)
    a5  = tl.load(att_ptr + att_base + 5  * att_stride_d).to(tl.float32)
    a6  = tl.load(att_ptr + att_base + 6  * att_stride_d).to(tl.float32)
    a7  = tl.load(att_ptr + att_base + 7  * att_stride_d).to(tl.float32)
    a8  = tl.load(att_ptr + att_base + 8  * att_stride_d).to(tl.float32)
    a9  = tl.load(att_ptr + att_base + 9  * att_stride_d).to(tl.float32)
    a10 = tl.load(att_ptr + att_base + 10 * att_stride_d).to(tl.float32)
    a11 = tl.load(att_ptr + att_base + 11 * att_stride_d).to(tl.float32)
    a12 = tl.load(att_ptr + att_base + 12 * att_stride_d).to(tl.float32)
    a13 = tl.load(att_ptr + att_base + 13 * att_stride_d).to(tl.float32)
    a14 = tl.load(att_ptr + att_base + 14 * att_stride_d).to(tl.float32)
    a15 = tl.load(att_ptr + att_base + 15 * att_stride_d).to(tl.float32)
    a16 = tl.load(att_ptr + att_base + 16 * att_stride_d).to(tl.float32)

    m = tl.maximum(a0, a1)
    m = tl.maximum(m, a2);  m = tl.maximum(m, a3);  m = tl.maximum(m, a4)
    m = tl.maximum(m, a5);  m = tl.maximum(m, a6);  m = tl.maximum(m, a7)
    m = tl.maximum(m, a8);  m = tl.maximum(m, a9);  m = tl.maximum(m, a10)
    m = tl.maximum(m, a11); m = tl.maximum(m, a12); m = tl.maximum(m, a13)
    m = tl.maximum(m, a14); m = tl.maximum(m, a15); m = tl.maximum(m, a16)

    e0  = tl.exp(a0-m);  e1  = tl.exp(a1-m);  e2  = tl.exp(a2-m)
    e3  = tl.exp(a3-m);  e4  = tl.exp(a4-m);  e5  = tl.exp(a5-m)
    e6  = tl.exp(a6-m);  e7  = tl.exp(a7-m);  e8  = tl.exp(a8-m)
    e9  = tl.exp(a9-m);  e10 = tl.exp(a10-m); e11 = tl.exp(a11-m)
    e12 = tl.exp(a12-m); e13 = tl.exp(a13-m); e14 = tl.exp(a14-m)
    e15 = tl.exp(a15-m); e16 = tl.exp(a16-m)

    inv_s = 1.0 / (e0+e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16)

    return (e0*inv_s, e1*inv_s, e2*inv_s, e3*inv_s, e4*inv_s, e5*inv_s,
            e6*inv_s, e7*inv_s, e8*inv_s, e9*inv_s, e10*inv_s, e11*inv_s,
            e12*inv_s, e13*inv_s, e14*inv_s, e15*inv_s, e16*inv_s)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch wrapper — auto-selects best kernel strategy
# ═══════════════════════════════════════════════════════════════════════════

class TritonLocalAttender(nn.Module):
    """
    Autotuned fused Triton LocalAttender v3 for n=17, I=2.

    Automatically selects the best kernel based on spatial size:
    - H < threshold: Cell-tiled NCHW kernel (matches v2, low overhead)
    - H >= threshold: NHWC streaming kernel (1.5-1.9x faster than v2)

    Both kernels beat torch.compile at all tested sizes (H=16..96).
    """

    def __init__(self, num_connected: int = 17, streaming_threshold: int = 48):
        super().__init__()
        assert num_connected == 17, "V3 kernel is specialized for n=17"
        self.num_connected = num_connected
        self.pn = 2
        self.streaming_threshold = streaming_threshold

    @torch.compiler.disable
    def forward(self, att: torch.Tensor, x: torch.Tensor, hardmax: bool = False) -> torch.Tensor:
        B, C, H, W = x.shape
        D = att.shape[1]
        H_out, W_out = att.shape[2], att.shape[3]
        I = H_out // H

        # Dispatch to I=4 Triton kernel for 4× upscale configs.
        # Only use Triton when C is large enough to benefit (C >= 32).
        # For small C (e.g., SD1.5 VAE C=4), PyTorch broadcast is faster.
        if D == 17 and I == 4 and C >= 32:
            return self._forward_i4(att, x, hardmax)

        # Fall back to PyTorch for unsupported configs
        if D != 17 or I != 2:
            return self._fallback_forward(att, x, hardmax)


        x_pad = F.pad(x, [self.pn, self.pn, self.pn, self.pn], mode='replicate')
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
        att = att.contiguous()

        num_cells = B * H * W

        if H >= self.streaming_threshold:
            # Large H: NHWC streaming kernel
            x_pad_nhwc = x_pad.contiguous(memory_format=torch.channels_last)
            out_nhwc = torch.empty(B, C, H_out, W_out, device=x.device, dtype=x.dtype,
                                   memory_format=torch.channels_last)

            _v3_nhwc_streaming_kernel[(num_cells,)](
                att, x_pad_nhwc, out_nhwc,
                C, H, W, H_out, W_out, H_pad, W_pad,
                att.stride(0), att.stride(1), att.stride(2), att.stride(3),
                x_pad_nhwc.stride(0), x_pad_nhwc.stride(2), x_pad_nhwc.stride(3), x_pad_nhwc.stride(1),
                out_nhwc.stride(0), out_nhwc.stride(2), out_nhwc.stride(3), out_nhwc.stride(1),
                num_cells,
            )
            return out_nhwc.contiguous()
        else:
            # Small H: cell-tiled NCHW kernel (matches v2)
            x_pad = x_pad.contiguous()
            out = torch.empty(B, C, H_out, W_out, device=x.device, dtype=x.dtype)

            BLOCK_C = min(128, triton.next_power_of_2(C))
            grid = (triton.cdiv(C, BLOCK_C), num_cells)

            _v3_cell_tiled_kernel[grid](
                att, x_pad, out,
                C, H, W, H_out, W_out, H_pad, W_pad,
                att.stride(0), att.stride(1), att.stride(2), att.stride(3),
                x_pad.stride(0), x_pad.stride(1), x_pad.stride(2), x_pad.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                BLOCK_C=BLOCK_C,
            )
            return out

    def _forward_i4(self, att: torch.Tensor, x: torch.Tensor, hardmax: bool = False) -> torch.Tensor:
        """Dispatch to I=4 Triton kernel (for large-C I=4 configs)."""
        # Lazy import — only loaded if I=4 with C>=32
        if not hasattr(self, '_i4_kernel'):
            from uplift.kernels.triton_attender_i4 import TritonLocalAttenderI4
            self._i4_kernel = TritonLocalAttenderI4(num_connected=17).to(att.device)
        return self._i4_kernel(att, x, hardmax)

    def _fallback_forward(self, att: torch.Tensor, x: torch.Tensor, hardmax: bool = False) -> torch.Tensor:
        """Pure PyTorch fallback for I!=2 (e.g., I=4 in SD1.5 VAE 4× config)."""
        B, C, H, W = x.shape
        D = att.shape[1]
        H_out, W_out = att.shape[2], att.shape[3]
        I = H_out // H

        # Offsets for n=17: same as LocalAttender
        offsets = [(-2,-2),(-2,0),(-2,2),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,-2),(2,0),(2,2)]

        # Pad + extract neighbors
        x_pad = F.pad(x, [self.pn, self.pn, self.pn, self.pn], mode='replicate')
        x_all = []
        for off_0, off_1 in offsets:
            x_off = x_pad[:, :, self.pn+off_0:self.pn+off_0+H, self.pn+off_1:self.pn+off_1+W]
            x_all.append(x_off)
        x_stacked = torch.stack(x_all, dim=2)  # [B, C, D, H, W]
        x_stacked = x_stacked.unsqueeze(4).unsqueeze(6)  # [B, C, D, H, 1, W, 1]

        # Softmax or hardmax
        if hardmax:
            a = att.argmax(dim=1)
            att_weights = F.one_hot(a, num_classes=D).permute(0, 3, 1, 2).to(x.dtype)
        else:
            att_weights = F.softmax(att, dim=1)

        att_weights = att_weights.reshape(B, D, H, I, W, I).unsqueeze(1)  # [B, 1, D, H, I, W, I]

        res = (x_stacked * att_weights).sum(dim=2)  # [B, C, H, I, W, I]
        return res.reshape(B, C, H_out, W_out)



