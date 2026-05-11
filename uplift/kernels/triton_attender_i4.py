"""
Fused Triton kernel for UPLiFT LocalAttender — I=4 cell-tiled variant.

Extends v3 to handle up_factor=4 (SD1.5 VAE 4× config). Each low-res cell
maps to I×I = 4×4 = 16 high-res output pixels. All 16 share the same 17
neighbor value loads → 16× data reuse over per-pixel tiling.

Target: n=17, I=4, C=4 (SD1.5 VAE latent) on H100 PCIe.

For C=4, the entire channel dimension fits in a single BLOCK_C=8 block,
so each program handles one complete cell with zero channel looping.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_INT32_MAX = 2_147_483_647

# n=17 offsets: (-2,-2),(-2,0),(-2,2),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),
#               (0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,-2),(2,0),(2,2)
# Padding = 2 (max offset magnitude)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel: Cell-tiled for n=17, I=4
# Grid: (ceil(C/BLOCK_C), B*H*W)
# Each program processes one low-res cell → 16 output pixels (4×4)
# Value neighbors loaded once per cell, reused across all 16 sub-pixels.
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _v3_cell_tiled_i4_kernel(
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
    USE_I64_OFFSETS: tl.constexpr,
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
    if USE_I64_OFFSETS:
        out_batch_base = b.to(tl.int64) * out_stride_b
    else:
        out_batch_base = b * out_stride_b
    out_c = c_offs * out_stride_c

    # ─── Load 17 neighbor values (shared across 4×4=16 output pixels) ───
    # Padded coordinates: (h_lr + PN + oh, w_lr + PN + ow) where PN=2
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

    h_out_base = h_lr * 4
    w_out_base = w_lr * 4

    # ─── Process 4×4=16 sub-pixels with shared values ───
    for si in tl.static_range(0, 4):
        for sj in tl.static_range(0, 4):
            h_out = h_out_base + si
            w_out = w_out_base + sj
            att_base = b * att_stride_b + h_out * att_stride_h + w_out * att_stride_w

            # Load 17 attention logits and softmax in fp32
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

            # Weighted sum: fp32 weights * native dtype values
            acc = (e0*inv_s)*v0 + (e1*inv_s)*v1 + (e2*inv_s)*v2
            acc += (e3*inv_s)*v3 + (e4*inv_s)*v4 + (e5*inv_s)*v5
            acc += (e6*inv_s)*v6 + (e7*inv_s)*v7 + (e8*inv_s)*v8
            acc += (e9*inv_s)*v9 + (e10*inv_s)*v10 + (e11*inv_s)*v11
            acc += (e12*inv_s)*v12 + (e13*inv_s)*v13 + (e14*inv_s)*v14
            acc += (e15*inv_s)*v15 + (e16*inv_s)*v16

            if USE_I64_OFFSETS:
                out_spatial = h_out.to(tl.int64) * out_stride_h + w_out.to(tl.int64) * out_stride_w
            else:
                out_spatial = h_out * out_stride_h + w_out * out_stride_w
            out_off = out_batch_base + out_spatial + out_c
            tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel: NHWC streaming for n=17, I=4 (large H, C > small threshold)
# Grid: (B*H*W,)
# One program per cell, loops over channels with pre-computed softmax weights.
# All 16 sub-pixel softmax weights precomputed upfront for maximum reuse.
# ═══════════════════════════════════════════════════════════════════════════════

def _autotune_configs_i4():
    configs = []
    for block_c in [4, 8, 16, 32, 64, 128]:
        for nw in [4, 8]:
            for ns in [2, 3]:
                configs.append(
                    triton.Config({'BLOCK_C': block_c}, num_warps=nw, num_stages=ns)
                )
    return configs


@triton.jit
def _load_softmax_weights_17(att_ptr, att_base, att_stride_d):
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


@triton.autotune(configs=_autotune_configs_i4(), key=['C', 'H', 'W', 'USE_I64_OFFSETS'])
@triton.jit
def _v3_nhwc_streaming_i4_kernel(
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
    USE_I64_OFFSETS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    NHWC streaming kernel for I=4. One program per cell, loops over channels.
    Pre-computes all 16 sub-pixel softmax weights (16×17 = 272 fp32 scalars)
    and reuses them across the channel loop.
    """
    pid_cell = tl.program_id(0)

    b = pid_cell // (H * W)
    rem = pid_cell % (H * W)
    h_lr = rem // W
    w_lr = rem % W

    h_out_base = h_lr * 4
    w_out_base = w_lr * 4
    if USE_I64_OFFSETS:
        out_base = (
            b.to(tl.int64) * out_stride_b
            + h_out_base.to(tl.int64) * out_stride_h
            + w_out_base.to(tl.int64) * out_stride_w
        )
    else:
        out_base = b * out_stride_b + h_out_base * out_stride_h + w_out_base * out_stride_w

    # Precompute softmax weights for all 16 sub-pixels (272 fp32 scalars)
    # Row 0: sub-pixels (0,0), (0,1), (0,2), (0,3)
    att_b00 = b * att_stride_b + (h_out_base+0) * att_stride_h + (w_out_base+0) * att_stride_w
    w00_0,w00_1,w00_2,w00_3,w00_4,w00_5,w00_6,w00_7,w00_8,w00_9,w00_10,w00_11,w00_12,w00_13,w00_14,w00_15,w00_16 = _load_softmax_weights_17(att_ptr, att_b00, att_stride_d)
    att_b01 = b * att_stride_b + (h_out_base+0) * att_stride_h + (w_out_base+1) * att_stride_w
    w01_0,w01_1,w01_2,w01_3,w01_4,w01_5,w01_6,w01_7,w01_8,w01_9,w01_10,w01_11,w01_12,w01_13,w01_14,w01_15,w01_16 = _load_softmax_weights_17(att_ptr, att_b01, att_stride_d)
    att_b02 = b * att_stride_b + (h_out_base+0) * att_stride_h + (w_out_base+2) * att_stride_w
    w02_0,w02_1,w02_2,w02_3,w02_4,w02_5,w02_6,w02_7,w02_8,w02_9,w02_10,w02_11,w02_12,w02_13,w02_14,w02_15,w02_16 = _load_softmax_weights_17(att_ptr, att_b02, att_stride_d)
    att_b03 = b * att_stride_b + (h_out_base+0) * att_stride_h + (w_out_base+3) * att_stride_w
    w03_0,w03_1,w03_2,w03_3,w03_4,w03_5,w03_6,w03_7,w03_8,w03_9,w03_10,w03_11,w03_12,w03_13,w03_14,w03_15,w03_16 = _load_softmax_weights_17(att_ptr, att_b03, att_stride_d)

    # Row 1
    att_b10 = b * att_stride_b + (h_out_base+1) * att_stride_h + (w_out_base+0) * att_stride_w
    w10_0,w10_1,w10_2,w10_3,w10_4,w10_5,w10_6,w10_7,w10_8,w10_9,w10_10,w10_11,w10_12,w10_13,w10_14,w10_15,w10_16 = _load_softmax_weights_17(att_ptr, att_b10, att_stride_d)
    att_b11 = b * att_stride_b + (h_out_base+1) * att_stride_h + (w_out_base+1) * att_stride_w
    w11_0,w11_1,w11_2,w11_3,w11_4,w11_5,w11_6,w11_7,w11_8,w11_9,w11_10,w11_11,w11_12,w11_13,w11_14,w11_15,w11_16 = _load_softmax_weights_17(att_ptr, att_b11, att_stride_d)
    att_b12 = b * att_stride_b + (h_out_base+1) * att_stride_h + (w_out_base+2) * att_stride_w
    w12_0,w12_1,w12_2,w12_3,w12_4,w12_5,w12_6,w12_7,w12_8,w12_9,w12_10,w12_11,w12_12,w12_13,w12_14,w12_15,w12_16 = _load_softmax_weights_17(att_ptr, att_b12, att_stride_d)
    att_b13 = b * att_stride_b + (h_out_base+1) * att_stride_h + (w_out_base+3) * att_stride_w
    w13_0,w13_1,w13_2,w13_3,w13_4,w13_5,w13_6,w13_7,w13_8,w13_9,w13_10,w13_11,w13_12,w13_13,w13_14,w13_15,w13_16 = _load_softmax_weights_17(att_ptr, att_b13, att_stride_d)

    # Row 2
    att_b20 = b * att_stride_b + (h_out_base+2) * att_stride_h + (w_out_base+0) * att_stride_w
    w20_0,w20_1,w20_2,w20_3,w20_4,w20_5,w20_6,w20_7,w20_8,w20_9,w20_10,w20_11,w20_12,w20_13,w20_14,w20_15,w20_16 = _load_softmax_weights_17(att_ptr, att_b20, att_stride_d)
    att_b21 = b * att_stride_b + (h_out_base+2) * att_stride_h + (w_out_base+1) * att_stride_w
    w21_0,w21_1,w21_2,w21_3,w21_4,w21_5,w21_6,w21_7,w21_8,w21_9,w21_10,w21_11,w21_12,w21_13,w21_14,w21_15,w21_16 = _load_softmax_weights_17(att_ptr, att_b21, att_stride_d)
    att_b22 = b * att_stride_b + (h_out_base+2) * att_stride_h + (w_out_base+2) * att_stride_w
    w22_0,w22_1,w22_2,w22_3,w22_4,w22_5,w22_6,w22_7,w22_8,w22_9,w22_10,w22_11,w22_12,w22_13,w22_14,w22_15,w22_16 = _load_softmax_weights_17(att_ptr, att_b22, att_stride_d)
    att_b23 = b * att_stride_b + (h_out_base+2) * att_stride_h + (w_out_base+3) * att_stride_w
    w23_0,w23_1,w23_2,w23_3,w23_4,w23_5,w23_6,w23_7,w23_8,w23_9,w23_10,w23_11,w23_12,w23_13,w23_14,w23_15,w23_16 = _load_softmax_weights_17(att_ptr, att_b23, att_stride_d)

    # Row 3
    att_b30 = b * att_stride_b + (h_out_base+3) * att_stride_h + (w_out_base+0) * att_stride_w
    w30_0,w30_1,w30_2,w30_3,w30_4,w30_5,w30_6,w30_7,w30_8,w30_9,w30_10,w30_11,w30_12,w30_13,w30_14,w30_15,w30_16 = _load_softmax_weights_17(att_ptr, att_b30, att_stride_d)
    att_b31 = b * att_stride_b + (h_out_base+3) * att_stride_h + (w_out_base+1) * att_stride_w
    w31_0,w31_1,w31_2,w31_3,w31_4,w31_5,w31_6,w31_7,w31_8,w31_9,w31_10,w31_11,w31_12,w31_13,w31_14,w31_15,w31_16 = _load_softmax_weights_17(att_ptr, att_b31, att_stride_d)
    att_b32 = b * att_stride_b + (h_out_base+3) * att_stride_h + (w_out_base+2) * att_stride_w
    w32_0,w32_1,w32_2,w32_3,w32_4,w32_5,w32_6,w32_7,w32_8,w32_9,w32_10,w32_11,w32_12,w32_13,w32_14,w32_15,w32_16 = _load_softmax_weights_17(att_ptr, att_b32, att_stride_d)
    att_b33 = b * att_stride_b + (h_out_base+3) * att_stride_h + (w_out_base+3) * att_stride_w
    w33_0,w33_1,w33_2,w33_3,w33_4,w33_5,w33_6,w33_7,w33_8,w33_9,w33_10,w33_11,w33_12,w33_13,w33_14,w33_15,w33_16 = _load_softmax_weights_17(att_ptr, att_b33, att_stride_d)

    # Channel loop
    num_c_blocks = tl.cdiv(C, BLOCK_C)
    for c_block_idx in range(num_c_blocks):
        c_offs = c_block_idx * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        out_c = c_offs * out_stride_c

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

        # Weighted sum + store for all 16 sub-pixels
        # Row 0
        acc = w00_0*v0+w00_1*v1+w00_2*v2+w00_3*v3+w00_4*v4+w00_5*v5+w00_6*v6+w00_7*v7+w00_8*v8+w00_9*v9+w00_10*v10+w00_11*v11+w00_12*v12+w00_13*v13+w00_14*v14+w00_15*v15+w00_16*v16
        tl.store(out_ptr + out_base + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w01_0*v0+w01_1*v1+w01_2*v2+w01_3*v3+w01_4*v4+w01_5*v5+w01_6*v6+w01_7*v7+w01_8*v8+w01_9*v9+w01_10*v10+w01_11*v11+w01_12*v12+w01_13*v13+w01_14*v14+w01_15*v15+w01_16*v16
        tl.store(out_ptr + out_base + out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w02_0*v0+w02_1*v1+w02_2*v2+w02_3*v3+w02_4*v4+w02_5*v5+w02_6*v6+w02_7*v7+w02_8*v8+w02_9*v9+w02_10*v10+w02_11*v11+w02_12*v12+w02_13*v13+w02_14*v14+w02_15*v15+w02_16*v16
        tl.store(out_ptr + out_base + 2*out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w03_0*v0+w03_1*v1+w03_2*v2+w03_3*v3+w03_4*v4+w03_5*v5+w03_6*v6+w03_7*v7+w03_8*v8+w03_9*v9+w03_10*v10+w03_11*v11+w03_12*v12+w03_13*v13+w03_14*v14+w03_15*v15+w03_16*v16
        tl.store(out_ptr + out_base + 3*out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)

        # Row 1
        acc = w10_0*v0+w10_1*v1+w10_2*v2+w10_3*v3+w10_4*v4+w10_5*v5+w10_6*v6+w10_7*v7+w10_8*v8+w10_9*v9+w10_10*v10+w10_11*v11+w10_12*v12+w10_13*v13+w10_14*v14+w10_15*v15+w10_16*v16
        tl.store(out_ptr + out_base + out_stride_h + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w11_0*v0+w11_1*v1+w11_2*v2+w11_3*v3+w11_4*v4+w11_5*v5+w11_6*v6+w11_7*v7+w11_8*v8+w11_9*v9+w11_10*v10+w11_11*v11+w11_12*v12+w11_13*v13+w11_14*v14+w11_15*v15+w11_16*v16
        tl.store(out_ptr + out_base + out_stride_h + out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w12_0*v0+w12_1*v1+w12_2*v2+w12_3*v3+w12_4*v4+w12_5*v5+w12_6*v6+w12_7*v7+w12_8*v8+w12_9*v9+w12_10*v10+w12_11*v11+w12_12*v12+w12_13*v13+w12_14*v14+w12_15*v15+w12_16*v16
        tl.store(out_ptr + out_base + out_stride_h + 2*out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w13_0*v0+w13_1*v1+w13_2*v2+w13_3*v3+w13_4*v4+w13_5*v5+w13_6*v6+w13_7*v7+w13_8*v8+w13_9*v9+w13_10*v10+w13_11*v11+w13_12*v12+w13_13*v13+w13_14*v14+w13_15*v15+w13_16*v16
        tl.store(out_ptr + out_base + out_stride_h + 3*out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)

        # Row 2
        acc = w20_0*v0+w20_1*v1+w20_2*v2+w20_3*v3+w20_4*v4+w20_5*v5+w20_6*v6+w20_7*v7+w20_8*v8+w20_9*v9+w20_10*v10+w20_11*v11+w20_12*v12+w20_13*v13+w20_14*v14+w20_15*v15+w20_16*v16
        tl.store(out_ptr + out_base + 2*out_stride_h + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w21_0*v0+w21_1*v1+w21_2*v2+w21_3*v3+w21_4*v4+w21_5*v5+w21_6*v6+w21_7*v7+w21_8*v8+w21_9*v9+w21_10*v10+w21_11*v11+w21_12*v12+w21_13*v13+w21_14*v14+w21_15*v15+w21_16*v16
        tl.store(out_ptr + out_base + 2*out_stride_h + out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w22_0*v0+w22_1*v1+w22_2*v2+w22_3*v3+w22_4*v4+w22_5*v5+w22_6*v6+w22_7*v7+w22_8*v8+w22_9*v9+w22_10*v10+w22_11*v11+w22_12*v12+w22_13*v13+w22_14*v14+w22_15*v15+w22_16*v16
        tl.store(out_ptr + out_base + 2*out_stride_h + 2*out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w23_0*v0+w23_1*v1+w23_2*v2+w23_3*v3+w23_4*v4+w23_5*v5+w23_6*v6+w23_7*v7+w23_8*v8+w23_9*v9+w23_10*v10+w23_11*v11+w23_12*v12+w23_13*v13+w23_14*v14+w23_15*v15+w23_16*v16
        tl.store(out_ptr + out_base + 2*out_stride_h + 3*out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)

        # Row 3
        acc = w30_0*v0+w30_1*v1+w30_2*v2+w30_3*v3+w30_4*v4+w30_5*v5+w30_6*v6+w30_7*v7+w30_8*v8+w30_9*v9+w30_10*v10+w30_11*v11+w30_12*v12+w30_13*v13+w30_14*v14+w30_15*v15+w30_16*v16
        tl.store(out_ptr + out_base + 3*out_stride_h + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w31_0*v0+w31_1*v1+w31_2*v2+w31_3*v3+w31_4*v4+w31_5*v5+w31_6*v6+w31_7*v7+w31_8*v8+w31_9*v9+w31_10*v10+w31_11*v11+w31_12*v12+w31_13*v13+w31_14*v14+w31_15*v15+w31_16*v16
        tl.store(out_ptr + out_base + 3*out_stride_h + out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w32_0*v0+w32_1*v1+w32_2*v2+w32_3*v3+w32_4*v4+w32_5*v5+w32_6*v6+w32_7*v7+w32_8*v8+w32_9*v9+w32_10*v10+w32_11*v11+w32_12*v12+w32_13*v13+w32_14*v14+w32_15*v15+w32_16*v16
        tl.store(out_ptr + out_base + 3*out_stride_h + 2*out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)
        acc = w33_0*v0+w33_1*v1+w33_2*v2+w33_3*v3+w33_4*v4+w33_5*v5+w33_6*v6+w33_7*v7+w33_8*v8+w33_9*v9+w33_10*v10+w33_11*v11+w33_12*v12+w33_13*v13+w33_14*v14+w33_15*v15+w33_16*v16
        tl.store(out_ptr + out_base + 3*out_stride_h + 3*out_stride_w + out_c, acc.to(out_ptr.dtype.element_ty), mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TritonLocalAttenderI4(nn.Module):
    """
    Cell-tiled fused Triton LocalAttender for n=17, I=4.

    Each low-res cell maps to 4×4=16 high-res output pixels. All 16 pixels
    share the same 17 neighbor value loads → 16× data reuse.

    Automatically selects kernel:
    - Small H + small C: cell-tiled NCHW (low overhead, all channels in one block)
    - Large H or large C: NHWC streaming (channel loop, precomputed softmax)
    """

    def __init__(self, num_connected: int = 17, streaming_threshold: int = 32):
        super().__init__()
        assert num_connected == 17, "I4 kernel is specialized for n=17"
        self.num_connected = num_connected
        self.pn = 2
        self.streaming_threshold = streaming_threshold

    @torch.compiler.disable
    def forward(self, att: torch.Tensor, x: torch.Tensor, hardmax: bool = False) -> torch.Tensor:
        """
        Args:
            att: [B, 17, H_out, W_out] attention logits (pre-softmax), H_out=H*4
            x:   [B, C, H, W] value features
        Returns:
            [B, C, H_out, W_out] attended features
        """
        B, C, H, W = x.shape
        D = att.shape[1]
        H_out, W_out = att.shape[2], att.shape[3]
        I = H_out // H
        assert D == 17 and I == 4, f"Expected D=17 I=4, got D={D} I={I}"

        x_pad = F.pad(x, [self.pn, self.pn, self.pn, self.pn], mode='replicate')
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
        att = att.contiguous()

        num_cells = B * H * W

        if H >= self.streaming_threshold and C > 8:
            # NHWC streaming kernel
            x_pad_nhwc = x_pad.contiguous(memory_format=torch.channels_last)
            out_nhwc = torch.empty(B, C, H_out, W_out, device=x.device, dtype=x.dtype,
                                   memory_format=torch.channels_last)

            _v3_nhwc_streaming_i4_kernel[(num_cells,)](
                att, x_pad_nhwc, out_nhwc,
                C, H, W, H_out, W_out, H_pad, W_pad,
                att.stride(0), att.stride(1), att.stride(2), att.stride(3),
                x_pad_nhwc.stride(0), x_pad_nhwc.stride(2), x_pad_nhwc.stride(3), x_pad_nhwc.stride(1),
                out_nhwc.stride(0), out_nhwc.stride(2), out_nhwc.stride(3), out_nhwc.stride(1),
                num_cells,
                USE_I64_OFFSETS=out_nhwc.numel() > _INT32_MAX,
            )
            return out_nhwc.contiguous()
        else:
            # Cell-tiled NCHW kernel
            x_pad = x_pad.contiguous()
            out = torch.empty(B, C, H_out, W_out, device=x.device, dtype=x.dtype)

            BLOCK_C = min(128, triton.next_power_of_2(max(C, 4)))
            grid = (triton.cdiv(C, BLOCK_C), num_cells)

            _v3_cell_tiled_i4_kernel[grid](
                att, x_pad, out,
                C, H, W, H_out, W_out, H_pad, W_pad,
                att.stride(0), att.stride(1), att.stride(2), att.stride(3),
                x_pad.stride(0), x_pad.stride(1), x_pad.stride(2), x_pad.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                USE_I64_OFFSETS=out.numel() > _INT32_MAX,
                BLOCK_C=BLOCK_C,
            )
            return out


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness test + Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def test_and_bench():
    import sys
    import statistics
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    from lifted import LocalAttender
    from triton_local_attender import TritonLocalAttender

    device = torch.device('cuda')
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True

    # ─── Correctness ────────────────────────────────────────────────────────
    print("=" * 80)
    print("CORRECTNESS TESTS — I=4 cell-tiled kernel")
    print("=" * 80)
    all_pass = True

    for C in [4, 16, 64, 128, 384]:
        for H in [8, 16, 32, 64]:
            B, D, I = 1, 17, 4
            H_out, W_out = H * I, H * I
            x = torch.randn(B, C, H, H, device=device, dtype=dtype)
            att = torch.randn(B, D, H_out, W_out, device=device, dtype=dtype)

            orig = LocalAttender(num_connected=D).to(device).eval()
            fused = TritonLocalAttenderI4().to(device).eval()

            with torch.no_grad():
                o1 = orig(att, x).float()
                o2 = fused(att, x).float()

            max_err = (o1 - o2).abs().max().item()
            rel_err = ((o1 - o2).abs() / (o1.abs() + 1e-8)).mean().item()
            ok = max_err < 0.05
            all_pass = all_pass and ok
            print(f"  [{'PASS' if ok else 'FAIL'}] C={C:3d} H={H:3d}: max_err={max_err:.6f} rel_err={rel_err:.8f}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    # ─── Benchmarks ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BENCHMARK: C=4 (SD1.5 VAE), n=17, I=4, bf16, B=1")
    print("=" * 80)

    def bench_fn(fn, warmup=30, iters=200):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(); e.record(); torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        return statistics.median(times), torch.cuda.max_memory_allocated() / (1024**2)

    print(f"  {'H':>3}  {'eager':>9}  {'v1':>9}  {'I4 cell':>9}  {'vs_eager':>9}  {'vs_v1':>9}")
    print("  " + "-" * 55)

    for H in [8, 16, 32, 64, 128]:
        B, C, D, I = 1, 4, 17, 4
        x = torch.randn(B, C, H, H, device=device, dtype=dtype)
        att = torch.randn(B, D, H * I, H * I, device=device, dtype=dtype)

        orig = LocalAttender(num_connected=D).to(device).eval()
        t_eager, _ = bench_fn(lambda: orig(att, x))

        v1 = TritonLocalAttender(num_connected=D).to(device).eval()
        t_v1, _ = bench_fn(lambda: v1(att, x))

        fused = TritonLocalAttenderI4().to(device).eval()
        t_i4, _ = bench_fn(lambda: fused(att, x))

        print(f"  {H:3d}  {t_eager:7.3f}ms  {t_v1:7.3f}ms  {t_i4:7.3f}ms  {t_eager/t_i4:7.2f}x  {t_v1/t_i4:7.2f}x")

        del orig, v1, fused
        torch.cuda.empty_cache()

    # Also test with larger C (other configs that might use I=4)
    print(f"\n  --- Larger C ---")
    print(f"  {'C':>3} {'H':>3}  {'eager':>9}  {'I4 cell':>9}  {'vs_eager':>9}")
    print("  " + "-" * 40)

    for C in [64, 128, 384]:
        for H in [16, 32]:
            B, D, I = 1, 17, 4
            x = torch.randn(B, C, H, H, device=device, dtype=dtype)
            att = torch.randn(B, D, H * I, H * I, device=device, dtype=dtype)

            orig = LocalAttender(num_connected=D).to(device).eval()
            t_eager, _ = bench_fn(lambda: orig(att, x))

            fused = TritonLocalAttenderI4().to(device).eval()
            t_i4, _ = bench_fn(lambda: fused(att, x))

            print(f"  {C:3d} {H:3d}  {t_eager:7.3f}ms  {t_i4:7.3f}ms  {t_eager/t_i4:7.2f}x")

            del orig, fused
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    test_and_bench()
