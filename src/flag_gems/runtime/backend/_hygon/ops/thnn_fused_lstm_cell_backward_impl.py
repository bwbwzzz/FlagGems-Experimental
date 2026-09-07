# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _cell_bwd_kblock_kernel(
    ws_ptr,
    ghy_ptr,
    gcy_ptr,
    cx_ptr,
    cy_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    grad_bias_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per column block of the (4*H) gate axis; loops over the
    # (small) batch rows inside the program. All math in fp32; the per-row
    # gradients are rounded to the pointer dtype before being accumulated
    # into the bias so fp16/bf16 sums stay bit-exact vs torch.sum(dim=0).
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < 4 * H
    g = k // H
    h = k - g * H
    dt = grad_gates_ptr.dtype.element_ty
    bsum = tl.zeros([BLOCK], dtype=tl.float32)
    for n in range(0, N):
        base = n * 4 * H
        crow = n * H
        o_g = tl.load(ws_ptr + base + 3 * H + h, mask=mask, other=0.0).to(tl.float32)
        cx_v = tl.load(cx_ptr + crow + h, mask=mask, other=0.0).to(tl.float32)
        cy_v = tl.load(cy_ptr + crow + h, mask=mask, other=0.0).to(tl.float32)
        ghy_v = tl.load(ghy_ptr + crow + h, mask=mask, other=0.0).to(tl.float32)
        gcy_v = tl.load(gcy_ptr + crow + h, mask=mask, other=0.0).to(tl.float32)
        ig_v = tl.load(ws_ptr + base + h, mask=mask, other=0.0).to(tl.float32)
        gg_v = tl.load(ws_ptr + base + 2 * H + h, mask=mask, other=0.0).to(tl.float32)
        f_v = tl.load(ws_ptr + base + H + h, mask=mask, other=0.0).to(tl.float32)
        gv = tl.where(
            g == 0,
            ig_v,
            tl.where(g == 1, f_v, tl.where(g == 2, gg_v, o_g)),
        )
        tanh_cy = tl.extra.libdevice.tanh(cy_v)
        dcy = gcy_v + ghy_v * o_g * (1.0 - tanh_cy * tanh_cy)
        grad = tl.where(
            g == 0,
            dcy * gg_v * gv * (1.0 - gv),
            tl.where(
                g == 1,
                dcy * cx_v * gv * (1.0 - gv),
                tl.where(
                    g == 2,
                    dcy * ig_v * (1.0 - gv * gv),
                    ghy_v * tanh_cy * gv * (1.0 - gv),
                ),
            ),
        )
        grad_r = grad.to(dt)
        tl.store(grad_gates_ptr + base + k, grad_r, mask=mask)
        tl.store(grad_cx_ptr + crow + h, dcy * f_v, mask=mask & (g == 0))
        bsum += grad_r.to(tl.float32)
    tl.store(grad_bias_ptr + k, bsum, mask=mask)


@triton.jit
def _cell_bwd_btile_kernel(
    ws_ptr,
    ghy_ptr,
    gcy_ptr,
    cx_ptr,
    cy_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    grad_bias_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    # One program per column block; batch rows are processed in vectorized
    # ROWS-row tiles inside the program. Single launch for the whole op.
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < 4 * H
    g = k // H
    h = k - g * H
    dt = grad_gates_ptr.dtype.element_ty
    bsum = tl.zeros([BLOCK], dtype=tl.float32)
    g_eq = g[None, :]
    for c0 in range(0, N, ROWS):
        ns = c0 + tl.arange(0, ROWS)
        n_mask = ns < N
        row_off = ns[:, None] * 4 * H + k[None, :]
        c_off = ns[:, None] * H + h[None, :]
        m2 = n_mask[:, None] & mask[None, :]
        o_g = tl.load(
            ws_ptr + ns[:, None] * 4 * H + 3 * H + h[None, :],
            mask=m2,
            other=0.0,
        ).to(tl.float32)
        cx_v = tl.load(cx_ptr + c_off, mask=m2, other=0.0).to(tl.float32)
        cy_v = tl.load(cy_ptr + c_off, mask=m2, other=0.0).to(tl.float32)
        ghy_v = tl.load(ghy_ptr + c_off, mask=m2, other=0.0).to(tl.float32)
        gcy_v = tl.load(gcy_ptr + c_off, mask=m2, other=0.0).to(tl.float32)
        ig_v = tl.load(
            ws_ptr + ns[:, None] * 4 * H + h[None, :],
            mask=m2,
            other=0.0,
        ).to(tl.float32)
        gg_v = tl.load(
            ws_ptr + ns[:, None] * 4 * H + 2 * H + h[None, :],
            mask=m2,
            other=0.0,
        ).to(tl.float32)
        f_v = tl.load(
            ws_ptr + ns[:, None] * 4 * H + H + h[None, :],
            mask=m2,
            other=0.0,
        ).to(tl.float32)
        gv = tl.where(
            g_eq == 0,
            ig_v,
            tl.where(g_eq == 1, f_v, tl.where(g_eq == 2, gg_v, o_g)),
        )
        tanh_cy = tl.extra.libdevice.tanh(cy_v)
        dcy = gcy_v + ghy_v * o_g * (1.0 - tanh_cy * tanh_cy)
        grad = tl.where(
            g_eq == 0,
            dcy * gg_v * gv * (1.0 - gv),
            tl.where(
                g_eq == 1,
                dcy * cx_v * gv * (1.0 - gv),
                tl.where(
                    g_eq == 2,
                    dcy * ig_v * (1.0 - gv * gv),
                    ghy_v * tanh_cy * gv * (1.0 - gv),
                ),
            ),
        )
        grad_r = grad.to(dt)
        tl.store(grad_gates_ptr + row_off, grad_r, mask=m2)
        tl.store(grad_cx_ptr + c_off, dcy * f_v, mask=m2 & (g_eq == 0))
        bsum += tl.sum(grad_r.to(tl.float32), axis=0)
    tl.store(grad_bias_ptr + k, bsum, mask=mask)


@triton.jit
def _cell_bwd_kernel(
    ws_ptr,
    ghy_ptr,
    gcy_ptr,
    cx_ptr,
    cy_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # One program per batch row; BLOCK_H covers the hidden dimension.
    # All math in fp32; stores round to the pointer dtype (matches ATen).
    pid = tl.program_id(0)
    n = pid
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    row = n * 4 * H
    crow = n * H

    i_gate = tl.load(ws_ptr + row + offs, mask=mask, other=0.0).to(tl.float32)
    f_gate = tl.load(ws_ptr + row + H + offs, mask=mask, other=0.0).to(tl.float32)
    g_gate = tl.load(ws_ptr + row + 2 * H + offs, mask=mask, other=0.0).to(tl.float32)
    o_gate = tl.load(ws_ptr + row + 3 * H + offs, mask=mask, other=0.0).to(tl.float32)
    cx_v = tl.load(cx_ptr + crow + offs, mask=mask, other=0.0).to(tl.float32)
    cy_v = tl.load(cy_ptr + crow + offs, mask=mask, other=0.0).to(tl.float32)
    ghy_v = tl.load(ghy_ptr + crow + offs, mask=mask, other=0.0).to(tl.float32)
    gcy_v = tl.load(gcy_ptr + crow + offs, mask=mask, other=0.0).to(tl.float32)

    tanh_cy = tl.extra.libdevice.tanh(cy_v)
    dcy = gcy_v + ghy_v * o_gate * (1.0 - tanh_cy * tanh_cy)

    grad_i = dcy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = dcy * cx_v * f_gate * (1.0 - f_gate)
    grad_g = dcy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = ghy_v * tanh_cy * o_gate * (1.0 - o_gate)

    tl.store(grad_gates_ptr + row + offs, grad_i, mask=mask)
    tl.store(grad_gates_ptr + row + H + offs, grad_f, mask=mask)
    tl.store(grad_gates_ptr + row + 2 * H + offs, grad_g, mask=mask)
    tl.store(grad_gates_ptr + row + 3 * H + offs, grad_o, mask=mask)
    tl.store(grad_cx_ptr + crow + offs, dcy * f_gate, mask=mask)


@triton.jit
def _bias_tree_kernel(
    grad_gates_ptr,
    grad_bias_ptr,
    N,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Sum the stored (dtype-rounded) gate gradients over the batch in fp32.
    # For fp16/bf16 the per-row values are exactly representable in fp32, so
    # any accumulation order reproduces torch.sum(dim=0) bit-for-bit.
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < 4 * H
    ns = tl.arange(0, ROWS)
    offs = ns[:, None] * 4 * H + k[None, :]
    m2 = mask[None, :] & (ns[:, None] < N)
    tile = tl.load(grad_gates_ptr + offs, mask=m2, other=0.0).to(tl.float32)
    acc = tl.sum(tile, axis=0)
    tl.store(grad_bias_ptr + k, acc, mask=mask)


def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    logger.debug("GEMS_HYGON THNN_FUSED_LSTM_CELL_BACKWARD_IMPL")
    N, gates_dim = workspace.shape
    H = gates_dim // 4

    grad_input_gates = torch.empty_like(workspace)
    grad_cx = torch.empty_like(cx)
    grad_biases = torch.empty(4 * H, dtype=workspace.dtype, device=workspace.device)

    if N <= 4:
        # Small batch: one fused kernel, rows looped inside each column block.
        _cell_bwd_kblock_kernel[(triton.cdiv(4 * H, 64),)](
            workspace,
            grad_hy,
            grad_cy,
            cx,
            cy,
            grad_input_gates,
            grad_cx,
            grad_biases,
            N=N,
            H=H,
            BLOCK=64,
            num_warps=1,
        )
    elif N <= 16:
        # Medium/large batch: vectorized 8-row tiles inside one kernel.
        _cell_bwd_btile_kernel[(triton.cdiv(4 * H, 64),)](
            workspace,
            grad_hy,
            grad_cy,
            cx,
            cy,
            grad_input_gates,
            grad_cx,
            grad_biases,
            N=N,
            H=H,
            BLOCK=64,
            ROWS=8,
            num_warps=8,
        )
    else:
        # Very large batch: per-row kernel plus tree reduce over the batch axis.
        _cell_bwd_kernel[(N,)](
            workspace,
            grad_hy,
            grad_cy,
            cx,
            cy,
            grad_input_gates,
            grad_cx,
            H=H,
            BLOCK_H=64,
            num_warps=2,
        )
        _bias_tree_kernel[(triton.cdiv(4 * H, 64),)](
            grad_input_gates,
            grad_biases,
            N=N,
            H=H,
            BLOCK=64,
            ROWS=16,
            num_warps=2,
        )
    return grad_input_gates, grad_cx, grad_biases
