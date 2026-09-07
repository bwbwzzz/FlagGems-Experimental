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
def bn_eval_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    C,
    S,
    numel,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = (offs // S) % C
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    scale = tl.rsqrt(v + eps)
    y = (x - m) * scale * w + b
    tl.store(out_ptr + offs, y, mask=mask)


def _batch_norm_no_update(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS_HYGON BATCH_NORM_NO_UPDATE")
    if weight is None:
        weight = torch.ones(
            input.shape[1] if input.dim() >= 2 else 1,
            dtype=input.dtype,
            device=input.device,
        )
    if bias is None:
        bias = torch.zeros_like(weight)
    if running_mean is None:
        running_mean = torch.zeros_like(weight)
    if running_var is None:
        running_var = torch.ones_like(weight)

    rank = input.dim()
    C = input.shape[1] if rank >= 2 else 1
    S = 1
    for s in input.shape[2:]:
        S *= s
    numel = input.numel()

    out = torch.empty_like(input)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    bn_eval_kernel[grid](
        input,
        weight,
        bias,
        running_mean,
        running_var,
        out,
        C,
        S,
        numel,
        float(eps),
        BLOCK=BLOCK,
    )

    # aten _batch_norm_no_update schema: (out, save_mean, save_var, reserved)
    save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
    save_var = torch.empty((0,), dtype=input.dtype, device=input.device)
    reserved = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return out, save_mean, save_var, reserved
