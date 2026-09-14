import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4


def _block_cfg(elem_size):
    # 128-bit vectorized accesses per thread: fp16/bf16 -> 8 elems/thread,
    # fp32 -> 4 elems/thread, wider types -> 1 elem/thread scalar.
    if elem_size <= 2:
        return 4096, 8
    if elem_size == 4:
        return 2048, 8
    return 1024, 4


@triton.jit
def _process_block(x_ptr, found_inf_ptr, inv_scale_ptr, start, n, BLOCK: tl.constexpr):
    offs = start * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    scale = tl.load(inv_scale_ptr)
    bad = (x != x) | (tl.abs(x) == float("inf"))
    any_bad = tl.max(bad.to(tl.int32))
    if any_bad > 0:
        tl.store(found_inf_ptr, 1.0)
    tl.store(x_ptr + offs, x * scale, mask=mask)


@triton.jit
def _unscale_check_kernel(
    x_ptr,
    found_inf_ptr,
    inv_scale_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    _process_block(x_ptr, found_inf_ptr, inv_scale_ptr, pid, n_elements, BLOCK)


@triton.jit
def _unscale2_kernel(
    x0,
    x1,
    found_inf_ptr,
    inv_scale_ptr,
    n0,
    n1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    if pid < b0:
        _process_block(x0, found_inf_ptr, inv_scale_ptr, pid, n0, BLOCK)
    else:
        _process_block(x1, found_inf_ptr, inv_scale_ptr, pid - b0, n1, BLOCK)


def run(tensors, found_inf, inv_scale):
    if len(tensors) == 0:
        return
    unique = []
    seen = set()
    for t in tensors:
        p = t.data_ptr()
        if p not in seen:
            seen.add(p)
            unique.append(t)
    if len(unique) == 2 and unique[0].element_size() == unique[1].element_size():
        t0, t1 = unique
        n0, n1 = t0.numel(), t1.numel()
        BLOCK, WARPS = _block_cfg(t0.element_size())
        grid = (triton.cdiv(n0, BLOCK) + triton.cdiv(n1, BLOCK),)
        if grid[0] == 0:
            return
        _unscale2_kernel[grid](
            t0,
            t1,
            found_inf,
            inv_scale,
            n0,
            n1,
            BLOCK=BLOCK,
            num_warps=WARPS,
        )
        return
    for t in unique:
        n = t.numel()
        if n == 0:
            continue
        BLOCK, WARPS = _block_cfg(t.element_size())
        grid = (triton.cdiv(n, BLOCK),)
        _unscale_check_kernel[grid](
            t,
            found_inf,
            inv_scale,
            n,
            BLOCK=BLOCK,
            num_warps=WARPS,
        )


# Alias for FlagGems import convention
amp_foreach_non_finite_check_and_unscale_ = run
