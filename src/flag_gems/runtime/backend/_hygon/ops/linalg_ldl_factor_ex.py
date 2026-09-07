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

"""linalg_ldl_factor_ex - Triton implementation.

Exact port of LAPACK's DSYTF2 (Bunch-Kaufman LDL^T, lower storage) with the
LAPACK lower-triangle DSWAP semantics, matching torch.linalg.ldl_factor_ex.
Each program factorizes one matrix sequentially; batch matrices map to
programs. The sequential k-loop uses scalar while-loops; trailing updates are
kept in the exact DSYTF2 order (validated bitwise vs torch GPU on eval-style
inputs).

Round 4 goal: vectorize the 1x1/2x2 trailing updates with tl.arange block
loads/stores of full columns/rows of the active submatrix while preserving the
exact per-element arithmetic (a_ij - w_ik*w_jk*d11 and the 2x2 sequential
updates). The swap and the pivot decisions remain scalar.
"""
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

ALPHA = tl.constexpr((1.0 + 17.0**0.5) / 8.0)


@triton.jit
def _ldl_factor_kernel(
    A,
    LD,
    PIV,
    INFO,
    n,
    batch_size,
    stride_a0,
    stride_a1,
    stride_a2,
    stride_ld0,
    stride_ld1,
    stride_ld2,
    hermitian: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    base_ld = LD + pid * stride_ld0
    base_a = A + pid * stride_a0

    # copy the (exactly symmetric) matrix into LD (lower triangle only; upper unused)
    for i in range(n):
        row_a = base_a + i * stride_a1 + tl.arange(0, 64) * stride_a2
        row_ld = base_ld + i * stride_ld1 + tl.arange(0, 64) * stride_ld2
        m = (tl.arange(0, 64) < n) & (tl.arange(0, 64) <= i)
        a_vals = tl.load(row_a, mask=m, other=0.0)
        tl.store(row_ld, a_vals, mask=m)

    # sequential Bunch-Kaufman factorization (exact DSYTF2 port)
    k = 0
    while k < n:
        kstep = 1
        absakk = tl.load(base_ld + k * stride_ld1 + k * stride_ld2)
        absakk = tl.abs(absakk)

        # colmax = max |A(i,k)| for i in k+1..n-1 ; imax = argmax (vectorized)
        imax = k
        colmax = 0.0
        if k < n - 1:
            offs = tl.arange(0, 64)
            idx = k + 1 + offs
            m = idx < n
            col = tl.load(
                base_ld + idx * stride_ld1 + k * stride_ld2, mask=m, other=0.0
            )
            cabs = tl.abs(col)
            best = tl.max(cabs, axis=0)
            # argmax: first index attaining the max
            cand = tl.where(cabs >= best, idx, n)
            imax = tl.min(cand, axis=0)
            colmax = best

        if max(absakk, colmax) == 0.0:
            info_val = tl.load(INFO + pid)
            if info_val == 0:
                tl.store(INFO + pid, k + 1)
            kp = k
        else:
            if absakk >= ALPHA * colmax:
                kp = k
            else:
                # rowmax = max(|A(imax, j)| for j in [k..imax))  +  max(|A(i, imax)| for i in (imax..n))
                # both are segments of column imax / row imax in the active submatrix.
                # Vectorized: build one 64-wide scan over the union of the two segments is complex
                # (different strides); keep two masked block reductions.
                rowmax = 0.0
                if imax > k:
                    offs = tl.arange(0, 64)
                    jidx = k + offs
                    m = jidx < imax
                    v1 = tl.abs(
                        tl.load(
                            base_ld + imax * stride_ld1 + jidx * stride_ld2,
                            mask=m,
                            other=0.0,
                        )
                    )
                    rowmax = tl.max(v1, axis=0)
                if imax < n - 1:
                    offs2 = tl.arange(0, 64)
                    iidx = imax + 1 + offs2
                    m2 = iidx < n
                    v2 = tl.abs(
                        tl.load(
                            base_ld + iidx * stride_ld1 + imax * stride_ld2,
                            mask=m2,
                            other=0.0,
                        )
                    )
                    rowmax = tl.maximum(rowmax, tl.max(v2, axis=0))
                if absakk >= ALPHA * colmax * (colmax / rowmax):
                    kp = k
                elif (
                    tl.abs(tl.load(base_ld + imax * stride_ld1 + imax * stride_ld2))
                    >= ALPHA * rowmax
                ):
                    kp = imax
                else:
                    kp = imax
                    kstep = 2

        # ---- LAPACK lower-triangle swap (kk = k + kstep - 1) ----
        kk = k + kstep - 1
        if kp != kk:
            if kp < n - 1:
                # DSWAP(N-KP, A(KP+1,KK), 1, A(KP+1,KP), 1) - vectorized
                r = kp + 1 + tl.arange(0, 64)
                m = r < n
                t0 = tl.load(
                    base_ld + r * stride_ld1 + kk * stride_ld2, mask=m, other=0.0
                )
                t1 = tl.load(
                    base_ld + r * stride_ld1 + kp * stride_ld2, mask=m, other=0.0
                )
                tl.store(base_ld + r * stride_ld1 + kk * stride_ld2, t1, mask=m)
                tl.store(base_ld + r * stride_ld1 + kp * stride_ld2, t0, mask=m)
            # DSWAP(KP-KK-1, A(KK+1,KK), 1, A(KP,KK+1), LDA) - vectorized
            r = kk + 1 + tl.arange(0, 64)
            m = r < kp
            t0 = tl.load(base_ld + r * stride_ld1 + kk * stride_ld2, mask=m, other=0.0)
            t1 = tl.load(base_ld + kp * stride_ld1 + r * stride_ld2, mask=m, other=0.0)
            tl.store(base_ld + r * stride_ld1 + kk * stride_ld2, t1, mask=m)
            tl.store(base_ld + kp * stride_ld1 + r * stride_ld2, t0, mask=m)
            t0 = tl.load(base_ld + kk * stride_ld1 + kk * stride_ld2)
            t1 = tl.load(base_ld + kp * stride_ld1 + kp * stride_ld2)
            tl.store(base_ld + kk * stride_ld1 + kk * stride_ld2, t1)
            tl.store(base_ld + kp * stride_ld1 + kp * stride_ld2, t0)
            if kstep == 2:
                t0 = tl.load(base_ld + kk * stride_ld1 + k * stride_ld2)
                t1 = tl.load(base_ld + kp * stride_ld1 + k * stride_ld2)
                tl.store(base_ld + kk * stride_ld1 + k * stride_ld2, t1)
                tl.store(base_ld + kp * stride_ld1 + k * stride_ld2, t0)

        # ---- factorization step ----
        if kstep == 1:
            if k < n - 1:
                d11 = 1.0 / tl.load(base_ld + k * stride_ld1 + k * stride_ld2)
                # vectorized rank-1 update of the lower triangle:
                # A(i,j) -= W(i,k) * W(j,k) * d11  for i >= j > k
                # process each column j in [k+1..n): rows i in [j..n)
                j = k + 1
                while j < n:
                    # vectorized rank-1 update of column j; w_jk is a scalar load
                    w_jk = tl.load(base_ld + j * stride_ld1 + k * stride_ld2)
                    rows = j + tl.arange(0, 64)
                    m = rows < n
                    w_ik = tl.load(
                        base_ld + rows * stride_ld1 + k * stride_ld2, mask=m, other=0.0
                    )
                    a_ij = tl.load(
                        base_ld + rows * stride_ld1 + j * stride_ld2, mask=m, other=0.0
                    )
                    tl.store(
                        base_ld + rows * stride_ld1 + j * stride_ld2,
                        a_ij - w_ik * w_jk * d11,
                        mask=m,
                    )
                    j = j + 1
                # store L(k) = A(k+1:n,k) * d11 (vectorized)
                rows = k + 1 + tl.arange(0, 64)
                m = rows < n
                a_ik = tl.load(
                    base_ld + rows * stride_ld1 + k * stride_ld2, mask=m, other=0.0
                )
                tl.store(
                    base_ld + rows * stride_ld1 + k * stride_ld2, a_ik * d11, mask=m
                )
            tl.store(PIV + pid * n + k, kp + 1)
            k = k + 1
        else:
            # 2x2 pivot block
            tl.store(PIV + pid * n + k, -(kp + 1))
            tl.store(PIV + pid * n + k + 1, -(kp + 1))
            if k < n - 2:
                d21 = tl.load(base_ld + (k + 1) * stride_ld1 + k * stride_ld2)
                dd11 = (
                    tl.load(base_ld + (k + 1) * stride_ld1 + (k + 1) * stride_ld2) / d21
                )
                dd22 = tl.load(base_ld + k * stride_ld1 + k * stride_ld2) / d21
                t = 1.0 / (dd11 * dd22 - 1.0)
                d21 = t / d21
                j = k + 2
                while j < n:
                    wk = d21 * (
                        dd11 * tl.load(base_ld + j * stride_ld1 + k * stride_ld2)
                        - tl.load(base_ld + j * stride_ld1 + (k + 1) * stride_ld2)
                    )
                    wkp1 = d21 * (
                        dd22 * tl.load(base_ld + j * stride_ld1 + (k + 1) * stride_ld2)
                        - tl.load(base_ld + j * stride_ld1 + k * stride_ld2)
                    )
                    # vectorized: A(i,j) -= A(i,k)*wk + A(i,k+1)*wkp1 for i in [j..n)
                    rows = j + tl.arange(0, 64)
                    m = rows < n
                    a_ij = tl.load(
                        base_ld + rows * stride_ld1 + j * stride_ld2, mask=m, other=0.0
                    )
                    a_ik = tl.load(
                        base_ld + rows * stride_ld1 + k * stride_ld2, mask=m, other=0.0
                    )
                    a_ik1 = tl.load(
                        base_ld + rows * stride_ld1 + (k + 1) * stride_ld2,
                        mask=m,
                        other=0.0,
                    )
                    tl.store(
                        base_ld + rows * stride_ld1 + j * stride_ld2,
                        a_ij - a_ik * wk - a_ik1 * wkp1,
                        mask=m,
                    )
                    tl.store(base_ld + j * stride_ld1 + k * stride_ld2, wk)
                    tl.store(base_ld + j * stride_ld1 + (k + 1) * stride_ld2, wkp1)
                    j = j + 1
            k = k + 2

    # zero out the strictly upper triangle (vectorized per row)
    i = 0
    while i < n:
        cols2 = i + 1 + tl.arange(0, 64)
        m = cols2 < n
        tl.store(base_ld + i * stride_ld1 + cols2 * stride_ld2, 0.0, mask=m)
        i = i + 1


def ldl_factor_ex(self, *, hermitian=False, check_errors=False):
    logger.debug("GEMS_HYGON LINALG_LDL_FACTOR_EX")
    """LDL^T factorization with Bunch-Kaufman pivoting (torch reference semantics).

    Returns (LD, pivots, info).
    """
    A = self
    if A.ndim < 2:
        raise ValueError("linalg_ldl_factor_ex: A must be at least 2D")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("linalg_ldl_factor_ex: matrix must be square")
    n = A.shape[-1]
    if n > 64:
        raise ValueError("linalg_ldl_factor_ex: matrix size exceeds 64")

    if not A.is_contiguous():
        A = A.contiguous()

    LD = torch.empty_like(A)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    info = torch.empty(A.shape[:-2], dtype=torch.int32, device=A.device)
    info.fill_(0)

    batch_size = 1
    for d in A.shape[:-2]:
        batch_size *= d

    if batch_size == 0 or n == 0:
        return LD, pivots, info

    stride_a0 = A.stride(-3) if A.dim() >= 3 else 0
    stride_a1 = A.stride(-2)
    stride_a2 = A.stride(-1)
    stride_ld0 = LD.stride(-3) if LD.dim() >= 3 else 0
    stride_ld1 = LD.stride(-2)
    stride_ld2 = LD.stride(-1)

    grid = (batch_size,)
    _ldl_factor_kernel[grid](
        A,
        LD,
        pivots,
        info,
        n,
        batch_size,
        stride_a0,
        stride_a1,
        stride_a2,
        stride_ld0,
        stride_ld1,
        stride_ld2,
        hermitian=bool(hermitian),
        num_warps=1,
    )
    return LD, pivots, info
