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

import pytest
import torch

from . import base, consts


def _multigammaln_input_fn(shape, dtype, device):
    # special_multigammaln requires x > (p - 1) / 2; keep inputs in range
    def gen(shape, dtype, device):
        inp = torch.rand(shape, dtype=dtype, device=device)
        return inp + 2.0  # p=5 -> (p-1)/2 + 1 = 2.0

    yield gen(shape, dtype, device),


@pytest.mark.special_multigammaln
def test_special_multigammaln():
    bench = base.GenericBenchmark(
        op_name="special_multigammaln",
        input_fn=_multigammaln_input_fn,
        torch_op=lambda a: torch.ops.aten.special_multigammaln(a, 5),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
