"""Fused add + relu: z = relu(x + y).

The intermediate variable `tmp` stays entirely in UB (no GM round-trip),
demonstrating the transpiler's intermediate-variable optimization.
"""
from ascend_transpiler import ascend_op, Tensor, float16, relu


@ascend_op
def fused_add_relu(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    tmp = x + y
    return relu(tmp)
