"""GELU activation: z = gelu(x).

Uses the Ascend C built-in Gelu API directly (available in CANN >= 7.0).
"""
from ascend_transpiler import ascend_op, Tensor, float16, gelu


@ascend_op
def gelu_custom(x: Tensor[float16]) -> Tensor[float16]:
    return gelu(x)
