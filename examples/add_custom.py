"""Elementwise add: z = x + y (float16)."""
from ascend_transpiler import ascend_op, Tensor, float16


@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
