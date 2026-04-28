"""Elementwise relu: z = max(x, 0) (float32)."""
from ascend_transpiler import ascend_op, Tensor, float32, relu


@ascend_op
def relu_custom(x: Tensor[float32]) -> Tensor[float32]:
    return relu(x)
