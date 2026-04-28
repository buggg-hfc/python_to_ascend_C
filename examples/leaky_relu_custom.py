"""Leaky ReLU: z = x if x > 0 else alpha * x.

The alpha parameter is passed as a scalar to the AscendC LeakyRelu API.
"""
from ascend_transpiler import ascend_op, Tensor, float32, leaky_relu


@ascend_op
def leaky_relu_custom(x: Tensor[float32]) -> Tensor[float32]:
    return leaky_relu(x, alpha=0.1)
