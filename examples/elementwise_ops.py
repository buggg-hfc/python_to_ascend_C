"""Showcase of common elementwise operators in one file.

Each function transpiles to its own kernel .cpp file.
Run: ascend-transpiler examples/elementwise_ops.py -o out/
"""
from ascend_transpiler import (
    ascend_op, Tensor, float16, float32,
    relu, gelu, silu, sigmoid, tanh,
    sin, cos, exp, log, sqrt, reciprocal,
    floor, ceil, abs, sign,
    maximum, minimum,
)


@ascend_op
def relu_op(x: Tensor[float16]) -> Tensor[float16]:
    return relu(x)


@ascend_op
def gelu_op(x: Tensor[float16]) -> Tensor[float16]:
    return gelu(x)


@ascend_op
def silu_op(x: Tensor[float16]) -> Tensor[float16]:
    return silu(x)


@ascend_op
def sin_op(x: Tensor[float32]) -> Tensor[float32]:
    return sin(x)


@ascend_op
def cos_op(x: Tensor[float32]) -> Tensor[float32]:
    return cos(x)


@ascend_op
def reciprocal_op(x: Tensor[float32]) -> Tensor[float32]:
    return reciprocal(x)


@ascend_op
def maximum_op(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    """Element-wise maximum of two tensors."""
    return maximum(x, y)


@ascend_op
def minimum_op(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    """Element-wise minimum of two tensors."""
    return minimum(x, y)


@ascend_op
def clamp_op(x: Tensor[float16], lo: Tensor[float16], hi: Tensor[float16]) -> Tensor[float16]:
    """Clamp x to [lo, hi] using elementwise max and min."""
    clamped_lo = maximum(x, lo)
    return minimum(clamped_lo, hi)
