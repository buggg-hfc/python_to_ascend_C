"""Softmax — composed from primitives.

softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_i - max(x)))

Note: This example demonstrates how to build composite operators from
primitives using the transpiler's fused-op support. The intermediate
tensors (shifted, exp_vals) stay in UB without GM round-trips.

Because softmax requires a reduction (sum) followed by elementwise ops,
it needs the COMPOSITE pattern. For now, write it as two separate ops:
  1. reduce_max to find the row maximum (reduction kernel)
  2. softmax_normalize for exp(x - max) / sum(exp(x - max)) (elementwise + reduction)

Alternatively, use the single-op shortcut if your CANN version provides a
built-in SoftmaxV2 API — just map it as a unary op in your own extension.
"""
from ascend_transpiler import ascend_op, Tensor, float32, exp, reduce_sum, reduce_max


@ascend_op
def exp_and_sum(x: Tensor[float32]) -> Tensor[float32]:
    """Exp + row-wise sum: demonstrates fused exp followed by reduction."""
    return reduce_sum(exp(x), axis=-1)


@ascend_op
def row_max(x: Tensor[float32]) -> Tensor[float32]:
    """Row-wise max: z[i] = max(x[i, :])."""
    return reduce_max(x, axis=-1)
