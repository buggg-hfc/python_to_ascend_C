"""Scalar broadcast multiply: z = x * 2.0.

One operand is a Python scalar constant — the transpiler detects this and
emits AscendC `Muls` instead of `Mul`, avoiding a dummy tensor allocation.
"""
from ascend_transpiler import ascend_op, tile, Tensor, float16


@ascend_op(tile_size=512)
def scale_custom(x: Tensor[float16]) -> Tensor[float16]:
    return x * 2.0
