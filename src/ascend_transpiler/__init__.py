"""ascend_transpiler — Python-to-Ascend-C transpiler."""
from ascend_transpiler.dsl.decorators import (
    ascend_op,
    tile,
    relu,
    sqrt,
    exp,
    log,
    abs,
    tanh,
    sigmoid,
    sin,
    cos,
    floor,
    ceil,
    round,
    sign,
    reciprocal,
    gelu,
    silu,
    leaky_relu,
    rsqrt,
    logical_not,
    maximum,
    minimum,
    logical_and,
    logical_or,
    maxs,
    mins,
    shift_left,
    shift_right,
    axpy,
    duplicate,
    clamp,
    cast,
    matmul,
    reduce_sum,
    reduce_max,
    reduce_min,
    reduce_mean,
)
from ascend_transpiler.dsl.types import (
    Tensor,
    float16,
    float32,
    float64,
    bfloat16,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    mxfp8,
    mxfp4,
    hif8,
)
from ascend_transpiler.transpiler import Transpiler

__all__ = [
    # Decorators
    "ascend_op", "tile",
    # Primitives — unary
    "relu", "sqrt", "exp", "log", "abs", "tanh", "sigmoid",
    "sin", "cos", "floor", "ceil", "round", "sign", "reciprocal",
    "gelu", "silu", "leaky_relu", "rsqrt", "logical_not",
    # Primitives — binary elementwise tensor×tensor
    "maximum", "minimum", "logical_and", "logical_or",
    # Primitives — scalar binary (tensor×scalar)
    "maxs", "mins", "shift_left", "shift_right",
    # Primitives — multi-input / special
    "axpy", "duplicate", "clamp",
    # Special
    "cast", "matmul",
    "reduce_sum", "reduce_max", "reduce_min", "reduce_mean",
    # Types
    "Tensor",
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "mxfp8", "mxfp4", "hif8",
    # Programmatic API
    "Transpiler",
]
