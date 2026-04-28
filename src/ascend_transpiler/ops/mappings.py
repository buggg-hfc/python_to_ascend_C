"""All mapping tables between Python DSL and Ascend C APIs."""
from __future__ import annotations

import ast

from ascend_transpiler.ir.operator_ir import OpKind

# ---------------------------------------------------------------------------
# dtype: Python DSL name → C++ type used in templates
# ---------------------------------------------------------------------------
DTYPE_TO_CPP: dict[str, str] = {
    "float16": "half",
    "float32": "float",
    "float64": "double",
    "bfloat16": "bfloat16_t",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    # Ascend-specific microscaling / hiFloat formats
    "mxfp8": "fp8e4m3_t",   # requires Ascend 910C+
    "mxfp4": "fp4e2m1_t",   # requires Ascend 910C+; 64-elem alignment
    "hif8": "hif8_t",        # HiFloat8, Ascend-specific
}

SUPPORTED_DTYPES: set[str] = set(DTYPE_TO_CPP)

# dtype element size in bytes (used by tiling calculator)
DTYPE_ITEMSIZE: dict[str, int] = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "bfloat16": 2,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "mxfp8": 1,
    "mxfp4": 1,  # packed 2-per-byte; treat as 1 for size calc
    "hif8": 1,
}

# ---------------------------------------------------------------------------
# AST BinOp node → OpKind (tensor × tensor)
# ---------------------------------------------------------------------------
BINOP_AST_TO_KIND: dict[type, OpKind] = {
    ast.Add: OpKind.ADD,
    ast.Sub: OpKind.SUB,
    ast.Mult: OpKind.MUL,
    ast.Div: OpKind.DIV,
    ast.FloorDiv: OpKind.FLOORDIV,
    ast.Mod: OpKind.MOD,
    ast.Pow: OpKind.POW,
}

# AST BinOp node → OpKind (tensor × scalar)
BINOP_AST_TO_SCALAR_KIND: dict[type, OpKind] = {
    ast.Add: OpKind.ADDS,
    ast.Sub: OpKind.SUBS,
    ast.Mult: OpKind.MULS,
    ast.Div: OpKind.DIVS,
}

# ---------------------------------------------------------------------------
# Call name → OpKind
# ---------------------------------------------------------------------------
CALL_NAME_TO_KIND: dict[str, OpKind] = {
    "relu": OpKind.RELU,
    "sqrt": OpKind.SQRT,
    "exp": OpKind.EXP,
    "log": OpKind.LOG,
    "abs": OpKind.ABS,
    "tanh": OpKind.TANH,
    "sigmoid": OpKind.SIGMOID,
    "sin": OpKind.SIN,
    "cos": OpKind.COS,
    "floor": OpKind.FLOOR,
    "ceil": OpKind.CEIL,
    "round": OpKind.ROUND,
    "sign": OpKind.SIGN,
    "reciprocal": OpKind.RECIPROCAL,
    "gelu": OpKind.GELU,
    "silu": OpKind.SILU,
    "leaky_relu": OpKind.LEAKY_RELU,
    "maximum": OpKind.MAXIMUM,
    "minimum": OpKind.MINIMUM,
    "cast": OpKind.CAST,
    "matmul": OpKind.MATMUL,
    "reduce_sum": OpKind.REDUCE_SUM,
    "reduce_max": OpKind.REDUCE_MAX,
    "reduce_min": OpKind.REDUCE_MIN,
    "reduce_mean": OpKind.REDUCE_MEAN,
    # common aliases
    "sum": OpKind.REDUCE_SUM,
    "mean": OpKind.REDUCE_MEAN,
}

# ---------------------------------------------------------------------------
# OpKind → AscendC API name
# ---------------------------------------------------------------------------
BINOP_KIND_TO_API: dict[OpKind, str] = {
    OpKind.ADD: "Add",
    OpKind.SUB: "Sub",
    OpKind.MUL: "Mul",
    OpKind.DIV: "Div",
    OpKind.FLOORDIV: "Div",  # closest; real floor division needs Cast post
    OpKind.MOD: "Mod",
    OpKind.POW: "Pow",
}

SCALAR_BINOP_KIND_TO_API: dict[OpKind, str] = {
    OpKind.ADDS: "Adds",
    OpKind.SUBS: "Subs",
    OpKind.MULS: "Muls",
    OpKind.DIVS: "Divs",
}

UNOP_KIND_TO_API: dict[OpKind, str] = {
    OpKind.RELU: "Relu",
    OpKind.SQRT: "Sqrt",
    OpKind.EXP: "Exp",
    OpKind.LOG: "Log",
    OpKind.ABS: "Abs",
    OpKind.NEG: "Neg",
    OpKind.TANH: "Tanh",
    OpKind.SIGMOID: "Sigmoid",
    OpKind.SIN: "Sin",
    OpKind.COS: "Cos",
    OpKind.FLOOR: "Floor",
    OpKind.CEIL: "Ceil",
    OpKind.ROUND: "Round",
    OpKind.SIGN: "Sign",
    OpKind.RECIPROCAL: "Reciprocal",
    OpKind.GELU: "Gelu",
    OpKind.SILU: "Silu",
    OpKind.LEAKY_RELU: "LeakyRelu",
}

BINOP_WITH_PARAM_KIND_TO_API: dict[OpKind, str] = {
    OpKind.LEAKY_RELU: "LeakyRelu",  # LeakyRelu(dst, src, alpha, len)
}

ELEMENTWISE_BINARY_KIND_TO_API: dict[OpKind, str] = {
    OpKind.MAXIMUM: "Maximum",
    OpKind.MINIMUM: "Minimum",
}

REDUCE_KIND_TO_API: dict[OpKind, str] = {
    OpKind.REDUCE_SUM: "ReduceSum",
    OpKind.REDUCE_MAX: "ReduceMax",
    OpKind.REDUCE_MIN: "ReduceMin",
    OpKind.REDUCE_MEAN: "ReduceMean",
}

# Sets for category classification
REDUCTION_OPS: frozenset[OpKind] = frozenset(REDUCE_KIND_TO_API)
MATMUL_OPS: frozenset[OpKind] = frozenset({OpKind.MATMUL})
SCALAR_OPS: frozenset[OpKind] = frozenset(SCALAR_BINOP_KIND_TO_API)
UNARY_OPS: frozenset[OpKind] = frozenset(UNOP_KIND_TO_API)
BINARY_OPS: frozenset[OpKind] = frozenset(BINOP_KIND_TO_API)
ELEMENTWISE_BINARY_OPS: frozenset[OpKind] = frozenset(ELEMENTWISE_BINARY_KIND_TO_API)
# Ops with an extra scalar parameter (alpha, etc.)
PARAMETERIZED_UNARY_OPS: frozenset[OpKind] = frozenset({OpKind.LEAKY_RELU})
