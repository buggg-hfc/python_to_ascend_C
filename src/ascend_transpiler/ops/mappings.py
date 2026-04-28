"""All mapping tables between Python DSL and AscendC APIs.

API source: AscendC Memory矢量计算API (basic vector API set).
High-level math library APIs (Tanh, Sin, Gelu, etc.) are noted where used.
"""
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
    ast.FloorDiv: OpKind.FLOORDIV,   # no direct AscendC API; codegen raises error
    ast.Mod: OpKind.MOD,             # no direct AscendC API; codegen raises error
    ast.Pow: OpKind.POW,             # no direct AscendC API; codegen raises error
    ast.BitAnd: OpKind.LOGICAL_AND,
    ast.BitOr: OpKind.LOGICAL_OR,
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
    # Unary
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
    "rsqrt": OpKind.RSQRT,
    "logical_not": OpKind.LOGICAL_NOT,
    # Binary tensor×tensor
    "maximum": OpKind.MAXIMUM,
    "minimum": OpKind.MINIMUM,
    "logical_and": OpKind.LOGICAL_AND,
    "logical_or": OpKind.LOGICAL_OR,
    "add_relu": OpKind.ADD_RELU,
    "sub_relu": OpKind.SUB_RELU,
    "mul_cast": OpKind.MUL_CAST,
    # Binary tensor×scalar
    "maxs": OpKind.MAXS,
    "mins": OpKind.MINS,
    "ands": OpKind.ANDS,
    "ors": OpKind.ORS,
    "shift_left": OpKind.SHIFT_LEFT,
    "shift_right": OpKind.SHIFT_RIGHT,
    # 3-input fused
    "axpy": OpKind.AXPY,
    "mul_add_dst": OpKind.MUL_ADD_DST,
    "fused_mul_add": OpKind.FUSED_MUL_ADD,
    "mul_add_relu": OpKind.MUL_ADD_RELU,
    # Compare & select
    "compare": OpKind.COMPARE,
    "compares": OpKind.COMPARES,
    "select": OpKind.SELECT,
    # Data fill
    "duplicate": OpKind.DUPLICATE,
    "create_vec_index": OpKind.CREATE_VEC_INDEX,
    # Type conversion
    "cast": OpKind.CAST,
    # MatMul
    "matmul": OpKind.MATMUL,
    # Reductions
    "reduce_sum": OpKind.REDUCE_SUM,
    "reduce_max": OpKind.REDUCE_MAX,
    "reduce_min": OpKind.REDUCE_MIN,
    "reduce_mean": OpKind.REDUCE_MEAN,
    # Common aliases
    "sum": OpKind.REDUCE_SUM,
    "mean": OpKind.REDUCE_MEAN,
}

# ---------------------------------------------------------------------------
# OpKind → AscendC API name
# ---------------------------------------------------------------------------

# Basic binary ops confirmed in AscendC vector API table
BINOP_KIND_TO_API: dict[OpKind, str] = {
    OpKind.ADD: "Add",
    OpKind.SUB: "Sub",
    OpKind.MUL: "Mul",
    OpKind.DIV: "Div",
    OpKind.ADD_RELU: "AddRelu",
    OpKind.SUB_RELU: "SubRelu",
    # FLOORDIV / MOD / POW intentionally omitted — no AscendC basic API;
    # codegen raises UnsupportedOperationError for these.
}

SCALAR_BINOP_KIND_TO_API: dict[OpKind, str] = {
    OpKind.ADDS: "Adds",
    OpKind.SUBS: "Subs",
    OpKind.MULS: "Muls",
    OpKind.DIVS: "Divs",
    OpKind.MAXS: "Maxs",
    OpKind.MINS: "Mins",
    OpKind.ANDS: "Ands",
    OpKind.ORS: "Ors",
    OpKind.SHIFT_LEFT: "ShiftLeft",
    OpKind.SHIFT_RIGHT: "ShiftRight",
}

UNOP_KIND_TO_API: dict[OpKind, str] = {
    OpKind.RELU: "Relu",
    OpKind.SQRT: "Sqrt",
    OpKind.EXP: "Exp",
    OpKind.LOG: "Ln",          # AscendC basic API: Ln (natural log)
    OpKind.ABS: "Abs",
    # NEG not in basic API table — expanded to Muls(dst, src, -1, len) in codegen
    OpKind.TANH: "Tanh",       # high-level math library
    OpKind.SIGMOID: "Sigmoid", # high-level math library
    OpKind.SIN: "Sin",         # high-level math library
    OpKind.COS: "Cos",         # high-level math library
    OpKind.FLOOR: "Floor",     # high-level math library
    OpKind.CEIL: "Ceil",       # high-level math library
    OpKind.ROUND: "Round",     # high-level math library
    OpKind.SIGN: "Sign",       # high-level math library
    OpKind.RECIPROCAL: "Reciprocal",
    OpKind.GELU: "Gelu",       # high-level math library
    OpKind.SILU: "Silu",       # high-level math library
    OpKind.LEAKY_RELU: "LeakyRelu",
    OpKind.RSQRT: "Rsqrt",
    OpKind.LOGICAL_NOT: "Not",
}

BINOP_WITH_PARAM_KIND_TO_API: dict[OpKind, str] = {
    OpKind.LEAKY_RELU: "LeakyRelu",
}

ELEMENTWISE_BINARY_KIND_TO_API: dict[OpKind, str] = {
    OpKind.MAXIMUM: "Max",
    OpKind.MINIMUM: "Min",
    OpKind.LOGICAL_AND: "And",
    OpKind.LOGICAL_OR: "Or",
}

REDUCE_KIND_TO_API: dict[OpKind, str] = {
    OpKind.REDUCE_SUM: "ReduceSum",
    OpKind.REDUCE_MAX: "ReduceMax",
    OpKind.REDUCE_MIN: "ReduceMin",
    OpKind.REDUCE_MEAN: "ReduceMean",  # high-level; may not exist on all hardware
}

# Compare mode string → AscendC constant name
COMPARE_MODE_TO_CONST: dict[str, str] = {
    "eq": "CMPMODE_EQ",
    "ne": "CMPMODE_NE",
    "lt": "CMPMODE_LT",
    "gt": "CMPMODE_GT",
    "le": "CMPMODE_LE",
    "ge": "CMPMODE_GE",
}

# 3-input in-place fused ops: MulAddDst / FusedMulAdd / MulAddRelu
INPLACE_TERNARY_KIND_TO_API: dict[OpKind, str] = {
    OpKind.MUL_ADD_DST: "MulAddDst",
    OpKind.FUSED_MUL_ADD: "FusedMulAdd",
    OpKind.MUL_ADD_RELU: "MulAddRelu",
}

# ---------------------------------------------------------------------------
# Category sets (frozenset — auto-derived from API dicts above)
# ---------------------------------------------------------------------------
REDUCTION_OPS: frozenset[OpKind] = frozenset(REDUCE_KIND_TO_API)
MATMUL_OPS: frozenset[OpKind] = frozenset({OpKind.MATMUL})
SCALAR_OPS: frozenset[OpKind] = frozenset(SCALAR_BINOP_KIND_TO_API)
UNARY_OPS: frozenset[OpKind] = frozenset(UNOP_KIND_TO_API)
BINARY_OPS: frozenset[OpKind] = frozenset(BINOP_KIND_TO_API)
ELEMENTWISE_BINARY_OPS: frozenset[OpKind] = frozenset(ELEMENTWISE_BINARY_KIND_TO_API)
PARAMETERIZED_UNARY_OPS: frozenset[OpKind] = frozenset({OpKind.LEAKY_RELU})
AXPY_OPS: frozenset[OpKind] = frozenset({OpKind.AXPY})
INPLACE_TERNARY_OPS: frozenset[OpKind] = frozenset(INPLACE_TERNARY_KIND_TO_API)
DUPLICATE_OPS: frozenset[OpKind] = frozenset({OpKind.DUPLICATE})
COMPARE_OPS: frozenset[OpKind] = frozenset({OpKind.COMPARE, OpKind.COMPARES})
