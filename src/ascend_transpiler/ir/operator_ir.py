from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class OpKind(Enum):
    # Elementwise binary (tensor × tensor) — basic AscendC vector API
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    # NOTE: FLOORDIV / MOD / POW are not in the AscendC basic vector API.
    # The analyzer still parses them so user gets a clear UnsupportedOperationError
    # from codegen rather than a silent mis-emit.
    FLOORDIV = auto()
    MOD = auto()
    POW = auto()
    # Fused binary ops (tensor × tensor → tensor)
    ADD_RELU = auto()     # AddRelu: relu(x + y)
    SUB_RELU = auto()     # SubRelu: relu(x - y)
    MUL_CAST = auto()     # MulCast: Mul + Cast  attrs: {'target_dtype': str}
    # Elementwise binary (tensor × scalar)
    ADDS = auto()
    SUBS = auto()
    MULS = auto()
    DIVS = auto()
    MAXS = auto()
    MINS = auto()
    ANDS = auto()         # Ands: per-element AND with scalar
    ORS = auto()          # Ors: per-element OR with scalar
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    # 3-input in-place fused ops (dst, src0, src1) — dst is accumulator AND output
    AXPY = auto()         # Axpy: dst += alpha * src  attrs: {'alpha': float}
    MUL_ADD_DST = auto()  # MulAddDst: dst = src0*src1 + dst  inputs: [src0, src1, acc]
    FUSED_MUL_ADD = auto()# FusedMulAdd: dst = dst*src0 + src1  inputs: [acc, src0, src1]
    MUL_ADD_RELU = auto() # MulAddRelu: relu(dst*src0+src1)  inputs: [acc, src0, src1]
    # Elementwise unary
    RELU = auto()
    SQRT = auto()
    EXP = auto()
    LOG = auto()
    ABS = auto()
    NEG = auto()          # Expanded in codegen to Muls(dst, src, -1, len)
    TANH = auto()
    SIGMOID = auto()
    SIN = auto()
    COS = auto()
    FLOOR = auto()
    CEIL = auto()
    ROUND = auto()
    SIGN = auto()
    RECIPROCAL = auto()
    GELU = auto()
    SILU = auto()
    LEAKY_RELU = auto()   # attrs: {'alpha': float}
    RSQRT = auto()
    LOGICAL_NOT = auto()
    # Elementwise binary (tensor × tensor)
    MAXIMUM = auto()
    MINIMUM = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    # Compare and select
    COMPARE = auto()      # Compare(dst, x, y, mode, len) → bit mask  attrs: {'mode': str}
    COMPARES = auto()     # Compares(dst, x, scalar, mode, len)  attrs: {'scalar_value', 'mode'}
    SELECT = auto()       # Select(dst, src0, src1, mask, len)  inputs: [src0, src1, mask]
    # Type conversion
    CAST = auto()         # attrs: {'target_dtype': str}
    # Data fill / index
    DUPLICATE = auto()    # attrs: {'fill_value': float}
    CREATE_VEC_INDEX = auto()  # attrs: {'start': int/float}
    # Reduction
    REDUCE_SUM = auto()   # attrs: {'axis': int, 'keepdims': bool}
    REDUCE_MAX = auto()
    REDUCE_MIN = auto()
    REDUCE_MEAN = auto()  # high-level API; may not exist on all hardware
    # MatMul (cube core)
    MATMUL = auto()


class OpCategory(Enum):
    ELEMENTWISE = "ELEMENTWISE"
    REDUCTION = "REDUCTION"
    MATMUL = "MATMUL"


@dataclass
class TensorSpec:
    name: str
    dtype: str
    is_input: bool = True
    shape: tuple | None = None
    layout: str = "ND"


@dataclass
class IRNode:
    kind: OpKind
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    lineno: int | None = None


@dataclass
class TilingConfig:
    block_size: int = 256
    buffer_num: int = 2
    block_dim: int = 8


@dataclass
class OperatorIR:
    name: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    nodes: list[IRNode]
    tiling: TilingConfig = field(default_factory=TilingConfig)
    category: OpCategory = OpCategory.ELEMENTWISE
    var_types: dict[str, str] = field(default_factory=dict)
