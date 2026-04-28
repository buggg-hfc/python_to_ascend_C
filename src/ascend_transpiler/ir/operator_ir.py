from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class OpKind(Enum):
    # Elementwise binary (tensor × tensor)
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOORDIV = auto()
    MOD = auto()
    POW = auto()
    # Elementwise binary (tensor × scalar) — AscendC uses separate API: Adds, Muls …
    ADDS = auto()
    SUBS = auto()
    MULS = auto()
    DIVS = auto()
    # Elementwise unary
    RELU = auto()
    SQRT = auto()
    EXP = auto()
    LOG = auto()
    ABS = auto()
    NEG = auto()
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
    # Elementwise binary (tensor × tensor, result shape = max of shapes)
    MAXIMUM = auto()      # element-wise max (distinct from ReduceMax)
    MINIMUM = auto()      # element-wise min (distinct from ReduceMin)
    # Type conversion
    CAST = auto()   # attrs: {'target_dtype': str}
    # Reduction
    REDUCE_SUM = auto()   # attrs: {'axis': int, 'keepdims': bool}
    REDUCE_MAX = auto()
    REDUCE_MIN = auto()
    REDUCE_MEAN = auto()
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
