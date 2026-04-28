"""Tests for the AST analyzer."""
import pytest

from ascend_transpiler.analyzer.ast_analyzer import analyze_file
from ascend_transpiler.exceptions import (
    MissingAnnotationError,
    UnsupportedOperationError,
    UnsupportedDTypeError,
)
from ascend_transpiler.ir.operator_ir import OpCategory, OpKind


def _src(body: str) -> str:
    return f"from ascend_transpiler import *\n{body}"


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

def test_add_op_structure():
    src = _src("""
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
""")
    [ir] = analyze_file(src)
    assert ir.name == "add_custom"
    assert len(ir.inputs) == 2
    assert ir.inputs[0].name == "x"
    assert ir.inputs[0].dtype == "float16"
    assert ir.inputs[1].name == "y"
    assert len(ir.outputs) == 1
    assert ir.outputs[0].dtype == "float16"
    assert len(ir.nodes) == 1
    assert ir.nodes[0].kind == OpKind.ADD
    assert ir.category == OpCategory.ELEMENTWISE


def test_relu_op():
    src = _src("""
@ascend_op
def relu_custom(x: Tensor[float32]) -> Tensor[float32]:
    return relu(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.RELU
    assert ir.category == OpCategory.ELEMENTWISE


def test_fused_add_relu_node_order():
    src = _src("""
@ascend_op
def fused(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    tmp = x + y
    return relu(tmp)
""")
    [ir] = analyze_file(src)
    assert len(ir.nodes) == 2
    assert ir.nodes[0].kind == OpKind.ADD
    assert ir.nodes[1].kind == OpKind.RELU


def test_scalar_multiply_emits_muls():
    src = _src("""
@ascend_op
def scale(x: Tensor[float16]) -> Tensor[float16]:
    return x * 2.0
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.MULS
    assert ir.nodes[0].attrs["scalar_value"] == 2.0


def test_reduction_category():
    src = _src("""
@ascend_op
def reduce_sum_custom(x: Tensor[float32]) -> Tensor[float32]:
    return reduce_sum(x, axis=-1)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.REDUCE_SUM
    assert ir.nodes[0].attrs["axis"] == -1
    assert ir.category == OpCategory.REDUCTION


def test_matmul_category():
    src = _src("""
@ascend_op
def mm(a: Tensor[float16], b: Tensor[float16]) -> Tensor[float32]:
    return matmul(a, b)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.MATMUL
    assert ir.category == OpCategory.MATMUL


def test_multiple_ops_in_file():
    src = _src("""
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y

@ascend_op
def relu_custom(x: Tensor[float16]) -> Tensor[float16]:
    return relu(x)
""")
    irs = analyze_file(src)
    assert len(irs) == 2
    assert {ir.name for ir in irs} == {"add_custom", "relu_custom"}


# ---------------------------------------------------------------------------
# Tile config extraction
# ---------------------------------------------------------------------------

def test_tile_size_from_decorator():
    src = _src("""
@ascend_op(tile_size=512)
def scale(x: Tensor[float16]) -> Tensor[float16]:
    return x * 2.0
""")
    [ir] = analyze_file(src)
    assert ir.tiling.block_size == 512


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def test_var_types_propagated():
    src = _src("""
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
""")
    [ir] = analyze_file(src)
    assert "x" in ir.var_types
    assert ir.var_types["x"] == "float16"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_annotation_raises():
    src = _src("""
@ascend_op
def bad_op(x) -> Tensor[float16]:
    return x
""")
    with pytest.raises(MissingAnnotationError):
        analyze_file(src)


def test_unsupported_dtype_raises():
    src = _src("""
@ascend_op
def bad_op(x: Tensor[complex128]) -> Tensor[complex128]:
    return x
""")
    with pytest.raises(UnsupportedDTypeError):
        analyze_file(src)


def test_unknown_function_raises():
    src = _src("""
@ascend_op
def bad_op(x: Tensor[float16]) -> Tensor[float16]:
    return unknown_func(x)
""")
    with pytest.raises(UnsupportedOperationError):
        analyze_file(src)
