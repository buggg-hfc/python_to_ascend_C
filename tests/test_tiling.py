"""Tests for tiling calculator."""
import pytest

from ascend_transpiler.analyzer.ast_analyzer import analyze_file
from ascend_transpiler.exceptions import TilingError
from ascend_transpiler.tiling.calculator import TilingCalculator


def _transpile(src: str):
    irs = analyze_file(f"from ascend_transpiler import *\n{src}")
    calc = TilingCalculator()
    for ir in irs:
        ir.tiling = calc.calculate(ir)
    return irs


def test_float16_tile_size_positive():
    src = """
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
"""
    [ir] = _transpile(src)
    assert ir.tiling.block_size > 0


def test_float32_tile_smaller_than_float16():
    src_f16 = """
@ascend_op
def add16(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
"""
    src_f32 = """
@ascend_op
def add32(x: Tensor[float32], y: Tensor[float32]) -> Tensor[float32]:
    return x + y
"""
    [ir16] = _transpile(src_f16)
    [ir32] = _transpile(src_f32)
    assert ir16.tiling.block_size > ir32.tiling.block_size


def test_tile_size_float16_alignment():
    """tile_size must be a multiple of 16 (32 bytes / 2 bytes per float16)."""
    src = """
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
"""
    [ir] = _transpile(src)
    assert ir.tiling.block_size % 16 == 0


def test_tile_size_float32_alignment():
    """tile_size must be a multiple of 8 (32 bytes / 4 bytes per float32)."""
    src = """
@ascend_op
def add32(x: Tensor[float32], y: Tensor[float32]) -> Tensor[float32]:
    return x + y
"""
    [ir] = _transpile(src)
    assert ir.tiling.block_size % 8 == 0


def test_user_tile_size_override():
    src = """
@ascend_op(tile_size=512)
def scale(x: Tensor[float16]) -> Tensor[float16]:
    return x * 2.0
"""
    [ir] = _transpile(src)
    assert ir.tiling.block_size == 512


def test_unaligned_user_tile_raises():
    from ascend_transpiler.ir.operator_ir import TilingConfig
    from ascend_transpiler.tiling.calculator import TilingCalculator

    src = """
@ascend_op(tile_size=513)
def scale(x: Tensor[float16]) -> Tensor[float16]:
    return x * 2.0
"""
    irs = analyze_file(f"from ascend_transpiler import *\n{src}")
    calc = TilingCalculator()
    with pytest.raises(TilingError):
        calc.calculate(irs[0])
