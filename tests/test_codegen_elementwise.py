"""Tests for elementwise code generation."""
import pytest

from ascend_transpiler.analyzer.ast_analyzer import analyze_file
from ascend_transpiler.codegen.generator import CodeGenerator
from ascend_transpiler.tiling.calculator import TilingCalculator


def _generate(src: str) -> dict[str, str]:
    irs = analyze_file(f"from ascend_transpiler import *\n{src}")
    calc = TilingCalculator()
    gen = CodeGenerator()
    results = {}
    for ir in irs:
        ir.tiling = calc.calculate(ir)
        results.update(gen.generate(ir))
    return results


def test_add_custom_cpp_required_strings():
    src = """
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
"""
    files = _generate(src)
    cpp = files["add_custom.cpp"]
    assert '#include "kernel_operator.h"' in cpp
    assert "using namespace AscendC;" in cpp
    assert "class AddCustom {" in cpp
    assert "__aicore__ inline void Init(" in cpp
    assert "DataCopy(xLocal, xGm[progress * this->tileLength]" in cpp
    assert "Add(zLocal, xLocal, yLocal, this->tileLength);" in cpp
    assert 'extern "C" __global__ __aicore__ void add_custom(' in cpp


def test_add_custom_no_template_leakage():
    src = """
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
"""
    files = _generate(src)
    for content in files.values():
        assert "{{" not in content
        assert "}}" not in content


def test_relu_uses_relu_api():
    src = """
@ascend_op
def relu_custom(x: Tensor[float32]) -> Tensor[float32]:
    return relu(x)
"""
    files = _generate(src)
    cpp = files["relu_custom.cpp"]
    assert "Relu(zLocal, xLocal, this->tileLength);" in cpp
    assert "float" in cpp  # float32 → float


def test_scalar_mul_uses_muls():
    src = """
@ascend_op
def scale(x: Tensor[float16]) -> Tensor[float16]:
    return x * 2.0
"""
    files = _generate(src)
    cpp = files["scale.cpp"]
    assert "Muls(" in cpp
    assert "2.0" in cpp


def test_fused_op_emits_intermediate():
    src = """
@ascend_op
def fused(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    tmp = x + y
    return relu(tmp)
"""
    files = _generate(src)
    cpp = files["fused.cpp"]
    assert "Add(" in cpp
    assert "Relu(" in cpp


def test_tiling_header_generated():
    src = """
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
"""
    files = _generate(src)
    assert "add_custom_tiling.h" in files
    assert "add_custom_tiling.cpp" in files
    h = files["add_custom_tiling.h"]
    assert "BEGIN_TILING_DATA_DEF" in h
    assert "totalLength" in h
    assert "tileLength" in h


def test_reduction_tiling_header_has_extra_fields():
    src = """
@ascend_op
def reduce_sum_custom(x: Tensor[float32]) -> Tensor[float32]:
    return reduce_sum(x, axis=-1)
"""
    files = _generate(src)
    h = files["reduce_sum_custom_tiling.h"]
    assert "reduceLen" in h
    assert "numRows" in h
