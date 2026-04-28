"""Tests for newly added operators and template formatting fixes."""
import pytest

from ascend_transpiler.analyzer.ast_analyzer import analyze_file
from ascend_transpiler.codegen.generator import CodeGenerator
from ascend_transpiler.ir.operator_ir import OpKind
from ascend_transpiler.tiling.calculator import TilingCalculator


def _src(body: str) -> str:
    return f"from ascend_transpiler import *\n{body}"


def _generate(src: str) -> dict[str, str]:
    irs = analyze_file(f"from ascend_transpiler import *\n{src}")
    calc = TilingCalculator()
    gen = CodeGenerator()
    results = {}
    for ir in irs:
        ir.tiling = calc.calculate(ir)
        results.update(gen.generate(ir))
    return results


# ---------------------------------------------------------------------------
# New unary ops — analyzer
# ---------------------------------------------------------------------------

def test_sin_op_kind():
    src = _src("""
@ascend_op
def sin_op(x: Tensor[float32]) -> Tensor[float32]:
    return sin(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.SIN


def test_cos_op_kind():
    src = _src("""
@ascend_op
def cos_op(x: Tensor[float32]) -> Tensor[float32]:
    return cos(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.COS


def test_gelu_op_kind():
    src = _src("""
@ascend_op
def gelu_op(x: Tensor[float16]) -> Tensor[float16]:
    return gelu(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.GELU


def test_silu_op_kind():
    src = _src("""
@ascend_op
def silu_op(x: Tensor[float16]) -> Tensor[float16]:
    return silu(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.SILU


def test_reciprocal_op_kind():
    src = _src("""
@ascend_op
def recip_op(x: Tensor[float32]) -> Tensor[float32]:
    return reciprocal(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.RECIPROCAL


def test_floor_ceil_round_op_kinds():
    for fname, expected in [("floor", OpKind.FLOOR), ("ceil", OpKind.CEIL), ("round", OpKind.ROUND)]:
        src = _src(f"""
@ascend_op
def op(x: Tensor[float32]) -> Tensor[float32]:
    return {fname}(x)
""")
        [ir] = analyze_file(src)
        assert ir.nodes[0].kind == expected, f"Expected {expected} for {fname}"


def test_sign_op_kind():
    src = _src("""
@ascend_op
def sign_op(x: Tensor[float32]) -> Tensor[float32]:
    return sign(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.SIGN


# ---------------------------------------------------------------------------
# leaky_relu — alpha parameter extraction
# ---------------------------------------------------------------------------

def test_leaky_relu_default_alpha():
    src = _src("""
@ascend_op
def leaky_op(x: Tensor[float32]) -> Tensor[float32]:
    return leaky_relu(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.LEAKY_RELU
    assert ir.nodes[0].attrs["alpha"] == 0.01


def test_leaky_relu_positional_alpha():
    src = _src("""
@ascend_op
def leaky_op(x: Tensor[float32]) -> Tensor[float32]:
    return leaky_relu(x, 0.2)
""")
    [ir] = analyze_file(src)
    assert abs(ir.nodes[0].attrs["alpha"] - 0.2) < 1e-9


def test_leaky_relu_keyword_alpha():
    src = _src("""
@ascend_op
def leaky_op(x: Tensor[float32]) -> Tensor[float32]:
    return leaky_relu(x, alpha=0.1)
""")
    [ir] = analyze_file(src)
    assert abs(ir.nodes[0].attrs["alpha"] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# elementwise binary ops — maximum / minimum
# ---------------------------------------------------------------------------

def test_maximum_op_kind():
    src = _src("""
@ascend_op
def max_op(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return maximum(x, y)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.MAXIMUM
    assert set(ir.nodes[0].inputs) == {"x", "y"}


def test_minimum_op_kind():
    src = _src("""
@ascend_op
def min_op(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return minimum(x, y)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.MINIMUM


# ---------------------------------------------------------------------------
# Code generation — new ops emit correct API names
# ---------------------------------------------------------------------------

def test_gelu_codegen_emits_gelu_api():
    src = """
@ascend_op
def gelu_op(x: Tensor[float16]) -> Tensor[float16]:
    return gelu(x)
"""
    files = _generate(src)
    cpp = files["gelu_op.cpp"]
    assert "Gelu(zLocal, xLocal, this->tileLength);" in cpp


def test_sin_codegen_emits_sin_api():
    src = """
@ascend_op
def sin_op(x: Tensor[float32]) -> Tensor[float32]:
    return sin(x)
"""
    files = _generate(src)
    assert "Sin(zLocal, xLocal, this->tileLength);" in files["sin_op.cpp"]


def test_leaky_relu_codegen_emits_leakyrelu_with_alpha():
    src = """
@ascend_op
def leaky_op(x: Tensor[float32]) -> Tensor[float32]:
    return leaky_relu(x, alpha=0.1)
"""
    files = _generate(src)
    cpp = files["leaky_op.cpp"]
    assert "LeakyRelu(" in cpp
    assert "0.1" in cpp


def test_maximum_codegen_emits_max_api():
    src = """
@ascend_op
def max_op(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return maximum(x, y)
"""
    files = _generate(src)
    assert "Max(zLocal, xLocal, yLocal, this->tileLength);" in files["max_op.cpp"]


def test_minimum_codegen_emits_min_api():
    src = """
@ascend_op
def min_op(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return minimum(x, y)
"""
    files = _generate(src)
    assert "Min(zLocal, xLocal, yLocal, this->tileLength);" in files["min_op.cpp"]


def test_clamp_fused_uses_maximum_then_minimum():
    src = """
@ascend_op
def clamp_op(x: Tensor[float16], lo: Tensor[float16], hi: Tensor[float16]) -> Tensor[float16]:
    clamped_lo = maximum(x, lo)
    return minimum(clamped_lo, hi)
"""
    files = _generate(src)
    cpp = files["clamp_op.cpp"]
    assert "Max(" in cpp
    assert "Min(" in cpp


# ---------------------------------------------------------------------------
# Template formatting — Init parameters must be on separate lines
# ---------------------------------------------------------------------------

def test_init_params_on_separate_lines_single_input():
    src = """
@ascend_op
def relu_op(x: Tensor[float16]) -> Tensor[float16]:
    return relu(x)
"""
    files = _generate(src)
    cpp = files["relu_op.cpp"]
    # The tensor GM_ADDR params (x, z) must each be on their own line.
    # The fixed "GM_ADDR workspace, GM_ADDR tiling)" line is intentionally shared.
    lines = cpp.splitlines()
    tensor_gm_lines = [
        l for l in lines
        if "GM_ADDR" in l and "void" not in l and "extern" not in l
        and "workspace" not in l and "tiling" not in l
    ]
    for line in tensor_gm_lines:
        assert line.count("GM_ADDR") == 1, f"Multiple tensor GM_ADDR on one line: {line!r}"
    # Verify no Init inputs are collapsed: "Init(" must be followed by a newline before GM_ADDR x
    assert "Init(\n" in cpp or "Init(\r\n" in cpp, "Init( should open on its own line"


def test_init_params_on_separate_lines_two_inputs():
    src = """
@ascend_op
def add_op(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
"""
    files = _generate(src)
    cpp = files["add_op.cpp"]
    lines = cpp.splitlines()
    tensor_gm_lines = [
        l for l in lines
        if "GM_ADDR" in l and "void" not in l and "extern" not in l
        and "workspace" not in l and "tiling" not in l
    ]
    for line in tensor_gm_lines:
        assert line.count("GM_ADDR") == 1, f"Multiple tensor GM_ADDR on one line: {line!r}"
    # Specifically verify x and y are on different lines
    x_line = next((l for l in lines if "GM_ADDR x" in l), None)
    y_line = next((l for l in lines if "GM_ADDR y" in l), None)
    assert x_line is not None
    assert y_line is not None
    assert x_line != y_line, "x and y GM_ADDR should be on separate lines"


def test_no_template_leakage_new_ops():
    src = """
@ascend_op
def gelu_op(x: Tensor[float16]) -> Tensor[float16]:
    return gelu(x)
"""
    files = _generate(src)
    for content in files.values():
        assert "{{" not in content
        assert "}}" not in content


# ---------------------------------------------------------------------------
# Phase 0: API naming fixes — log→Ln, maximum→Max, minimum→Min
# ---------------------------------------------------------------------------

def test_log_codegen_emits_ln_api():
    src = """
@ascend_op
def log_op(x: Tensor[float32]) -> Tensor[float32]:
    return log(x)
"""
    files = _generate(src)
    assert "Ln(zLocal, xLocal, this->tileLength);" in files["log_op.cpp"]


# ---------------------------------------------------------------------------
# Phase 1: New unary ops — rsqrt, logical_not
# ---------------------------------------------------------------------------

def test_rsqrt_op_kind():
    src = _src("""
@ascend_op
def rsqrt_op(x: Tensor[float32]) -> Tensor[float32]:
    return rsqrt(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.RSQRT


def test_rsqrt_codegen_emits_rsqrt_api():
    src = """
@ascend_op
def rsqrt_op(x: Tensor[float32]) -> Tensor[float32]:
    return rsqrt(x)
"""
    files = _generate(src)
    assert "Rsqrt(zLocal, xLocal, this->tileLength);" in files["rsqrt_op.cpp"]


def test_logical_not_op_kind():
    src = _src("""
@ascend_op
def not_op(x: Tensor[uint8]) -> Tensor[uint8]:
    return logical_not(x)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.LOGICAL_NOT


def test_logical_not_codegen_emits_not_api():
    src = """
@ascend_op
def not_op(x: Tensor[uint8]) -> Tensor[uint8]:
    return logical_not(x)
"""
    files = _generate(src)
    assert "Not(zLocal, xLocal, this->tileLength);" in files["not_op.cpp"]


# ---------------------------------------------------------------------------
# Phase 2: Logical binary ops — logical_and, logical_or, & | operators
# ---------------------------------------------------------------------------

def test_logical_and_op_kind():
    src = _src("""
@ascend_op
def and_op(x: Tensor[uint8], y: Tensor[uint8]) -> Tensor[uint8]:
    return logical_and(x, y)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.LOGICAL_AND


def test_logical_or_op_kind():
    src = _src("""
@ascend_op
def or_op(x: Tensor[uint8], y: Tensor[uint8]) -> Tensor[uint8]:
    return logical_or(x, y)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.LOGICAL_OR


def test_logical_and_codegen_emits_and_api():
    src = """
@ascend_op
def and_op(x: Tensor[uint8], y: Tensor[uint8]) -> Tensor[uint8]:
    return logical_and(x, y)
"""
    files = _generate(src)
    assert "And(zLocal, xLocal, yLocal, this->tileLength);" in files["and_op.cpp"]


def test_logical_or_codegen_emits_or_api():
    src = """
@ascend_op
def or_op(x: Tensor[uint8], y: Tensor[uint8]) -> Tensor[uint8]:
    return logical_or(x, y)
"""
    files = _generate(src)
    assert "Or(zLocal, xLocal, yLocal, this->tileLength);" in files["or_op.cpp"]


def test_bitand_operator_maps_to_logical_and():
    src = _src("""
@ascend_op
def and_op(x: Tensor[uint8], y: Tensor[uint8]) -> Tensor[uint8]:
    return x & y
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.LOGICAL_AND


def test_bitor_operator_maps_to_logical_or():
    src = _src("""
@ascend_op
def or_op(x: Tensor[uint8], y: Tensor[uint8]) -> Tensor[uint8]:
    return x | y
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.LOGICAL_OR


# ---------------------------------------------------------------------------
# Phase 3: Scalar shifts — shift_left, shift_right
# ---------------------------------------------------------------------------

def test_shift_left_op_kind():
    src = _src("""
@ascend_op
def shl_op(x: Tensor[int32]) -> Tensor[int32]:
    return shift_left(x, 2)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.SHIFT_LEFT
    assert ir.nodes[0].attrs["scalar_value"] == 2


def test_shift_right_op_kind():
    src = _src("""
@ascend_op
def shr_op(x: Tensor[int32]) -> Tensor[int32]:
    return shift_right(x, 3)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.SHIFT_RIGHT
    assert ir.nodes[0].attrs["scalar_value"] == 3


def test_shift_left_codegen_emits_shiftleft_api():
    src = """
@ascend_op
def shl_op(x: Tensor[int32]) -> Tensor[int32]:
    return shift_left(x, 2)
"""
    files = _generate(src)
    assert "ShiftLeft(zLocal, xLocal, (int32_t)2, this->tileLength);" in files["shl_op.cpp"]


# ---------------------------------------------------------------------------
# Phase 4: Axpy
# ---------------------------------------------------------------------------

def test_axpy_op_kind():
    src = _src("""
@ascend_op
def axpy_op(x: Tensor[float32], y: Tensor[float32]) -> Tensor[float32]:
    return axpy(x, y, 0.5)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.AXPY
    assert abs(ir.nodes[0].attrs["alpha"] - 0.5) < 1e-9


def test_axpy_codegen_emits_axpy_api():
    src = """
@ascend_op
def axpy_op(x: Tensor[float32], y: Tensor[float32]) -> Tensor[float32]:
    return axpy(x, y, 2.0)
"""
    files = _generate(src)
    cpp = files["axpy_op.cpp"]
    assert "Axpy(" in cpp
    assert "2.0" in cpp


# ---------------------------------------------------------------------------
# Phase 5: Scalar clamp (Maxs + Mins)
# ---------------------------------------------------------------------------

def test_maxs_op_kind():
    src = _src("""
@ascend_op
def maxs_op(x: Tensor[float32]) -> Tensor[float32]:
    return maxs(x, 0.0)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.MAXS
    assert ir.nodes[0].attrs["scalar_value"] == 0.0


def test_mins_op_kind():
    src = _src("""
@ascend_op
def mins_op(x: Tensor[float32]) -> Tensor[float32]:
    return mins(x, 1.0)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.MINS


def test_maxs_codegen_emits_maxs_api():
    src = """
@ascend_op
def maxs_op(x: Tensor[float32]) -> Tensor[float32]:
    return maxs(x, 0.0)
"""
    files = _generate(src)
    assert "Maxs(zLocal, xLocal, (float)0.0, this->tileLength);" in files["maxs_op.cpp"]


def test_clamp_expands_to_maxs_and_mins():
    src = _src("""
@ascend_op
def clamp_op(x: Tensor[float32]) -> Tensor[float32]:
    return clamp(x, 0.0, 1.0)
""")
    [ir] = analyze_file(src)
    assert len(ir.nodes) == 2
    assert ir.nodes[0].kind == OpKind.MAXS
    assert ir.nodes[0].attrs["scalar_value"] == 0.0
    assert ir.nodes[1].kind == OpKind.MINS
    assert ir.nodes[1].attrs["scalar_value"] == 1.0


def test_clamp_codegen_emits_maxs_and_mins():
    src = """
@ascend_op
def clamp_op(x: Tensor[float32]) -> Tensor[float32]:
    return clamp(x, 0.0, 1.0)
"""
    files = _generate(src)
    cpp = files["clamp_op.cpp"]
    assert "Maxs(" in cpp
    assert "Mins(" in cpp


# ---------------------------------------------------------------------------
# Phase 6: Duplicate
# ---------------------------------------------------------------------------

def test_duplicate_op_kind():
    src = _src("""
@ascend_op
def dup_op(x: Tensor[float32]) -> Tensor[float32]:
    return duplicate(x, 0.0)
""")
    [ir] = analyze_file(src)
    assert ir.nodes[0].kind == OpKind.DUPLICATE
    assert ir.nodes[0].attrs["fill_value"] == 0.0


def test_duplicate_codegen_emits_duplicate_api():
    src = """
@ascend_op
def dup_op(x: Tensor[float32]) -> Tensor[float32]:
    return duplicate(x, 1.0)
"""
    files = _generate(src)
    assert "Duplicate(zLocal, (float)1.0, this->tileLength);" in files["dup_op.cpp"]
