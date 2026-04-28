"""End-to-end transpiler tests."""
import pathlib

import pytest

from ascend_transpiler.transpiler import Transpiler


def _transpile_source(src: str, tmp_path: pathlib.Path) -> dict[str, list[str]]:
    t = Transpiler()
    return t.transpile_source(f"from ascend_transpiler import *\n{src}", tmp_path)


def test_add_custom_writes_three_files(tmp_path):
    src = """
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
"""
    results = _transpile_source(src, tmp_path)
    assert "add_custom" in results
    written = results["add_custom"]
    assert len(written) == 3
    names = {pathlib.Path(p).name for p in written}
    assert names == {"add_custom.cpp", "add_custom_tiling.h", "add_custom_tiling.cpp"}


def test_all_files_are_nonempty(tmp_path):
    src = """
@ascend_op
def relu_custom(x: Tensor[float32]) -> Tensor[float32]:
    return relu(x)
"""
    results = _transpile_source(src, tmp_path)
    for path_str in results["relu_custom"]:
        content = pathlib.Path(path_str).read_text()
        assert len(content.strip()) > 0


def test_transpile_file(tmp_path):
    py_file = tmp_path / "add_custom.py"
    py_file.write_text(
        "from ascend_transpiler import *\n\n"
        "@ascend_op\n"
        "def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:\n"
        "    return x + y\n"
    )
    out_dir = tmp_path / "out"
    t = Transpiler()
    results = t.transpile_file(py_file, out_dir)
    assert "add_custom" in results
    assert out_dir.exists()


def test_multiple_ops_in_one_file(tmp_path):
    src = """
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y

@ascend_op
def relu_custom(x: Tensor[float16]) -> Tensor[float16]:
    return relu(x)
"""
    results = _transpile_source(src, tmp_path)
    assert "add_custom" in results
    assert "relu_custom" in results


def test_reduction_end_to_end(tmp_path):
    src = """
@ascend_op
def reduce_sum_custom(x: Tensor[float32]) -> Tensor[float32]:
    return reduce_sum(x, axis=-1)
"""
    results = _transpile_source(src, tmp_path)
    written = results["reduce_sum_custom"]
    cpp_path = next(p for p in written if p.endswith(".cpp") and "tiling" not in p)
    cpp = pathlib.Path(cpp_path).read_text()
    assert "ReduceSum" in cpp
