"""Transpiler — end-to-end pipeline: Python source → Ascend C files."""
from __future__ import annotations

import pathlib

from ascend_transpiler.analyzer.ast_analyzer import analyze_file
from ascend_transpiler.codegen.generator import CodeGenerator
from ascend_transpiler.ir.operator_ir import OperatorIR
from ascend_transpiler.tiling.calculator import TilingCalculator


class Transpiler:
    def __init__(self, ub_size_kb: int = 256, default_block_dim: int = 8):
        self._calculator = TilingCalculator(ub_size_kb, default_block_dim)
        self._generator = CodeGenerator()

    def transpile_source(self, source: str, output_dir: pathlib.Path) -> dict[str, list[str]]:
        """Transpile a Python source string; return {op_name: [written file paths]}."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ir_list = analyze_file(source)
        results: dict[str, list[str]] = {}
        for ir in ir_list:
            ir.tiling = self._calculator.calculate(ir)
            files = self._generator.generate(ir)
            written: list[str] = []
            for filename, content in files.items():
                out_path = output_dir / filename
                out_path.write_text(content, encoding="utf-8")
                written.append(str(out_path))
            results[ir.name] = written
        return results

    def transpile_file(self, path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, list[str]]:
        """Transpile a .py file; return {op_name: [written file paths]}."""
        source = path.read_text(encoding="utf-8")
        return self.transpile_source(source, output_dir)
