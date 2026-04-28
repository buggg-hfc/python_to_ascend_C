"""Main code generator — selects the right strategy by OpCategory."""
from __future__ import annotations

import jinja2

from ascend_transpiler.ir.operator_ir import OpCategory, OperatorIR

from .elementwise import ElementwiseCodegen
from .matmul import MatMulCodegen
from .reduction import ReductionCodegen


class CodeGenerator:
    def __init__(self):
        self._env = jinja2.Environment(
            loader=jinja2.PackageLoader("ascend_transpiler", "codegen/templates"),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def generate(self, ir: OperatorIR) -> dict[str, str]:
        """Return {filename: content} for all files to be written."""
        strategy = self._select_strategy(ir)
        return strategy.render(ir)

    def _select_strategy(self, ir: OperatorIR):
        if ir.category == OpCategory.MATMUL:
            return MatMulCodegen(self._env)
        if ir.category == OpCategory.REDUCTION:
            return ReductionCodegen(self._env)
        return ElementwiseCodegen(self._env)
