"""Reduction (Vector core) code generator."""
from __future__ import annotations

import jinja2

from ascend_transpiler.ir.operator_ir import OpKind, OperatorIR
from ascend_transpiler.ops.mappings import DTYPE_TO_CPP, REDUCE_KIND_TO_API


def _to_class_name(func_name: str) -> str:
    return "".join(part.capitalize() for part in func_name.split("_"))


def _get_reduce_api(ir: OperatorIR) -> str:
    for node in ir.nodes:
        if node.kind in REDUCE_KIND_TO_API:
            return REDUCE_KIND_TO_API[node.kind]
    return "ReduceSum"


class ReductionCodegen:
    def __init__(self, env: jinja2.Environment):
        self._env = env

    def render(self, ir: OperatorIR) -> dict[str, str]:
        class_name = _to_class_name(ir.name)
        output = ir.outputs[0]

        inputs_ctx = [
            {"name": t.name, "cpp_type": DTYPE_TO_CPP.get(t.dtype, "half")}
            for t in ir.inputs
        ]
        output_ctx = {
            "name": output.name,
            "cpp_type": DTYPE_TO_CPP.get(output.dtype, "half"),
        }

        kernel_ctx = {
            "func_name": ir.name,
            "class_name": class_name,
            "buffer_num": ir.tiling.buffer_num,
            "inputs": inputs_ctx,
            "output": output_ctx,
            "reduce_api": _get_reduce_api(ir),
        }
        tiling_ctx = {
            "func_name": ir.name,
            "class_name": class_name,
            "category": "REDUCTION",
            "block_dim": ir.tiling.block_dim,
            "tile_length": ir.tiling.block_size,
        }

        kernel_tmpl = self._env.get_template("kernel_reduction.cpp.j2")
        tiling_h_tmpl = self._env.get_template("tiling_data.h.j2")
        tiling_cpp_tmpl = self._env.get_template("tiling_func.cpp.j2")

        return {
            f"{ir.name}.cpp": kernel_tmpl.render(**kernel_ctx),
            f"{ir.name}_tiling.h": tiling_h_tmpl.render(**tiling_ctx),
            f"{ir.name}_tiling.cpp": tiling_cpp_tmpl.render(**tiling_ctx),
        }
