"""Elementwise (Vector core) code generator."""
from __future__ import annotations

import jinja2

from ascend_transpiler.ir.operator_ir import IRNode, OpKind, OperatorIR
from ascend_transpiler.ops.mappings import (
    BINARY_OPS,
    BINOP_KIND_TO_API,
    DTYPE_TO_CPP,
    REDUCE_KIND_TO_API,
    SCALAR_BINOP_KIND_TO_API,
    SCALAR_OPS,
    UNARY_OPS,
    UNOP_KIND_TO_API,
)

_MICROSCALING_DTYPES = {"mxfp8", "mxfp4", "hif8"}


def _to_class_name(func_name: str) -> str:
    return "".join(part.capitalize() for part in func_name.split("_"))


def _find_intermediates(ir: OperatorIR) -> list[dict]:
    """Return variable info for intermediate (UB-only) tensors."""
    input_names = {t.name for t in ir.inputs}
    output_names = {t.name for t in ir.outputs}
    produced: set[str] = set()
    intermediates: list[dict] = []
    seen: set[str] = set()
    for node in ir.nodes:
        for inp in node.inputs:
            if inp in produced and inp not in input_names and inp not in seen:
                dtype = ir.var_types.get(inp, ir.inputs[0].dtype if ir.inputs else "float16")
                intermediates.append({
                    "name": inp,
                    "cpp_type": DTYPE_TO_CPP.get(dtype, "half"),
                })
                seen.add(inp)
        for out in node.outputs:
            produced.add(out)
    return intermediates


def _local_name(var: str) -> str:
    return f"{var}Local"


def _build_compute_statements(ir: OperatorIR) -> list[str]:
    """Render each IR node as one or two C++ statements for Compute()."""
    stmts: list[str] = []
    output_name = ir.outputs[0].name if ir.outputs else "z"
    input_names = {t.name for t in ir.inputs}

    for node in ir.nodes:
        out_var = node.outputs[0] if node.outputs else output_name
        out_local = _local_name(out_var)
        tile_len = "this->tileLength"

        if node.kind in BINARY_OPS:
            api = BINOP_KIND_TO_API[node.kind]
            lhs = _local_name(node.inputs[0])
            rhs = _local_name(node.inputs[1])
            stmts.append(f"{api}({out_local}, {lhs}, {rhs}, {tile_len});")

        elif node.kind in SCALAR_OPS:
            api = SCALAR_BINOP_KIND_TO_API[node.kind]
            src = _local_name(node.inputs[0])
            scalar = node.attrs.get("scalar_value", 0)
            dtype = ir.var_types.get(node.inputs[0], "float16")
            cpp_type = DTYPE_TO_CPP.get(dtype, "half")
            stmts.append(f"{api}({out_local}, {src}, ({cpp_type}){scalar}, {tile_len});")

        elif node.kind in UNARY_OPS:
            api = UNOP_KIND_TO_API[node.kind]
            src = _local_name(node.inputs[0])
            stmts.append(f"{api}({out_local}, {src}, {tile_len});")

        elif node.kind == OpKind.CAST:
            src = _local_name(node.inputs[0])
            target_dtype = node.attrs.get("target_dtype", "float32")
            cpp_type = DTYPE_TO_CPP.get(target_dtype, "float")
            stmts.append(f"Cast({out_local}, {src}, RoundMode::CAST_NONE, {tile_len});")

        else:
            stmts.append(f"// Unsupported op: {node.kind}")

    return stmts


class ElementwiseCodegen:
    def __init__(self, env: jinja2.Environment):
        self._env = env

    def render(self, ir: OperatorIR) -> dict[str, str]:
        class_name = _to_class_name(ir.name)
        output = ir.outputs[0]
        dtype_set = {t.dtype for t in ir.inputs} | {output.dtype}
        has_microscaling = bool(dtype_set & _MICROSCALING_DTYPES)

        inputs_ctx = [
            {"name": t.name, "cpp_type": DTYPE_TO_CPP.get(t.dtype, "half")}
            for t in ir.inputs
        ]
        output_ctx = {
            "name": output.name,
            "cpp_type": DTYPE_TO_CPP.get(output.dtype, "half"),
        }
        intermediates = _find_intermediates(ir)
        compute_stmts = _build_compute_statements(ir)

        kernel_ctx = {
            "func_name": ir.name,
            "class_name": class_name,
            "buffer_num": ir.tiling.buffer_num,
            "inputs": inputs_ctx,
            "output": output_ctx,
            "intermediates": intermediates,
            "compute_statements": compute_stmts,
            "has_microscaling": has_microscaling,
        }
        tiling_ctx = {
            "func_name": ir.name,
            "class_name": class_name,
            "category": "ELEMENTWISE",
            "block_dim": ir.tiling.block_dim,
            "tile_length": ir.tiling.block_size,
        }

        kernel_tmpl = self._env.get_template("kernel_elementwise.cpp.j2")
        tiling_h_tmpl = self._env.get_template("tiling_data.h.j2")
        tiling_cpp_tmpl = self._env.get_template("tiling_func.cpp.j2")

        return {
            f"{ir.name}.cpp": kernel_tmpl.render(**kernel_ctx),
            f"{ir.name}_tiling.h": tiling_h_tmpl.render(**tiling_ctx),
            f"{ir.name}_tiling.cpp": tiling_cpp_tmpl.render(**tiling_ctx),
        }
