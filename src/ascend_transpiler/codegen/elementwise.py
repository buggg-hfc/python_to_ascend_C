"""Elementwise (Vector core) code generator."""
from __future__ import annotations

import jinja2

from ascend_transpiler.ir.operator_ir import IRNode, OpKind, OperatorIR
from ascend_transpiler.ops.mappings import (
    AXPY_OPS,
    BINARY_OPS,
    BINOP_KIND_TO_API,
    COMPARE_MODE_TO_CONST,
    COMPARE_OPS,
    DTYPE_TO_CPP,
    DUPLICATE_OPS,
    ELEMENTWISE_BINARY_KIND_TO_API,
    ELEMENTWISE_BINARY_OPS,
    INPLACE_TERNARY_KIND_TO_API,
    INPLACE_TERNARY_OPS,
    PARAMETERIZED_UNARY_OPS,
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
    """Render each IR node as one or more C++ statements for Compute()."""
    stmts: list[str] = []
    output_name = ir.outputs[0].name if ir.outputs else "z"
    primary_dtype = ir.inputs[0].dtype if ir.inputs else "float16"

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
            dtype = ir.var_types.get(node.inputs[0], primary_dtype)
            cpp_type = DTYPE_TO_CPP.get(dtype, "half")
            stmts.append(f"{api}({out_local}, {src}, ({cpp_type}){scalar}, {tile_len});")

        elif node.kind in PARAMETERIZED_UNARY_OPS:
            api = UNOP_KIND_TO_API[node.kind]
            src = _local_name(node.inputs[0])
            alpha = node.attrs.get("alpha", 0.01)
            dtype = ir.var_types.get(node.inputs[0], primary_dtype)
            cpp_type = DTYPE_TO_CPP.get(dtype, "half")
            stmts.append(f"{api}({out_local}, {src}, ({cpp_type}){alpha}, {tile_len});")

        elif node.kind == OpKind.NEG:
            # Neg is not in the AscendC basic API; expand to Muls(dst, src, -1, len)
            src = _local_name(node.inputs[0])
            dtype = ir.var_types.get(node.inputs[0], primary_dtype)
            cpp_type = DTYPE_TO_CPP.get(dtype, "half")
            stmts.append(f"Muls({out_local}, {src}, ({cpp_type})-1, {tile_len});")

        elif node.kind in UNARY_OPS:
            api = UNOP_KIND_TO_API[node.kind]
            src = _local_name(node.inputs[0])
            stmts.append(f"{api}({out_local}, {src}, {tile_len});")

        elif node.kind in ELEMENTWISE_BINARY_OPS:
            api = ELEMENTWISE_BINARY_KIND_TO_API[node.kind]
            lhs = _local_name(node.inputs[0])
            rhs = _local_name(node.inputs[1])
            stmts.append(f"{api}({out_local}, {lhs}, {rhs}, {tile_len});")

        elif node.kind == OpKind.CAST:
            src = _local_name(node.inputs[0])
            stmts.append(f"Cast({out_local}, {src}, RoundMode::CAST_NONE, {tile_len});")

        elif node.kind == OpKind.MUL_CAST:
            lhs = _local_name(node.inputs[0])
            rhs = _local_name(node.inputs[1])
            stmts.append(f"MulCast({out_local}, {lhs}, {rhs}, {tile_len});")

        elif node.kind in AXPY_OPS:
            src = _local_name(node.inputs[0])
            dst = _local_name(node.inputs[1])
            alpha = node.attrs.get("alpha", 1.0)
            dtype = ir.var_types.get(node.inputs[0], primary_dtype)
            cpp_type = DTYPE_TO_CPP.get(dtype, "half")
            stmts.append(f"Axpy({dst}, {src}, ({cpp_type}){alpha}, {tile_len});")

        elif node.kind in INPLACE_TERNARY_OPS:
            # MulAddDst / FusedMulAdd / MulAddRelu: dst is inputs[2] (in-place accumulator).
            # We copy acc → out first, then run the in-place op on out.
            api = INPLACE_TERNARY_KIND_TO_API[node.kind]
            src0 = _local_name(node.inputs[0])
            src1 = _local_name(node.inputs[1])
            acc  = _local_name(node.inputs[2])
            dtype = ir.var_types.get(node.inputs[2], primary_dtype)
            cpp_type = DTYPE_TO_CPP.get(dtype, "half")
            # Copy acc into out (Adds with 0 is the idiomatic UB-to-UB copy)
            stmts.append(f"Adds({out_local}, {acc}, ({cpp_type})0, {tile_len});")
            stmts.append(f"{api}({out_local}, {src0}, {src1}, {tile_len});")

        elif node.kind == OpKind.COMPARE:
            lhs = _local_name(node.inputs[0])
            rhs = _local_name(node.inputs[1])
            mode = node.attrs.get("mode", "eq")
            mode_const = COMPARE_MODE_TO_CONST.get(mode, "CMPMODE_EQ")
            stmts.append(f"Compare({out_local}, {lhs}, {rhs}, {mode_const}, {tile_len});")

        elif node.kind == OpKind.COMPARES:
            src = _local_name(node.inputs[0])
            scalar = node.attrs.get("scalar_value", 0)
            mode = node.attrs.get("mode", "eq")
            mode_const = COMPARE_MODE_TO_CONST.get(mode, "CMPMODE_EQ")
            dtype = ir.var_types.get(node.inputs[0], primary_dtype)
            cpp_type = DTYPE_TO_CPP.get(dtype, "half")
            stmts.append(
                f"Compares({out_local}, {src}, ({cpp_type}){scalar}, {mode_const}, {tile_len});"
            )

        elif node.kind == OpKind.SELECT:
            src0 = _local_name(node.inputs[0])
            src1 = _local_name(node.inputs[1])
            mask = _local_name(node.inputs[2])
            stmts.append(f"Select({out_local}, {src0}, {src1}, {mask}, {tile_len});")

        elif node.kind in DUPLICATE_OPS:
            fill = node.attrs.get("fill_value", 0.0)
            dtype = ir.var_types.get(out_var, primary_dtype)
            cpp_type = DTYPE_TO_CPP.get(dtype, "half")
            stmts.append(f"Duplicate({out_local}, ({cpp_type}){fill}, {tile_len});")

        elif node.kind == OpKind.CREATE_VEC_INDEX:
            start = node.attrs.get("start", 0)
            stmts.append(f"CreateVecIndex({out_local}, ({DTYPE_TO_CPP.get('int32','int32_t')}){start}, {tile_len});")

        else:
            from ascend_transpiler.exceptions import UnsupportedOperationError
            raise UnsupportedOperationError(str(node.kind), node.lineno)

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
