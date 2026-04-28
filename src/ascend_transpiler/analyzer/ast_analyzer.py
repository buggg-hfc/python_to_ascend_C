"""Python AST → OperatorIR.

Walks the AST of a @ascend_op-decorated function and builds the IR graph.
Key responsibilities:
  - Parse parameter type annotations (Tensor[dtype])
  - Detect scalar broadcasts (x * 2.0 → Muls, not Mul)
  - Track intermediate variables for fused ops (stay in UB, no queue roundtrip)
  - Run a forward type-inference pass to populate var_types for all variables
  - Classify the operator as ELEMENTWISE / REDUCTION / MATMUL
"""
from __future__ import annotations

import ast
from typing import Any

from ascend_transpiler.exceptions import (
    MissingAnnotationError,
    TypeMismatchError,
    UnsupportedDTypeError,
    UnsupportedOperationError,
)
from ascend_transpiler.ir.operator_ir import (
    IRNode,
    OpCategory,
    OpKind,
    OperatorIR,
    TensorSpec,
    TilingConfig,
)
from ascend_transpiler.ops.mappings import (
    BINOP_AST_TO_KIND,
    BINOP_AST_TO_SCALAR_KIND,
    CALL_NAME_TO_KIND,
    MATMUL_OPS,
    REDUCTION_OPS,
    SUPPORTED_DTYPES,
)

_ASCEND_OP_NAMES = {"ascend_op"}


class ASTAnalyzer:
    def __init__(self, func_def: ast.FunctionDef):
        self._func = func_def
        self._nodes: list[IRNode] = []
        self._inputs: list[TensorSpec] = []
        self._outputs: list[TensorSpec] = []
        self._var_counter: int = 0
        self._input_names: set[str] = set()
        self._var_types: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self) -> OperatorIR:
        self._parse_inputs()
        tiling = self._extract_tiling_config()
        self._walk_body()
        self._infer_var_types()
        category = self._categorize()
        return OperatorIR(
            name=self._func.name,
            inputs=self._inputs,
            outputs=self._outputs,
            nodes=self._nodes,
            tiling=tiling,
            category=category,
            var_types=dict(self._var_types),
        )

    # ------------------------------------------------------------------
    # Signature parsing
    # ------------------------------------------------------------------

    def _parse_inputs(self) -> None:
        fn = self._func
        for arg in fn.args.args:
            if arg.annotation is None:
                raise MissingAnnotationError(arg.arg, fn.name)
            dtype = self._parse_dtype_annotation(arg.annotation, arg.arg)
            layout = self._parse_layout_annotation(arg.annotation)
            spec = TensorSpec(name=arg.arg, dtype=dtype, is_input=True, layout=layout)
            self._inputs.append(spec)
            self._input_names.add(arg.arg)
            self._var_types[arg.arg] = dtype

    def _parse_dtype_annotation(self, node: ast.expr, param: str) -> str:
        """Extract dtype string from Tensor[dtype] or Tensor[dtype, shape] annotation."""
        if isinstance(node, ast.Subscript):
            # Tensor[float16]  or  Tensor[float16, (M, K)]
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                # First element is dtype
                dtype_node = slice_node.elts[0]
            else:
                dtype_node = slice_node
            dtype = self._eval_dtype_node(dtype_node, param)
        elif isinstance(node, ast.Name):
            # Bare `Tensor` annotation without dtype — not allowed
            raise MissingAnnotationError(param, self._func.name)
        else:
            raise MissingAnnotationError(param, self._func.name)
        if dtype not in SUPPORTED_DTYPES:
            raise UnsupportedDTypeError(dtype)
        return dtype

    def _eval_dtype_node(self, node: ast.expr, param: str) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        raise MissingAnnotationError(param, self._func.name)

    def _parse_layout_annotation(self, node: ast.expr) -> str:
        """Extract layout from Tensor[dtype, shape, layout='ND'] — default ND."""
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple):
            elts = node.slice.elts
            if len(elts) >= 3:
                third = elts[2]
                if isinstance(third, ast.Constant) and isinstance(third.value, str):
                    return third.value
        return "ND"

    # ------------------------------------------------------------------
    # Decorator / tiling config
    # ------------------------------------------------------------------

    def _extract_tiling_config(self) -> TilingConfig:
        cfg = TilingConfig()
        for dec in self._func.decorator_list:
            if isinstance(dec, ast.Call):
                name = self._decorator_name(dec)
                if name in _ASCEND_OP_NAMES:
                    for kw in dec.keywords:
                        if kw.arg == "tile_size" and isinstance(kw.value, ast.Constant):
                            cfg.block_size = int(kw.value.value)
                        elif kw.arg == "block_dim" and isinstance(kw.value, ast.Constant):
                            cfg.block_dim = int(kw.value.value)
                elif name == "tile":
                    for kw in dec.keywords:
                        if kw.arg == "block_size" and isinstance(kw.value, ast.Constant):
                            cfg.block_size = int(kw.value.value)
                        elif kw.arg == "buffer_num" and isinstance(kw.value, ast.Constant):
                            cfg.buffer_num = int(kw.value.value)
        return cfg

    def _decorator_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    # ------------------------------------------------------------------
    # Body walking
    # ------------------------------------------------------------------

    def _walk_body(self) -> None:
        for stmt in self._func.body:
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                # Give the output a canonical name (z, z1, z2 …) so generated code
                # uses readable tensor names instead of internal _tmpN names.
                n_outs = len(self._outputs)
                canonical = "z" if n_outs == 0 else f"z{n_outs}"
                actual = self._lower_expr(stmt.value, out_name=canonical)
                self._finalize_output(actual if actual is not None else canonical)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    out_name = stmt.targets[0].id
                    self._lower_expr(stmt.value, out_name=out_name)
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                # x += y  →  x = x + y
                var = stmt.target.id
                synthetic = ast.BinOp(
                    left=ast.Name(id=var, ctx=ast.Load()),
                    op=stmt.op,
                    right=stmt.value,
                )
                self._lower_expr(synthetic, out_name=var)
            elif isinstance(stmt, ast.Expr):
                pass  # ignore bare expressions / docstrings

    def _finalize_output(self, var_name: str) -> None:
        # Resolve dtype from var_types
        dtype = self._var_types.get(var_name, self._primary_input_dtype())
        # Rename output variable if it collides with an input name
        out_spec_name = var_name if var_name not in self._input_names else var_name + "_out"
        spec = TensorSpec(name=out_spec_name, dtype=dtype, is_input=False)
        self._outputs.append(spec)
        self._var_types[out_spec_name] = dtype
        # If the last node's output refers to var_name, update to canonical output name
        if self._nodes and self._nodes[-1].outputs and self._nodes[-1].outputs[-1] == var_name:
            self._nodes[-1].outputs[-1] = out_spec_name

    # ------------------------------------------------------------------
    # Expression lowering (recursive descent)
    # ------------------------------------------------------------------

    def _fresh_var(self) -> str:
        self._var_counter += 1
        return f"_tmp{self._var_counter}"

    def _lower_expr(self, node: ast.expr, out_name: str | None = None) -> str | None:
        """Recursively lower an AST expression into IR nodes.

        Returns the variable name holding the result.
        """
        out = out_name or self._fresh_var()

        if isinstance(node, ast.Name):
            # Simple variable reference — no node needed
            if out_name and out_name != node.id:
                # Emit an identity copy only if we're assigning to a new name
                self._emit_node(OpKind.RELU, [node.id], [out], node)  # use identity-like
                # Actually we want a true identity — skip and just alias
                # Remove the wrongly emitted node
                self._nodes.pop()
                # Propagate type
                self._var_types[out] = self._var_types.get(node.id, self._primary_input_dtype())
            return node.id if not out_name else out

        if isinstance(node, ast.Constant):
            # Scalar constant — not a standalone tensor node; return sentinel
            return None

        if isinstance(node, ast.BinOp):
            return self._lower_binop(node, out)

        if isinstance(node, ast.Call):
            return self._lower_call(node, out)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            src = self._lower_expr(node.operand)
            self._emit_node(OpKind.NEG, [src], [out], node)
            return out

        raise UnsupportedOperationError(
            ast.dump(node), getattr(node, "lineno", None)
        )

    def _lower_binop(self, node: ast.BinOp, out: str) -> str:
        lhs_is_scalar = isinstance(node.left, ast.Constant)
        rhs_is_scalar = isinstance(node.right, ast.Constant)

        if lhs_is_scalar and rhs_is_scalar:
            raise UnsupportedOperationError(
                "constant folding not supported", getattr(node, "lineno", None)
            )

        if rhs_is_scalar or lhs_is_scalar:
            # Scalar broadcast variant
            tensor_node = node.left if rhs_is_scalar else node.right
            scalar_val = node.right.value if rhs_is_scalar else node.left.value
            kind = BINOP_AST_TO_SCALAR_KIND.get(type(node.op))
            if kind is None:
                raise UnsupportedOperationError(
                    type(node.op).__name__, getattr(node, "lineno", None)
                )
            src = self._lower_expr(tensor_node)
            self._emit_node(kind, [src], [out], node, {"scalar_value": scalar_val})
            return out

        # Tensor × tensor
        kind = BINOP_AST_TO_KIND.get(type(node.op))
        if kind is None:
            raise UnsupportedOperationError(
                type(node.op).__name__, getattr(node, "lineno", None)
            )
        lhs = self._lower_expr(node.left)
        rhs = self._lower_expr(node.right)
        self._emit_node(kind, [lhs, rhs], [out], node)
        return out

    def _lower_call(self, node: ast.Call, out: str) -> str:
        if not isinstance(node.func, ast.Name):
            raise UnsupportedOperationError(
                ast.dump(node.func), getattr(node, "lineno", None)
            )
        fname = node.func.id
        kind = CALL_NAME_TO_KIND.get(fname)
        if kind is None:
            raise UnsupportedOperationError(fname, getattr(node, "lineno", None))

        attrs: dict[str, Any] = {}

        if kind == OpKind.CAST:
            # cast(x, dtype)
            src = self._lower_expr(node.args[0])
            target_dtype = self._eval_dtype_node(node.args[1], "cast")
            attrs["target_dtype"] = target_dtype
            self._emit_node(kind, [src], [out], node, attrs)
            return out

        if kind in REDUCTION_OPS:
            src = self._lower_expr(node.args[0])
            # axis keyword or positional second arg
            axis = -1
            keepdims = False
            if len(node.args) >= 2:
                arg2 = node.args[1]
                if isinstance(arg2, ast.Constant):
                    axis = int(arg2.value)
            for kw in node.keywords:
                if kw.arg == "axis" and isinstance(kw.value, ast.Constant):
                    axis = int(kw.value.value)
                elif kw.arg == "dim" and isinstance(kw.value, ast.Constant):
                    axis = int(kw.value.value)
                elif kw.arg == "keepdims" and isinstance(kw.value, ast.Constant):
                    keepdims = bool(kw.value.value)
            attrs = {"axis": axis, "keepdims": keepdims}
            self._emit_node(kind, [src], [out], node, attrs)
            return out

        if kind == OpKind.MATMUL:
            lhs = self._lower_expr(node.args[0])
            rhs = self._lower_expr(node.args[1])
            self._emit_node(kind, [lhs, rhs], [out], node)
            return out

        # Unary ops
        src = self._lower_expr(node.args[0])
        self._emit_node(kind, [src], [out], node, attrs)
        return out

    def _emit_node(self, kind: OpKind, inputs: list[str], outputs: list[str],
                   ast_node: ast.expr, attrs: dict[str, Any] | None = None) -> None:
        # Filter out None inputs (scalars that resolved to None)
        inputs = [i for i in inputs if i is not None]
        node = IRNode(
            kind=kind,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs or {},
            lineno=getattr(ast_node, "lineno", None),
        )
        self._nodes.append(node)

    # ------------------------------------------------------------------
    # Type inference
    # ------------------------------------------------------------------

    def _infer_var_types(self) -> None:
        for node in self._nodes:
            if node.kind == OpKind.CAST:
                target = node.attrs.get("target_dtype", self._primary_input_dtype())
                for out in node.outputs:
                    self._var_types[out] = target
                continue

            # Propagate dtype of first input
            if node.inputs:
                src_dtype = self._var_types.get(node.inputs[0], self._primary_input_dtype())
            else:
                src_dtype = self._primary_input_dtype()

            for out in node.outputs:
                if out in self._var_types:
                    existing = self._var_types[out]
                    if existing != src_dtype and node.kind not in REDUCTION_OPS:
                        raise TypeMismatchError(str(node.kind), existing, src_dtype)
                self._var_types[out] = src_dtype

    def _primary_input_dtype(self) -> str:
        if self._inputs:
            return self._inputs[0].dtype
        return "float16"

    # ------------------------------------------------------------------
    # Category detection
    # ------------------------------------------------------------------

    def _categorize(self) -> OpCategory:
        kinds = {n.kind for n in self._nodes}
        if kinds & MATMUL_OPS:
            return OpCategory.MATMUL
        if kinds & REDUCTION_OPS:
            return OpCategory.REDUCTION
        return OpCategory.ELEMENTWISE


# ---------------------------------------------------------------------------
# File-level entry point
# ---------------------------------------------------------------------------

def _has_ascend_op_decorator(func_def: ast.FunctionDef) -> bool:
    for dec in func_def.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in _ASCEND_OP_NAMES:
            return True
        if isinstance(dec, ast.Call):
            name = dec.func.id if isinstance(dec.func, ast.Name) else (
                dec.func.attr if isinstance(dec.func, ast.Attribute) else ""
            )
            if name in _ASCEND_OP_NAMES:
                return True
    return False


def analyze_file(source: str) -> list[OperatorIR]:
    """Parse a Python source string; return OperatorIR for every @ascend_op function."""
    tree = ast.parse(source)
    results: list[OperatorIR] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_ascend_op_decorator(node):
            results.append(ASTAnalyzer(node).analyze())
    return results
