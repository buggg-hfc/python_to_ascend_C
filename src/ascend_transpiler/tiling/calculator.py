"""Tiling calculator — converts OperatorIR into a concrete TilingConfig.

Auto-calculates tile_size from UB capacity, dtype element size, and number
of tensors (inputs + outputs + intermediates). Aligns to 32-byte boundary
as required by AscendC SIMD instructions.
"""
from __future__ import annotations

from ascend_transpiler.exceptions import TilingError
from ascend_transpiler.ir.operator_ir import IRNode, OpCategory, OperatorIR, TilingConfig
from ascend_transpiler.ops.mappings import DTYPE_ITEMSIZE, REDUCTION_OPS

# Typical on-chip Unified Buffer size for Ascend 910B
_DEFAULT_UB_SIZE_KB = 256
_UB_SIZE_BYTES = _DEFAULT_UB_SIZE_KB * 1024

# mxfp4 is packed 2 elements per byte — needs 64-elem alignment (32 bytes)
_MXFP4_ALIGN_ELEMS = 64


def _find_intermediates(ir: OperatorIR) -> set[str]:
    """Variable names that are produced by non-final IR nodes and consumed later.

    These stay entirely in UB during Compute() — no queue roundtrip.
    """
    input_names = {t.name for t in ir.inputs}
    output_names = {t.name for t in ir.outputs}
    produced: set[str] = set()
    intermediates: set[str] = set()
    for node in ir.nodes:
        for inp in node.inputs:
            if inp in produced and inp not in input_names:
                intermediates.add(inp)
        for out in node.outputs:
            produced.add(out)
    # Remove outputs — they use the queue
    intermediates -= output_names
    return intermediates


def _primary_dtype(ir: OperatorIR) -> str:
    if ir.inputs:
        return ir.inputs[0].dtype
    if ir.outputs:
        return ir.outputs[0].dtype
    return "float16"


class TilingCalculator:
    def __init__(self, ub_size_kb: int = _DEFAULT_UB_SIZE_KB, default_block_dim: int = 8):
        self._ub_bytes = ub_size_kb * 1024
        self._default_block_dim = default_block_dim

    def calculate(self, ir: OperatorIR) -> TilingConfig:
        cfg = ir.tiling  # may already have user overrides

        if ir.category == OpCategory.MATMUL:
            return self._calc_matmul(ir, cfg)
        if ir.category == OpCategory.REDUCTION:
            return self._calc_reduction(ir, cfg)
        return self._calc_elementwise(ir, cfg)

    # ------------------------------------------------------------------
    # Elementwise
    # ------------------------------------------------------------------

    def _calc_elementwise(self, ir: OperatorIR, cfg: TilingConfig) -> TilingConfig:
        if cfg.block_size != 256:
            # User supplied explicit override — just validate alignment
            return self._validate_and_return(cfg, _primary_dtype(ir))

        n_intermediates = len(_find_intermediates(ir))
        n_tensors = len(ir.inputs) + len(ir.outputs) + n_intermediates
        dtype = _primary_dtype(ir)
        elem_bytes = DTYPE_ITEMSIZE.get(dtype, 2)

        raw = self._ub_bytes // (n_tensors * cfg.buffer_num * elem_bytes)
        tile_size = self._align(raw, dtype)
        if tile_size <= 0:
            raise TilingError(
                f"Cannot fit even one tile: UB={self._ub_bytes}B, "
                f"n_tensors={n_tensors}, dtype={dtype}"
            )
        return TilingConfig(
            block_size=tile_size,
            buffer_num=cfg.buffer_num,
            block_dim=cfg.block_dim if cfg.block_dim != 8 else self._default_block_dim,
        )

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------

    def _calc_reduction(self, ir: OperatorIR, cfg: TilingConfig) -> TilingConfig:
        if cfg.block_size != 256:
            return self._validate_and_return(cfg, _primary_dtype(ir))

        dtype = _primary_dtype(ir)
        elem_bytes = DTYPE_ITEMSIZE.get(dtype, 2)
        # Need: input tile + output scalar + workspace buffer for ReduceSum
        n_tensors = len(ir.inputs) + len(ir.outputs) + 1  # +1 workspace
        raw = self._ub_bytes // (n_tensors * cfg.buffer_num * elem_bytes)
        tile_size = self._align(raw, dtype)
        if tile_size <= 0:
            raise TilingError(f"Reduction tile size underflow for dtype={dtype}")
        return TilingConfig(
            block_size=tile_size,
            buffer_num=cfg.buffer_num,
            block_dim=cfg.block_dim if cfg.block_dim != 8 else self._default_block_dim,
        )

    # ------------------------------------------------------------------
    # MatMul
    # ------------------------------------------------------------------

    def _calc_matmul(self, ir: OperatorIR, cfg: TilingConfig) -> TilingConfig:
        # For matmul: block_size repurposed as M/N tile = block_size, K tile = block_size
        if cfg.block_size == 256:
            cfg = TilingConfig(block_size=128, buffer_num=cfg.buffer_num,
                               block_dim=self._default_block_dim)
        return cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _align(self, raw: int, dtype: str) -> int:
        elem_bytes = DTYPE_ITEMSIZE.get(dtype, 2)
        if dtype == "mxfp4":
            return (raw // _MXFP4_ALIGN_ELEMS) * _MXFP4_ALIGN_ELEMS
        align_elems = max(1, 32 // elem_bytes)
        return (raw // align_elems) * align_elems

    def _validate_and_return(self, cfg: TilingConfig, dtype: str) -> TilingConfig:
        aligned = self._align(cfg.block_size, dtype)
        if aligned != cfg.block_size:
            raise TilingError(
                f"User-specified tile_size={cfg.block_size} is not aligned for dtype={dtype}. "
                f"Use {aligned} instead."
            )
        return cfg
