"""@ascend_op and @tile decorators — runtime no-ops; used by the transpiler as markers."""
from __future__ import annotations

from typing import Any, Callable


def ascend_op(_fn: Callable | None = None, *, tile_size: int | None = None,
              layout: str = "ND", block_dim: int = 8):
    """Mark a function as an Ascend C operator to be transpiled.

    Usage:
        @ascend_op
        def add(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]: ...

        @ascend_op(tile_size=256)
        def scale(x: Tensor[float16]) -> Tensor[float16]: ...
    """
    def decorator(fn: Callable) -> Callable:
        fn._ascend_op = True
        # Don't overwrite tile_size already set by @tile decorator
        fn._ascend_tile_size = tile_size if tile_size is not None else getattr(fn, "_ascend_tile_size", None)
        fn._ascend_layout = layout
        fn._ascend_block_dim = block_dim
        return fn

    if _fn is not None:
        # Called as bare @ascend_op (no parentheses)
        return decorator(_fn)
    # Called as @ascend_op(...) — return the decorator
    return decorator


def tile(block_size: int = 256, buffer_num: int = 2) -> Callable:
    """Optional tiling hint; overrides auto-calculation."""
    def decorator(fn: Callable) -> Callable:
        fn._ascend_tile_size = block_size
        fn._ascend_buffer_num = buffer_num
        return fn
    return decorator


# ---------------------------------------------------------------------------
# No-op DSL primitive stubs — imported by user code for IDE completion.
# The transpiler reads source text; these functions are NEVER called at runtime.
# ---------------------------------------------------------------------------
def _noop(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "Ascend DSL primitives must only be used inside @ascend_op functions. "
        "Call `ascend-transpiler` to transpile the source file instead of executing it."
    )


relu = _noop
sqrt = _noop
exp = _noop
log = _noop
abs = _noop
tanh = _noop
sigmoid = _noop
sin = _noop
cos = _noop
floor = _noop
ceil = _noop
round = _noop
sign = _noop
reciprocal = _noop
gelu = _noop
silu = _noop
leaky_relu = _noop
maximum = _noop
minimum = _noop
cast = _noop
matmul = _noop
reduce_sum = _noop
reduce_max = _noop
reduce_min = _noop
reduce_mean = _noop
