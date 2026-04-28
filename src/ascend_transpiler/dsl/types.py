"""DSL type stubs — used for IDE autocompletion and type annotations only.

These are never evaluated at runtime by the transpiler; it reads source text
and inspects AST annotation nodes directly.
"""
from __future__ import annotations


class _TensorMeta(type):
    def __getitem__(cls, params):
        return cls


class Tensor(metaclass=_TensorMeta):
    """Annotation stub: Tensor[dtype] or Tensor[dtype, (M, K)]."""
    pass


# dtype string constants so users can write `Tensor[float16]` without importing numpy
float16 = "float16"
float32 = "float32"
float64 = "float64"
bfloat16 = "bfloat16"
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
# Ascend-specific microscaling / hiFloat formats
mxfp8 = "mxfp8"
mxfp4 = "mxfp4"
hif8 = "hif8"
