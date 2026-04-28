"""Tests for DSL types and decorators."""
import pytest

from ascend_transpiler.dsl.types import Tensor, float16, float32, mxfp8, mxfp4, hif8
from ascend_transpiler.dsl.decorators import ascend_op, tile


def test_tensor_subscript_does_not_raise():
    ann = Tensor[float16]
    assert ann is Tensor


def test_tensor_subscript_with_shape():
    ann = Tensor[float16, (128, 64)]
    assert ann is Tensor


def test_dtype_constants_are_strings():
    assert float16 == "float16"
    assert float32 == "float32"
    assert mxfp8 == "mxfp8"
    assert mxfp4 == "mxfp4"
    assert hif8 == "hif8"


def test_ascend_op_bare_passthrough():
    @ascend_op
    def my_op(x: Tensor[float16]) -> Tensor[float16]:
        return x

    assert callable(my_op)
    assert hasattr(my_op, "_ascend_op")
    assert my_op._ascend_op is True
    assert my_op._ascend_tile_size is None


def test_ascend_op_with_tile_size():
    @ascend_op(tile_size=512)
    def my_op(x: Tensor[float16]) -> Tensor[float16]:
        return x

    assert my_op._ascend_tile_size == 512


def test_tile_decorator_sets_attrs():
    @ascend_op
    @tile(block_size=256, buffer_num=3)
    def my_op(x: Tensor[float16]) -> Tensor[float16]:
        return x

    assert my_op._ascend_tile_size == 256
    assert my_op._ascend_buffer_num == 3
