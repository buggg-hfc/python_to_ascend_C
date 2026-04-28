"""Matrix multiplication: C = A @ B (float16 inputs, float32 output via cube core)."""
from ascend_transpiler import ascend_op, Tensor, float16, float32, matmul, cast


@ascend_op
def matmul_custom(a: Tensor[float16], b: Tensor[float16]) -> Tensor[float32]:
    return matmul(a, b)
