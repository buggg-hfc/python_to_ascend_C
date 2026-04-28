"""Row-wise reduce sum: y[i] = sum(x[i, :])."""
from ascend_transpiler import ascend_op, Tensor, float32, reduce_sum


@ascend_op
def reduce_sum_custom(x: Tensor[float32]) -> Tensor[float32]:
    return reduce_sum(x, axis=-1)
