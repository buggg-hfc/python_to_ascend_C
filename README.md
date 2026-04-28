# python_to_ascend_C

A transpiler that converts Python operator definitions into **Ascend C** (C++ for Huawei Ascend NPU), so you can focus on operator logic without dealing with Ascend C boilerplate (pipe/queue management, tiling, explicit GM↔UB memory copies).

## Quick start

```bash
pip install -e ".[dev]"
```

Write an operator in Python:

```python
# my_ops.py
from ascend_transpiler import ascend_op, Tensor, float16, relu

@ascend_op
def fused_add_relu(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    tmp = x + y
    return relu(tmp)
```

Transpile it:

```bash
ascend-transpiler my_ops.py -o out/
# [fused_add_relu]
#   wrote: out/fused_add_relu.cpp
#   wrote: out/fused_add_relu_tiling.h
#   wrote: out/fused_add_relu_tiling.cpp
```

The generated `.cpp` is a complete, compilable Ascend C kernel with `Init()`, `Process()`, `CopyIn()`, `Compute()`, `CopyOut()`, and the `KERNEL_FUNC` entry point.

---

## How it works

```
Python source (@ascend_op function)
        │
        ▼
  AST Analyzer  ──► OperatorIR  ──► TilingCalculator
                                           │
                                           ▼
                                    CodeGenerator (Jinja2)
                                           │
                                    ┌──────┴───────┐
                                    ▼              ▼
                               kernel.cpp    kernel_tiling.h
                                              kernel_tiling.cpp
```

The transpiler identifies the **compute pattern** (Elementwise / Reduction / MatMul) and selects the appropriate code template. Intermediate variables in fused ops stay in UB memory — no queue roundtrip, no extra GM copy.

---

## DSL reference

### Decorator

```python
@ascend_op                          # auto tile_size from UB capacity
@ascend_op(tile_size=512)           # manual tile override
@ascend_op(tile_size=256, block_dim=8)
```

The `@tile` decorator can also be stacked:

```python
@ascend_op
@tile(block_size=256, buffer_num=2)
def my_op(x: Tensor[float16]) -> Tensor[float16]: ...
```

### Tensor type annotation

```python
Tensor[float16]                     # dynamic shape, ND layout
Tensor[float32, (M, K)]             # static shape hint
Tensor[float16, (N,), "NZ"]         # explicit layout
```

### Supported dtypes

| Python DSL | C++ type | Notes |
|-----------|----------|-------|
| `float16` | `half` | — |
| `float32` | `float` | — |
| `float64` | `double` | — |
| `bfloat16` | `bfloat16_t` | — |
| `int8` | `int8_t` | — |
| `int32` | `int32_t` | — |
| `int64` | `int64_t` | — |
| `uint8` | `uint8_t` | — |
| `mxfp8` | `fp8e4m3_t` | Ascend 910C+, MX microscaling |
| `mxfp4` | `fp4e2m1_t` | Ascend 910C+, MX microscaling |
| `hif8` | `hif8_t` | HiFloat8 |

### Supported operations

**Elementwise unary**

| DSL | AscendC API |
|-----|------------|
| `relu(x)` | `Relu` |
| `gelu(x)` | `Gelu` |
| `silu(x)` | `Silu` |
| `sigmoid(x)` | `Sigmoid` |
| `tanh(x)` | `Tanh` |
| `exp(x)` | `Exp` |
| `log(x)` | `Log` |
| `sqrt(x)` | `Sqrt` |
| `abs(x)` | `Abs` |
| `sin(x)` | `Sin` |
| `cos(x)` | `Cos` |
| `floor(x)` | `Floor` |
| `ceil(x)` | `Ceil` |
| `round(x)` | `Round` |
| `sign(x)` | `Sign` |
| `reciprocal(x)` | `Reciprocal` |
| `leaky_relu(x, alpha=0.01)` | `LeakyRelu` |
| `-x` | `Neg` |
| `cast(x, float32)` | `Cast` |

**Elementwise binary (tensor × tensor)**

| DSL | AscendC API |
|-----|------------|
| `x + y` | `Add` |
| `x - y` | `Sub` |
| `x * y` | `Mul` |
| `x / y` | `Div` |
| `x ** y` | `Pow` |
| `maximum(x, y)` | `Maximum` |
| `minimum(x, y)` | `Minimum` |

**Elementwise binary (tensor × scalar)**

| DSL | AscendC API |
|-----|------------|
| `x + 1.0` | `Adds` |
| `x * 2.0` | `Muls` |
| `x - 0.5` | `Subs` |
| `x / 4.0` | `Divs` |

**Reduction**

| DSL | AscendC API |
|-----|------------|
| `reduce_sum(x, axis=-1)` | `ReduceSum` |
| `reduce_max(x, axis=-1)` | `ReduceMax` |
| `reduce_min(x, axis=-1)` | `ReduceMin` |
| `reduce_mean(x, axis=-1)` | `ReduceMean` |

**MatMul (Cube core)**

| DSL | AscendC API |
|-----|------------|
| `matmul(a, b)` | `MatMul` |

---

## Examples

### Simple elementwise

```python
@ascend_op
def add_custom(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    return x + y
```

### Scalar broadcast

```python
@ascend_op(tile_size=512)
def scale_custom(x: Tensor[float16]) -> Tensor[float16]:
    return x * 2.0   # → Muls API, no dummy tensor
```

### Fused op (intermediate stays in UB)

```python
@ascend_op
def fused_add_relu(x: Tensor[float16], y: Tensor[float16]) -> Tensor[float16]:
    tmp = x + y        # tmp in UB via pipe.AllocTensor — no GM round-trip
    return relu(tmp)
```

### Clamp (composed from maximum + minimum)

```python
@ascend_op
def clamp_op(x: Tensor[float16], lo: Tensor[float16], hi: Tensor[float16]) -> Tensor[float16]:
    clamped_lo = maximum(x, lo)
    return minimum(clamped_lo, hi)
```

### Leaky ReLU with custom alpha

```python
@ascend_op
def leaky_relu_custom(x: Tensor[float32]) -> Tensor[float32]:
    return leaky_relu(x, alpha=0.1)
```

### Reduction

```python
@ascend_op
def reduce_sum_custom(x: Tensor[float32]) -> Tensor[float32]:
    return reduce_sum(x, axis=-1)
```

### MatMul

```python
@ascend_op
def matmul_custom(a: Tensor[float16], b: Tensor[float16]) -> Tensor[float32]:
    return matmul(a, b)
```

---

## CLI

```
ascend-transpiler <input.py> [-o OUTPUT_DIR] [--ub-size KB] [--block-dim N]

  input           Python source file with @ascend_op functions
  -o, --output-dir  Output directory (default: current dir)
  --ub-size KB    On-chip UB size in KB for auto tiling (default: 256)
  --block-dim N   Number of AI Core blocks (default: 8)
```

## Programmatic API

```python
from ascend_transpiler import Transpiler
import pathlib

t = Transpiler(ub_size_kb=256, default_block_dim=8)
results = t.transpile_file(pathlib.Path("my_ops.py"), pathlib.Path("out/"))
# {"add_custom": ["out/add_custom.cpp", "out/add_custom_tiling.h", ...]}
```

---

## Project structure

```
src/ascend_transpiler/
├── dsl/            # @ascend_op, Tensor[dtype], primitive stubs
├── analyzer/       # Python AST → OperatorIR
├── ir/             # OperatorIR, IRNode, OpKind
├── ops/            # dtype & op mapping tables
├── tiling/         # auto tile_size calculation
└── codegen/        # Jinja2 templates + pattern generators
    └── templates/  # kernel_elementwise / reduction / matmul .cpp.j2
```

## Running tests

```bash
pytest tests/ -v
```

---

## Reference

- [Ascend C API reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0109.html)
- [Operator development guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0004.html)
