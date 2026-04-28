class TranspilerError(Exception):
    pass


class UnsupportedOperationError(TranspilerError):
    def __init__(self, op_name: str, line: int | None = None):
        loc = f" (line {line})" if line else ""
        super().__init__(f"Unsupported operation: '{op_name}'{loc}")
        self.op_name = op_name
        self.line = line


class UnsupportedDTypeError(TranspilerError):
    def __init__(self, dtype: str):
        super().__init__(f"Unsupported dtype: '{dtype}'")
        self.dtype = dtype


class TypeMismatchError(TranspilerError):
    def __init__(self, op: str, expected: str, got: str):
        super().__init__(f"Type mismatch in '{op}': expected {expected}, got {got}")


class MissingAnnotationError(TranspilerError):
    def __init__(self, param: str, func: str):
        super().__init__(f"Parameter '{param}' in function '{func}' has no type annotation")
        self.param = param
        self.func = func


class TilingError(TranspilerError):
    pass
