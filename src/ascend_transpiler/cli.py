"""Command-line interface for ascend-transpiler."""
from __future__ import annotations

import argparse
import pathlib
import sys

from ascend_transpiler.exceptions import TranspilerError
from ascend_transpiler.transpiler import Transpiler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ascend-transpiler",
        description="Transpile Python @ascend_op functions to Ascend C (C++) kernel files.",
    )
    parser.add_argument("input", type=pathlib.Path, help="Input Python source file")
    parser.add_argument(
        "-o", "--output-dir", type=pathlib.Path, default=pathlib.Path("."),
        metavar="DIR", help="Output directory (default: current dir)",
    )
    parser.add_argument(
        "--ub-size", type=int, default=256, metavar="KB",
        help="On-chip Unified Buffer size in KB (default: 256)",
    )
    parser.add_argument(
        "--block-dim", type=int, default=8,
        help="Number of AI Core blocks (default: 8)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        transpiler = Transpiler(ub_size_kb=args.ub_size, default_block_dim=args.block_dim)
        results = transpiler.transpile_file(args.input, args.output_dir)
    except TranspilerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for op_name, written in results.items():
        print(f"[{op_name}]")
        for path in written:
            print(f"  wrote: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
