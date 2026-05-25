#!/usr/bin/env python3
"""Convert a .tflite model into a C header byte array for Arduino.

Usage:
  python convert_model_to_header.py \
      --input models/tflite/nepspot_int8.tflite \
      --output nepspot_model_data.h \
      --array-name nepspot_model_data
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TFLite model to C header")
    parser.add_argument(
        "--input",
        default="models/tflite/nepspot_int8.tflite",
        help="Path to input .tflite file",
    )
    parser.add_argument(
        "--output",
        default="nepspot_model_data.h",
        help="Path to output header file",
    )
    parser.add_argument(
        "--array-name",
        default="nepspot_model_data",
        help="C array symbol name",
    )
    return parser.parse_args()


def format_bytes(data: bytes, bytes_per_line: int = 12) -> str:
    hex_bytes = [f"0x{b:02x}" for b in data]
    lines = []
    for i in range(0, len(hex_bytes), bytes_per_line):
      lines.append("  " + ", ".join(hex_bytes[i:i + bytes_per_line]))
    return ",\n".join(lines)


def write_header(model_bytes: bytes, output_path: Path, array_name: str) -> None:
    include_guard = f"{array_name.upper()}_H_"

    body = format_bytes(model_bytes)

    content = (
        f"#ifndef {include_guard}\n"
        f"#define {include_guard}\n\n"
        "#include <stdint.h>\n\n"
        f"const unsigned char {array_name}[] = {{\n"
        f"{body}\n"
        "};\n\n"
        f"const unsigned int {array_name}_len = {len(model_bytes)};\n\n"
        f"#endif  // {include_guard}\n"
    )

    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    model_bytes = input_path.read_bytes()
    write_header(model_bytes, output_path, args.array_name)

    print(f"Wrote {output_path} ({len(model_bytes)} bytes)")


if __name__ == "__main__":
    main()
