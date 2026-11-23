### m17/convert.py
#
# Illustrate base-64 encoding and decoding.
#
# Authors: Mike Smith, using Google Gemini.
# Date:    20261121

import base64


def encode_spanish_to_base64(text: str) -> str:
    """Convert a Spanish string into a Base64-encoded ASCII string."""
    utf8_bytes = text.encode("utf-8")
    b64_bytes = base64.b64encode(utf8_bytes)
    return b64_bytes.decode("ascii")


def decode_base64_to_spanish(b64_text: str) -> str:
    """Convert a Base64 string back into a Spanish string."""
    utf8_bytes = base64.b64decode(b64_text.strip(), validate=True)
    return utf8_bytes.decode("utf-8")


def main() -> None:
    line = input("Enter a line of Spanish text: ")
    encoded = encode_spanish_to_base64(line)
    decoded = decode_base64_to_spanish(encoded)

    print(f"Encoded (Base64): {encoded}")
    print(f"Decoded (Spanish): {decoded}")


if __name__ == "__main__":
    main()
