from __future__ import annotations

import sys


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _parse_arg(arg: str) -> int:
    try:
        val = int(arg)
    except ValueError as exc:
        raise ValueError("Argument must be an integer") from exc
    return val


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python tools/factorial.py <non-negative-integer>")
        return 1
    n = _parse_arg(argv[0])
    print(factorial(n))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Lightweight self-checks
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        print("Self-checks passed.")
        sys.exit(0)
    sys.exit(main(sys.argv[1:]))
