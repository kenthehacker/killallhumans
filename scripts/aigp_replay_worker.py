"""Untrusted candidate replay worker; launch only through an isolation wrapper."""

from __future__ import annotations

import base64
import contextlib
import importlib
import io
import json
import random
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

_CANDIDATE_ROOT = Path(__file__).resolve().parent.parent


def _deep_freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    return value


def main() -> int:
    if len(sys.argv) != 4 or ":" not in sys.argv[1]:
        return 2
    try:
        seed = int(sys.argv[2])
    except ValueError:
        return 2
    if sys.argv[2] != str(seed):
        return 2
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    # ``-I`` intentionally discards PYTHONPATH and the script directory.  Add
    # only the candidate root derived from the securely selected absolute
    # worker path, after the worker's stdlib/NumPy dependencies are loaded.
    candidate_root = str(_CANDIDATE_ROOT)
    if candidate_root not in sys.path:
        sys.path.insert(0, candidate_root)
    module_name, attribute = sys.argv[1].split(":", 1)
    expected_relative = Path(sys.argv[3])
    if (
        expected_relative.is_absolute()
        or expected_relative.drive
        or not expected_relative.parts
        or any(part in {"", ".", ".."} for part in expected_relative.parts)
    ):
        return 2
    try:
        expected_source = (_CANDIDATE_ROOT / expected_relative).resolve(
            strict=True
        )
    except OSError:
        return 2
    if _CANDIDATE_ROOT not in expected_source.parents:
        return 2
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        module = importlib.import_module(module_name)
        module_spec = getattr(module, "__spec__", None)
        origin = getattr(module_spec, "origin", None)
        try:
            resolved_origin = (
                Path(origin).resolve(strict=True) if type(origin) is str else None
            )
        except OSError:
            return 4
        if resolved_origin != expected_source:
            return 4
        processor = getattr(module, attribute)
    for line in sys.stdin:
        request = json.loads(line)
        if (
            type(request) is not dict
            or set(request)
            != {"schema", "request_id", "image_npy_base64", "context"}
            or request["schema"] != "aigp-replay-worker-request/1"
            or type(request["request_id"]) is not int
            or request["request_id"] < 0
            or type(request["image_npy_base64"]) is not str
            or type(request["context"]) is not dict
        ):
            return 3
        image = np.load(
            io.BytesIO(base64.b64decode(request["image_npy_base64"], validate=True)),
            allow_pickle=False,
        )
        image.setflags(write=False)
        context = _deep_freeze(request["context"])
        with contextlib.redirect_stdout(sys.stderr):
            result = processor(image, context)
        protocol.write(
            json.dumps(
                {
                    "schema": "aigp-replay-worker-response/1",
                    "request_id": request["request_id"],
                    "result": result,
                },
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        protocol.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
