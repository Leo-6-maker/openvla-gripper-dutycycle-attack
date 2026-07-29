"""Small fail-closed publication primitive for sealed development roots."""
from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path


def rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""
    if os.name != "posix":
        raise RuntimeError("strict no-clobber publish is unsupported on this platform")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise RuntimeError("renameat2 is unavailable; refusing non-atomic publish") from exc
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(target)
    raise OSError(error, os.strerror(error), target)
