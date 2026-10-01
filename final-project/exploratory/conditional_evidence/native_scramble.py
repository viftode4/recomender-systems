"""Portable exact integer 2-switches, with optional system-compiler acceleration.

Both implementations use SplitMix64 with uint64 wraparound. Each of 20*E
sequential proposals draws an edge uniformly, then another edge uniformly from
the same rating's fixed row-major slot group. Rejection sampling maps random
uint64 words to bounded integers without modulo bias. Identical rows, columns,
edge slots, or occupied crossed cells reject a proposal. Accepted swaps move
only the two column endpoints. Query/candidate edges are outside this API.

The Python implementation is the portable exact reference. The native backend
uses only integer arithmetic; no fast-math, package installation or network is
used. Compilation artifacts stay under the ignored runs/ tree. A caller may
force ``backend='python'`` without inspecting the compiler or native cache.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

ALGORITHM = "sequential-rating-2switch-splitmix64-rejection-v1"
MASK64 = (1 << 64) - 1
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_suffix(".c")
CACHE_ROOT = ROOT / "runs/conditional-evidence-native"
_CACHE = None
_LOCK = threading.Lock()


class NativeUnavailable(RuntimeError):
    pass


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix(value):
    array = np.asarray(value)
    if (array.ndim != 2 or array.dtype.kind not in "biuf" or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > 5) or np.any(array != np.floor(array))):
        raise ValueError("History must be a finite integer category matrix 0..5")
    return np.array(array, dtype=np.uint8, order="C", copy=True)


def _seed(seed):
    if not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) <= MASK64:
        raise ValueError("Seed must be an integer in uint64 range")
    return int(seed)


def _next(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, value ^ (value >> 31)


def _bounded(state, count):
    threshold = ((-count) & MASK64) % count
    while True:
        state, value = _next(state)
        if value >= threshold:
            return state, value % count


def _python(result, seed):
    rows, cols = (x.tolist() for x in np.nonzero(result))
    count = len(rows)
    if min(result.shape) < 2 or count < 2:
        return result, {"attempts": 0, "successes": 0}
    ratings = [int(result[row, col]) for row, col in zip(rows, cols)]
    groups = [[index for index, value in enumerate(ratings) if value == rating] for rating in range(6)]
    successes = 0
    for _ in range(20 * count):
        seed, left = _bounded(seed, count)
        rating = ratings[left]
        seed, draw = _bounded(seed, len(groups[rating]))
        right = groups[rating][draw]
        a, b, c, d = rows[left], cols[left], rows[right], cols[right]
        if left == right or a == c or b == d or result[a, d] or result[c, b]:
            continue
        result[a, b] = result[c, d] = 0
        result[a, d] = result[c, b] = rating
        cols[left], cols[right] = d, b
        successes += 1
    return result, {"attempts": 20 * count, "successes": successes}


def _load_native():
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        if sys.platform not in ("darwin", "linux"):
            raise NativeUnavailable("Native compilation supports macOS/Linux; exact Python fallback available")
        compiler = shutil.which("clang")
        if compiler is None:
            raise NativeUnavailable("No system clang; exact Python fallback available")
        source_hash = _digest(SOURCE)
        folder = CACHE_ROOT / source_hash
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        library = folder / ("native_scramble" + suffix)
        manifest_path = folder / "manifest.json"
        flags = ["-std=c11", "-O3", "-fno-fast-math"]
        flags += ["-dynamiclib", "-Wl,-install_name,@rpath/native_scramble.dylib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
        if library.exists() or manifest_path.exists():
            if not (library.exists() and manifest_path.exists()):
                raise RuntimeError("Incomplete native cache; use a new/clean cache before a sealed run")
            record = json.loads(manifest_path.read_text())
            if (record["source_sha256"] != source_hash or record["library_sha256"] != _digest(library)
                    or record["machine"] != platform.machine() or record["flags"] != flags):
                raise RuntimeError("Native cache hash/platform/flags mismatch")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            version = subprocess.run([compiler, "--version"], check=True, capture_output=True, text=True).stdout.strip()
            # The final library is published only after successful compilation.
            with tempfile.TemporaryDirectory(prefix="build-", dir=folder) as temporary:
                built = Path(temporary) / library.name
                command = [compiler, *flags, str(SOURCE), "-o", str(built)]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise NativeUnavailable("System clang compilation failed: " + result.stderr.strip())
                record = {"algorithm": ALGORITHM, "source_sha256": source_hash,
                          "library_sha256": _digest(built), "compiler": compiler,
                          "compiler_sha256": _digest(compiler),
                          "compiler_version": version, "flags": flags,
                          "machine": platform.machine(), "system": platform.system()}
                built.replace(library)
                manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        if "compiler_sha256" not in record:
            # Upgrade only a preparation cache produced before this field existed,
            # after verifying its exact compiler version; sealed records include it.
            version = subprocess.run([record["compiler"], "--version"], check=True, capture_output=True, text=True).stdout.strip()
            if version != record["compiler_version"]:
                raise RuntimeError("Cannot recover compiler digest for a changed preparation compiler")
            record["compiler_sha256"] = _digest(record["compiler"])
            manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        loaded = ctypes.CDLL(str(library))
        function = loaded.ce_scramble
        function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
        function.restype = ctypes.c_int
        _CACHE = loaded, function, {**record, "backend": "native", "wrapper_sha256": _digest(__file__),
                                    "library_path": str(library.relative_to(ROOT))}
        return _CACHE


def native_provenance():
    """Resolve backend before a run seal and record compiler/library digests."""
    try:
        record = dict(_load_native()[2])
        if (record["source_sha256"] != _digest(SOURCE)
                or record["wrapper_sha256"] != _digest(__file__)
                or record["library_sha256"] != _digest(ROOT / record["library_path"])):
            raise RuntimeError("Native runtime source or library changed after initialization")
        return record
    except NativeUnavailable as error:
        return {"backend": "python", "algorithm": ALGORITHM, "reason": str(error),
                "source_sha256": _digest(SOURCE), "wrapper_sha256": _digest(__file__)}


def runtime_metadata():
    """Seal this dictionary; a later backend/compiler/library change must reject resume."""
    return native_provenance()


def scramble_history(matrix, seed, *, backend="auto"):
    result, seed = _matrix(matrix), _seed(seed)
    if backend not in ("auto", "native", "python"):
        raise ValueError("Backend must be auto, native or python")
    if backend == "python":
        return _python(result, seed)
    try:
        _, function, _ = _load_native()
    except NativeUnavailable:
        if backend == "native":
            raise
        return _python(result, seed)
    attempts, successes = ctypes.c_uint64(), ctypes.c_uint64()
    status = function(result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), *result.shape,
                      seed, ctypes.byref(attempts), ctypes.byref(successes))
    if status:
        raise RuntimeError(f"Native history scramble failed with status {status}")
    return result, {"attempts": attempts.value, "successes": successes.value}
