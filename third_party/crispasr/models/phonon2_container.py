"""Phonon-2's five-value/int6 transport, expanded without requantization.

Format reference: FermionResearch/Phonon-2/fermion_container.py (Apache-2.0).
This reader is independent of the vendored upstream reference reader.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile

import numpy as np

FORMAT = "fermion-five-value-parakeet-v1"


def check_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 mismatch: {actual}, expected {expected}")


def materialize(snapshot: Path) -> Path:
    """Return the extracted config/container directory; never execute Hub code."""
    cfg = json.loads((snapshot / "config.json").read_text())
    if "fermion" in cfg:
        check_sha256(snapshot / "model.fermion", cfg["fermion"]["container_sha256"])
        return snapshot
    if cfg.get("model_type") != "parakeet_tdt_five_value":
        raise ValueError(f"{snapshot}: not a Phonon-2 snapshot")
    artifact = cfg["artifact"]
    name = artifact["filename"]
    if Path(name).name != name:
        raise ValueError("Phonon-2 artifact filename must be a basename")
    archive = snapshot / name
    if not archive.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(cfg["model_id"], name, local_dir=str(snapshot))
    check_sha256(archive, artifact["sha256"])
    dest = snapshot / ".phonon2"
    dest.mkdir(exist_ok=True)
    if not (dest / "config.json").exists() or not (dest / "model.fermion").exists():
        import zstandard
        # Only these two regular files are needed. No extractall, links, paths,
        # or executable files from the archive are materialized.
        wanted = {"config.json", "model.fermion"}
        seen = set()
        with archive.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as rd:
            with tarfile.open(fileobj=rd, mode="r|") as tf:
                for member in tf:
                    if member.name not in wanted:
                        continue
                    if not member.isfile() or member.name in seen:
                        raise ValueError(f"invalid/duplicate Phonon-2 member: {member.name}")
                    limit = artifact["container_bytes"] if member.name == "model.fermion" else 2_000_000
                    if member.size > limit:
                        raise ValueError(f"oversized Phonon-2 member: {member.name}")
                    with tf.extractfile(member) as src, (dest / (member.name + ".partial")).open("wb") as out:
                        shutil.copyfileobj(src, out)
                    (dest / (member.name + ".partial")).replace(dest / member.name)
                    seen.add(member.name)
        if seen != wanted:
            raise ValueError(f"Phonon-2 archive missing {wanted - seen}")
    check_sha256(dest / "model.fermion", artifact["container_sha256"])
    return dest


def expand_five_value(blob: bytes, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("five_value requires a matrix")
    rows, cols = shape
    row_bytes = (cols + 4) // 5
    nsign = rows * row_bytes
    if len(blob) < nsign + 4 * rows:
        raise ValueError("truncated five_value record")
    packed = np.frombuffer(blob, dtype=np.uint8, count=nsign).reshape(rows, row_bytes)
    if np.any(packed >= 243):
        raise ValueError("invalid base-3 byte")
    digits = np.stack([(packed // (3 ** d)) % 3 for d in range(5)], axis=-1)
    signs = digits.reshape(rows, row_bytes * 5)[:, :cols].astype(np.int8) - 1
    nonzero = signs != 0
    nnz = int(nonzero.sum())
    nbits = (nnz + 7) // 8
    if len(blob) != nsign + nbits + 4 * rows:
        raise ValueError("five_value byte count disagrees with shape/codes")
    high_bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8, count=nbits, offset=nsign),
                              bitorder="little")[:nnz]
    high = np.zeros((rows, cols), dtype=bool)
    high[nonzero] = high_bits
    levels = np.frombuffer(blob, dtype="<f2", count=2 * rows, offset=nsign + nbits)
    magnitudes = np.where(high, levels[rows:, None], levels[:rows, None])
    return signs.astype(np.float16) * magnitudes


def expand_int(blob: bytes, shape: tuple[int, ...], bits: int) -> np.ndarray:
    count = math.prod(shape)
    rows = shape[0]
    nbody = ((count + 3) // 4) * 3 if bits == 6 else count
    if bits not in (6, 8) or len(blob) != nbody + 2 * rows:
        raise ValueError("invalid int table byte count/type")
    if bits == 6:
        b = np.frombuffer(blob, dtype=np.uint8, count=nbody).reshape(-1, 3).astype(np.uint32)
        packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        q = np.stack([(packed >> shift) & 63 for shift in (0, 6, 12, 18)], axis=-1)
        q = q.ravel()[:count].astype(np.int32) - 32
    else:
        q = np.frombuffer(blob, dtype=np.int8, count=count)
    scales = np.frombuffer(blob, dtype="<f2", count=rows, offset=nbody).astype(np.float32)
    return (q.reshape(rows, -1).astype(np.float32) * scales[:, None]).reshape(shape)


def iter_tensors(path: Path):
    """Yield HF names and exact expanded values, one record at a time."""
    with path.open("rb") as fh:
        size_bytes = fh.read(8)
        if len(size_bytes) != 8:
            raise ValueError("truncated Fermion header")
        size = int.from_bytes(size_bytes, "little")
        if not 0 < size <= 2_000_000:
            raise ValueError("invalid Fermion header size")
        header = json.loads(fh.read(size))
        if header.get("format") != FORMAT:
            raise ValueError(f"unsupported Fermion format: {header.get('format')}")
        seen = set()
        for entry in header["index"]:
            shape = tuple(entry["shape"])
            if any(type(d) is not int or d <= 0 for d in shape):
                raise ValueError(f"invalid shape: {shape}")
            kind = entry["k"]
            if not shape and kind != "fp16":
                raise ValueError("only fp16 records may be scalar")
            name = entry["n"] + (".weight" if kind == "five_value" else "")
            if name in seen:
                raise ValueError(f"duplicate tensor: {name}")
            seen.add(name)
            length = entry["b"]
            if type(length) is not int or not 0 < length <= 4 * math.prod(shape) + 4 * (shape[0] if shape else 1):
                raise ValueError(f"invalid record length: {name}")
            blob = fh.read(length)
            if len(blob) != length:
                raise ValueError(f"truncated tensor: {name}")
            if kind == "five_value":
                tensor = expand_five_value(blob, shape)
            elif kind in ("int6", "int8"):
                tensor = expand_int(blob, shape, int(kind[3:]))
            elif kind == "fp16" and length == 2 * math.prod(shape):
                tensor = np.frombuffer(blob, dtype="<f2").reshape(shape).copy()
            else:
                raise ValueError(f"unsupported/malformed record: {kind}")
            yield name, tensor
        if fh.read(1):
            raise ValueError("trailing bytes in Fermion container")
