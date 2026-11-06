"""Hashing and atomic metadata writes. Paths are caller-owned, never executable."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def object_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{number}: invalid JSON') from e
            if not isinstance(row, dict): raise ValueError(f'{path}:{number}: expected an object')
            yield row
