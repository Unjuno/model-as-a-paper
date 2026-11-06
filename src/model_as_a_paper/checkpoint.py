"""Atomic, hashed checkpoint pointers; only built-in types/tensors are loaded."""
import json
import os
from pathlib import Path
import tempfile
import uuid

import torch

from .io import atomic_json, file_sha256


def save_checkpoint(root, payload, *, best=False, commit_callback=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    filename = f'step-{payload["state"]["step"]:08d}-{uuid.uuid4().hex[:8]}.pt'
    fd, tmp = tempfile.mkstemp(prefix='.checkpoint-', dir=root)
    os.close(fd)
    try:
        with open(tmp, 'wb') as f:
            torch.save(payload, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, root/filename)
        pointer = dict(schema=1, file=filename, sha256=file_sha256(root/filename), step=payload['state']['step'])
        if (root/'latest.json').exists():
            atomic_json(root/'previous.json', json.loads((root/'latest.json').read_text()))
        atomic_json(root/'latest.json',pointer)
        if best: atomic_json(root/'best.json',pointer)
        # Keep all files referenced by these three pointers. Never delete the parent run.
        keep = {json.loads(p.read_text())['file'] for p in [root/'latest.json',root/'previous.json',root/'best.json'] if p.exists()}
        for p in root.glob('step-*.pt'):
            if p.name not in keep: p.unlink()
        if commit_callback is not None: commit_callback()
        return pointer
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_checkpoint(path):
    path = Path(path)
    if path.is_dir(): path = path/'latest.json'
    if path.suffix != '.json':
        raise ValueError('Load a hashed checkpoint pointer (.json) or its run directory, not an unverified .pt')
    pointer = json.loads(path.read_text())
    filename = pointer.get('file','')
    if not filename or Path(filename).name != filename or not filename.endswith('.pt'):
        raise ValueError('Unsafe checkpoint pointer')
    target = path.parent/filename
    if file_sha256(target) != pointer.get('sha256'):
        raise ValueError('Checkpoint integrity/hash mismatch')
    payload = torch.load(target, map_location='cpu', weights_only=True)
    if payload.get('schema') != 1: raise ValueError('Unsupported checkpoint schema')
    return payload


def restore_weights(model, weights):
    weights = dict(weights)
    weights['lm_head.weight'] = weights['embedding.weight']
    model.load_state_dict(weights, strict=True)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value
