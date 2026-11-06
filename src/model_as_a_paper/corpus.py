"""Provenance/split gate. Verification metadata is an attestation, not a proof."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

from .io import atomic_json, file_sha256, read_jsonl

SPLITS = ('train', 'validation', 'test')
LICENSES = frozenset({'cc-by-4.0', 'cc0-1.0'})
SPECIALS = ('<|pad|>', '<|bos|>', '<|eos|>', '<|answer|>')


def normalize_license(value):
    if not isinstance(value, str): return None
    value = value.strip().lower().rstrip('/')
    return {
        'http://creativecommons.org/licenses/by/4.0': 'cc-by-4.0',
        'https://creativecommons.org/licenses/by/4.0': 'cc-by-4.0',
        'http://creativecommons.org/publicdomain/zero/1.0': 'cc0-1.0',
        'https://creativecommons.org/publicdomain/zero/1.0': 'cc0-1.0',
    }.get(value, value)


def base_paper_id(value):
    return re.sub(r'v\d+$', '', str(value).removeprefix('arXiv:'))


def assigned_split(group_id, seed=42):
    bucket = int(hashlib.sha256(f'{seed}:{group_id}'.encode()).hexdigest()[:8], 16) / 2**32
    return 'train' if bucket < 0.90 else ('validation' if bucket < 0.95 else 'test')


def validate_record(row, allow_test_data=False):
    row = dict(row)
    if row.get('test_only') and not allow_test_data:
        raise ValueError('test_only fixture cannot enter a production corpus')
    for name in ('id', 'paper_id', 'source'):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f'Missing provenance field: {name}')
    if not row['source'].startswith(('https://', 'http://', 'doi:')):
        raise ValueError('source must identify a paper by HTTP(S) or DOI')
    kind = row.setdefault('kind', 'paper')
    if kind not in ('paper', 'sft', 'correction'):
        raise ValueError(f'Unsupported kind: {kind}')
    for name in (('text',) if kind == 'paper' else ('prompt', 'response')):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f'{kind} requires nonempty {name}')
        if any(s in row[name] for s in SPECIALS):
            raise ValueError(f'Reserved control token in {name}')
    if kind != 'paper':
        v = row.get('verification', {})
        if not isinstance(v, dict) or v.get('status') != 'verified' or not all(
            isinstance(v.get(k), str) and v[k].strip() for k in ('method', 'evidence', 'reviewer')
        ):
            raise ValueError('Derived examples require explicit independent verification metadata')
    if kind == 'correction' and (row.get('version_verified') is not True or not row.get('version')):
        raise ValueError('Corrections require version_verified=true and a known version')
    row['paper_id'] = base_paper_id(row['paper_id'])
    row['group_id'] = row.get('group_id') or row['paper_id']
    if not isinstance(row['group_id'], str): raise ValueError('group_id must be text')
    row['license'] = normalize_license(row.get('license'))
    return row


def prepare_corpus(input_path, out_dir, *, min_chars=1000, licenses=LICENSES, seed=42, allow_test_data=False):
    """Stream records into group-stable splits; reject cross-split identity conflicts.

    Exact normalized duplicate conflicts fail closed. Near-duplicates and scientific
    correctness still require a separate audit. No TeX code or macros are executed.
    """
    out = Path(out_dir)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    if min_chars < 1: raise ValueError('min_chars must be >= 1')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.corpus-', dir=out.parent))
    conn = sqlite3.connect(tmp/'dedup.sqlite')
    conn.executescript('CREATE TABLE identities (key TEXT PRIMARY KEY, split TEXT, family TEXT); CREATE TABLE texts (hash TEXT PRIMARY KEY, family TEXT); CREATE TABLE ids (id TEXT PRIMARY KEY);')
    counts = {s:0 for s in SPLITS}; rejected = Counter(); test_only = False
    handles = {s: open(tmp/f'{s}.jsonl', 'w', encoding='utf-8') for s in SPLITS}
    try:
        for raw in read_jsonl(input_path):
            row = validate_record(raw, allow_test_data)
            if row['license'] not in licenses:
                rejected['license'] += 1; continue
            text = row['text'] if row['kind'] == 'paper' else row['prompt']+'\n'+row['response']
            if len(text) < min_chars:
                rejected['short'] += 1; continue
            if row['kind'] == 'paper' and text.strip() == '%auto-ignore':
                rejected['stub'] += 1; continue
            split = row.get('split') or assigned_split(row['group_id'], seed)
            if split not in SPLITS: raise ValueError(f'Invalid split: {split}')
            for key in ('paper:'+row['paper_id'], 'group:'+row['group_id']):
                old = conn.execute('SELECT split,family FROM identities WHERE key=?', (key,)).fetchone()
                if old and old != (split, row['group_id']):
                    raise ValueError(f'Cross-split/group family conflict: {key}')
                conn.execute('INSERT OR IGNORE INTO identities VALUES (?,?,?)', (key, split, row['group_id']))
            digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
            old = conn.execute('SELECT family FROM texts WHERE hash=?', (digest,)).fetchone()
            if old:
                if old[0] != row['group_id']: raise ValueError('Exact duplicate across groups: reconcile families before preparing')
                rejected['duplicate'] += 1; continue
            try: conn.execute('INSERT INTO ids VALUES (?)', (row['id'],))
            except sqlite3.IntegrityError as e: raise ValueError(f'Duplicate id: {row["id"]}') from e
            conn.execute('INSERT INTO texts VALUES (?,?)', (digest, row['group_id']))
            row['split'], row['normalized_sha256'] = split, digest
            handles[split].write(json.dumps(row, ensure_ascii=False)+'\n')
            counts[split] += 1; test_only |= bool(row.get('test_only'))
        if not counts['train']: raise ValueError('No eligible training records after filtering')
        for f in handles.values(): f.close()
        conn.close(); (tmp/'dedup.sqlite').unlink()
        manifest = dict(schema=1, counts=counts, rejected=dict(rejected), seed=seed,
                        license_allowlist=sorted(licenses), test_only=test_only,
                        source_sha256=file_sha256(input_path),
                        files={s+'.jsonl':file_sha256(tmp/f'{s}.jsonl') for s in SPLITS})
        atomic_json(tmp/'manifest.json', manifest)
        tmp.rename(out)
        return manifest
    finally:
        for f in handles.values():
            if not f.closed: f.close()
        conn.close()
        if tmp.exists(): shutil.rmtree(tmp)
