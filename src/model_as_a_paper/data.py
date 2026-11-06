"""Bounded JSONL preparation, immutable memmap corpus and resumable sampling.

One window never crosses documents. Right padding does not require a dense
attention mask: valid causal positions cannot attend to padding on their right.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from .tokenizer import ByteBPE

DEFAULT_LICENSES = ('http://creativecommons.org/licenses/by/4.0/',
                    'https://creativecommons.org/licenses/by/4.0/',
                    'http://creativecommons.org/publicdomain/zero/1.0/',
                    'https://creativecommons.org/publicdomain/zero/1.0/')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def canonical_id(value):
    return re.sub(r'v\d+$', '', value.strip())


def prepare(input_path, output, vocab_size=16384, tokenizer_path=None, allow_test_data=False,
            allowed_licenses=DEFAULT_LICENSES, min_chars=1000, split=None,
            max_input_bytes=256 * 1024 * 1024, tokenizer_bytes=8 * 1024 * 1024):
    """Create a new corpus. `split` override is for separately curated update/eval sets."""
    input_path, output = Path(input_path), Path(output)
    if output.exists(): raise FileExistsError(f'refusing to overwrite {output}')
    if split not in (None, 'train', 'validation', 'test'):
        raise ValueError('unknown split')
    if min_chars < 1 or max_input_bytes < 1: raise ValueError('invalid size bounds')
    if input_path.stat().st_size > max_input_bytes:
        raise ValueError('input exceeds configured memory bound; prepare a smaller selection')
    rows = []; parents = {}; seen_text = {}; rejected = 0; any_test = False

    def root(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]; x = parents[x]
        return x

    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)

    with input_path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            r = json.loads(line)
            if not isinstance(r, dict) or not isinstance(r.get('paper_id'), str) or not r['paper_id'].strip():
                raise ValueError(f'line {line_no}: paper_id required')
            test = r.get('source_kind') == 'test_fixture'
            if test and not allow_test_data: raise ValueError('test fixture requires explicit opt-in')
            if not test and r.get('license') not in allowed_licenses:
                rejected += 1; continue
            kind = r.get('kind', 'paper')
            if kind not in ('paper', 'sft', 'correction'): raise ValueError('unknown record kind')
            if kind == 'paper':
                text = r.get('text')
                if not isinstance(text, str): raise ValueError('paper text required')
            else:
                if not (isinstance(r.get('prompt'), str) and isinstance(r.get('answer'), str) and r['answer'].strip()):
                    raise ValueError('nonempty answer and string prompt required')
                if r.get('verified') is not True or not isinstance(r.get('verification'), str) or not r['verification'].strip() or not r.get('evidence'):
                    raise ValueError('paper-derived supervision requires verification and evidence')
                evidence = r['evidence']
                if not isinstance(evidence, list) or any(not isinstance(e, dict) or not isinstance(e.get('paper_id'), str) or not e['paper_id'].strip() or not e.get('version') or not isinstance(e.get('span'), str) or not e['span'].strip() for e in evidence):
                    raise ValueError('evidence requires paper_id, version and source span')
                if kind == 'correction' and not r.get('version'):
                    raise ValueError('correction requires version provenance')
                text = r['prompt'] + '\n' + r['answer']
            if len(text) < min_chars or '\x00' in text:
                rejected += 1; continue
            if kind == 'paper' and (text.strip() == '%auto-ignore' or ('\\includepdf' in text and len(text) < 1000)):
                rejected += 1; continue
            r = dict(r, kind=kind, source_sha256=hashlib.sha256(text.encode()).hexdigest())
            group = str(r.get('group_id') or canonical_id(r['paper_id']))
            key = 'paper:' + canonical_id(r['paper_id']); union(key, 'group:' + group)
            if r['source_sha256'] in seen_text:
                union(key, seen_text[r['source_sha256']])
            else: seen_text[r['source_sha256']] = key
            r['_key'] = key; rows.append(r); any_test |= test
    if not rows: raise ValueError('no accepted documents (check licenses and filtering)')
    # Union before assigning splits prevents duplicate text or revisions leaking.
    dedup = set(); selected = []
    for r in rows:
        group = root(r['_key'])
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
        r['split'] = split or ('validation' if bucket < 5 else 'test' if bucket < 10 else 'train')
        r['group_id'] = group
        if r['source_sha256'] in dedup: continue
        dedup.add(r['source_sha256']); selected.append(r)
    train_rows = [r for r in selected if r['split'] == 'train']
    if tokenizer_path is None:
        if not train_rows: raise ValueError('no training split available for tokenizer')
        tok = ByteBPE.train((r.get('text', r.get('prompt', '') + '\n' + r.get('answer', ''))
                             for r in train_rows), vocab_size=vocab_size, max_bytes=tokenizer_bytes)
        training_ids = [r['paper_id'] for r in train_rows]
    else:
        tok = ByteBPE.load(tokenizer_path); training_ids = []
    output.mkdir(parents=True)
    tok.save(output/'tokenizer.json')
    records = []; offset = 0
    with (output/'tokens.bin').open('wb') as ft, (output/'mask.bin').open('wb') as fm, (output/'bytes.bin').open('wb') as fb:
        for r in selected:
            if r['kind'] == 'paper':
                ids = [tok.bos_id] + tok.encode(r['text']) + [tok.eos_id]
                masks = [0] + [1] * (len(ids) - 1)
            else:
                question = [tok.bos_id] + tok.encode(r['prompt']) + [tok.sep_id]
                answer = tok.encode(r['answer']) + [tok.eos_id]
                ids = question + answer; masks = [0] * len(question) + [1] * len(answer)
            np.array(ids, dtype='<u2').tofile(ft)
            np.array(masks, dtype='u1').tofile(fm)
            np.array([len(tok.tokens[i]) for i in ids], dtype='<u2').tofile(fb)
            record = {k: v for k, v in r.items() if k not in ('text', 'prompt', 'answer', '_key')}
            record.update(start=offset, end=offset + len(ids)); records.append(record); offset += len(ids)
    (output/'records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    files = {name: digest(output/name) for name in ('tokens.bin', 'mask.bin', 'bytes.bin', 'records.json', 'tokenizer.json')}
    manifest = {'format': 'map-corpus-v1', 'files': files, 'tokenizer_hash': tok.fingerprint,
                'vocab_size': tok.capacity, 'actual_vocab_size': len(tok.tokens),
                'tokens': offset, 'test_only': any_test, 'rejected': rejected,
                'input_sha256': digest(input_path), 'tokenizer_training_paper_ids': training_ids,
                'split_counts': {s: sum(r['split'] == s for r in records) for s in ('train', 'validation', 'test')}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class Corpus:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path/'manifest.json').read_text())
        if self.manifest.get('format') != 'map-corpus-v1': raise ValueError('unsupported corpus format')
        required = ('tokens.bin', 'mask.bin', 'bytes.bin', 'records.json', 'tokenizer.json')
        if set(self.manifest['files']) != set(required): raise ValueError('unexpected corpus files')
        for name in required:
            if digest(self.path/name) != self.manifest['files'][name]: raise ValueError(f'hash mismatch: {name}')
        self.fingerprint = digest(self.path/'manifest.json')
        self.tokenizer = ByteBPE.load(self.path/'tokenizer.json')
        if self.tokenizer.fingerprint != self.manifest['tokenizer_hash']: raise ValueError('tokenizer hash mismatch')
        self.tokens = np.memmap(self.path/'tokens.bin', mode='r', dtype='<u2')
        self.mask = np.memmap(self.path/'mask.bin', mode='r', dtype='u1')
        self.sizes = np.memmap(self.path/'bytes.bin', mode='r', dtype='<u2')
        self.records = json.loads((self.path/'records.json').read_text())
        n = self.manifest['tokens']
        if len(self.tokens) != n or len(self.mask) != n or len(self.sizes) != n:
            raise ValueError('corpus array length mismatch')
        last = 0
        for r in self.records:
            if r['start'] != last or not last < r['end'] <= n: raise ValueError('invalid document offsets')
            last = r['end']
        if last != n: raise ValueError('unindexed tokens')


class WindowDataset:
    def __init__(self, corpus, split, seq_len):
        if seq_len < 1: raise ValueError('seq_len must be positive')
        self.corpus, self.seq_len = corpus, seq_len
        self.windows = []
        for r in corpus.records:
            if r['split'] != split: continue
            if r['kind'] != 'paper' and r['end'] - r['start'] > seq_len + 1:
                raise ValueError('supervised record exceeds context; shorten prompt/answer or increase context')
            for start in range(r['start'], r['end'] - 1, seq_len):
                end = min(start + seq_len + 1, r['end'])
                if corpus.mask[start + 1:end].any(): self.windows.append((start, end))
        self.identity = f'{corpus.fingerprint}:{split}:{seq_len}'

    def __len__(self): return len(self.windows)

    def __getitem__(self, index):
        start, end = self.windows[index]; n = end - start - 1
        x = np.full(self.seq_len, self.corpus.tokenizer.pad_id, dtype=np.int64)
        y = np.full(self.seq_len, -100, dtype=np.int64)
        sizes = np.zeros(self.seq_len, dtype=np.int64)
        x[:n] = self.corpus.tokens[start:end-1]
        y[:n] = self.corpus.tokens[start+1:end]
        y[:n][self.corpus.mask[start+1:end] == 0] = -100
        sizes[:n] = self.corpus.sizes[start+1:end]
        sizes[y == -100] = 0
        return x, y, sizes


class StatefulStream:
    def __init__(self, dataset, seed):
        if not len(dataset): raise ValueError('empty training/evaluation split')
        self.dataset = dataset
        self.rng = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.rng)
        self.cursor = 0; self.epoch = 0

    def next(self):
        if self.cursor == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.rng)
            self.cursor = 0; self.epoch += 1
        i = int(self.order[self.cursor]); self.cursor += 1
        return self.dataset[i]

    def state_dict(self):
        return dict(identity=self.dataset.identity, rng=self.rng.get_state(),
                    order=self.order.clone(), cursor=self.cursor, epoch=self.epoch)

    def load_state_dict(self, state):
        if state['identity'] != self.dataset.identity: raise ValueError('sampler corpus/context changed')
        if not 0 <= state['cursor'] <= len(self.dataset): raise ValueError('bad sampler cursor')
        if sorted(state['order'].tolist()) != list(range(len(self.dataset))): raise ValueError('invalid permutation')
        self.rng.set_state(state['rng']); self.order = state['order'].clone()
        self.cursor = state['cursor']; self.epoch = state['epoch']
