"""Original software-test fixtures, not scientific training examples."""
import json
from pathlib import Path
import numpy as np


def paper(i, text=None, split='train', **extra):
    return dict(id=f'p{i}', paper_id=f'paper-{i}', group_id=f'family-{i}', version='1',
                source=f'https://example.invalid/paper-{i}', license='cc-by-4.0',
                kind='paper', split=split, text=text or f'Test document {i}. The variable x is only a software fixture.',
                test_only=True, **extra)


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    return Path(path)


class ByteCodec:
    """Test double for pre-tokenized fixtures; never a production tokenizer fallback."""
    vocab_size = 260
    bos_id, eos_id, pad_id, sep_id = 1, 2, 0, 3
    byte_lengths = [0]*4+[1]*256
    def encode(self, text): return [x+4 for x in text.encode('utf-8')]
    def decode(self, ids): return bytes(i-4 for i in ids if i >= 4).decode('utf-8', errors='replace')


def encoded_fixture(path, kind='paper', n=8):
    from model_as_a_paper.data import encode_rows
    path = Path(path)
    rows = {'train': [], 'validation': [], 'test': []}
    for split, size in [('train', n), ('validation', 2), ('test', 2)]:
        for i in range(size):
            r = paper(f'{split}-{i}', text='alpha beta alpha beta. '+str(i), split=split)
            if kind != 'paper':
                r.update(kind=kind, prompt='What is the test symbol?', response='Title: Fixture\nAbstract: alpha beta.',
                         version_verified=True, verification={'status':'verified', 'method':'fixture', 'evidence':'fixture', 'reviewer':'test'})
                r.pop('text')
            rows[split].append(r)
    tok = path.parent / (path.name + '-test-tokenizer.json')
    tok.write_text('{"test_only":true,"vocab_size":260}', encoding='utf-8')
    encode_rows(rows, ByteCodec(), tok, path, test_only=True)
    return path
