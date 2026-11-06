"""Memory-mapped, document-isolated data and restartable deterministic block streams."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .corpus import SPLITS
from .io import atomic_json, file_sha256, read_jsonl
from .tokenization import Codec


def encode_rows(rows_by_split, codec, tokenizer_path, out_dir, *, test_only=False, source_manifest=None):
    out = Path(out_dir)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.encoded-', dir=out.parent))
    splits = {}
    try:
        shutil.copyfile(tokenizer_path, tmp/'tokenizer.json')
        meta_path = Path(tokenizer_path).with_suffix('.meta.json')
        if meta_path.exists(): shutil.copyfile(meta_path, tmp/'tokenizer.meta.json')
        for split in SPLITS:
            offset = supervised = text_bytes = 0
            groups, papers, hashes, kinds = set(), set(), set(), set()
            with open(tmp/f'{split}.ids.bin','wb') as ids_file, open(tmp/f'{split}.mask.bin','wb') as mask_file, open(tmp/f'{split}.docs.jsonl','w',encoding='utf-8') as docs:
                for row in rows_by_split[split]:
                    if row['kind'] == 'paper':
                        ids = [codec.bos_id]+codec.encode(row['text'])+[codec.eos_id]
                        mask = [0]+[1]*(len(ids)-1)
                    else:
                        prefix = [codec.bos_id]+codec.encode(row['prompt'])+[codec.sep_id]
                        answer = codec.encode(row['response'])+[codec.eos_id]
                        ids, mask = prefix+answer, [0]*len(prefix)+[1]*len(answer)
                    if max(ids) >= codec.vocab_size: raise ValueError('Token ID exceeds vocabulary')
                    np.asarray(ids, dtype='<u2').tofile(ids_file)
                    np.asarray(mask, dtype='u1').tofile(mask_file)
                    doc = {k:row.get(k) for k in ('id','paper_id','group_id','version','kind','source','license','verification','version_verified')}
                    doc.update(offset=offset, length=len(ids))
                    docs.write(json.dumps(doc, ensure_ascii=False)+'\n')
                    offset += len(ids); supervised += sum(mask)
                    text_bytes += sum(codec.byte_lengths[i] for i, v in zip(ids, mask) if v)
                    groups.add(row['group_id']); papers.add(row['paper_id']); kinds.add(row['kind'])
                    if row.get('normalized_sha256'): hashes.add(row['normalized_sha256'])
                    test_only |= bool(row.get('test_only'))
            splits[split] = dict(tokens=offset, supervised_tokens=supervised, text_bytes=text_bytes,
                                 groups=sorted(groups), papers=sorted(papers), text_hashes=sorted(hashes), kinds=sorted(kinds))
        files = {p.name:file_sha256(p) for p in sorted(tmp.iterdir()) if p.is_file()}
        manifest = dict(schema=1, format='map-u16-docs-v1', vocab_size=codec.vocab_size,
                        byte_lengths=list(codec.byte_lengths), test_only=test_only, splits=splits,
                        source_manifest=source_manifest, files=files)
        atomic_json(tmp/'manifest.json', manifest)
        tmp.rename(out)
        return manifest
    finally:
        if tmp.exists(): shutil.rmtree(tmp)


def encode_corpus(corpus_dir, tokenizer_path, out_dir):
    root = Path(corpus_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    for filename, digest in manifest['files'].items():
        if file_sha256(root/filename) != digest: raise ValueError('Corpus hash mismatch')
    tok_path = Path(tokenizer_path)
    meta_path = tok_path.with_suffix('.meta.json')
    if not meta_path.is_file(): raise ValueError('Tokenizer training provenance sidecar is required')
    meta = json.loads(meta_path.read_text())
    if meta['tokenizer_sha256'] != file_sha256(tok_path) or meta['trained_split'] != 'train':
        raise ValueError('Tokenizer hash/provenance mismatch')
    return encode_rows({s:read_jsonl(root/f'{s}.jsonl') for s in SPLITS}, Codec(tok_path), tok_path,
                       out_dir, test_only=manifest['test_only'] or meta.get('test_only',False), source_manifest=manifest)


class EncodedCorpus:
    def __init__(self, root, verify=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('format') != 'map-u16-docs-v1': raise ValueError('Unsupported encoded format')
        self.fingerprint = file_sha256(self.root/'manifest.json')
        if verify:
            for name, expected in self.manifest['files'].items():
                if Path(name).name != name: raise ValueError('Unsafe manifest path')
                if file_sha256(self.root/name) != expected: raise ValueError(f'Dataset integrity/hash mismatch: {name}')
        self.tokenizer_hash = self.manifest['files']['tokenizer.json']
        self.byte_lengths = np.asarray(self.manifest['byte_lengths'], dtype=np.int64)
        if len(self.byte_lengths) != self.manifest['vocab_size']: raise ValueError('Invalid byte-length table')
        self._ids, self._masks = {}, {}
        for split in SPLITS:
            n = self.manifest['splits'][split]['tokens']
            if (self.root/f'{split}.ids.bin').stat().st_size != n*2 or (self.root/f'{split}.mask.bin').stat().st_size != n:
                raise ValueError('Encoded array length mismatch')
            self._ids[split] = np.memmap(self.root/f'{split}.ids.bin', mode='r', dtype='<u2') if n else np.empty(0,dtype='<u2')
            self._masks[split] = np.memmap(self.root/f'{split}.mask.bin', mode='r', dtype='u1') if n else np.empty(0,dtype='u1')

    def identities(self, splits):
        result = set()
        for split in splits:
            d = self.manifest['splits'][split]
            result.update('paper:'+x for x in d['papers'])
            result.update('group:'+x for x in d['groups'])
            result.update('text:'+x for x in d['text_hashes'])
        return result


class BlockDataset:
    def __init__(self, corpus, split, seq_len):
        if split not in SPLITS or seq_len < 1: raise ValueError('Invalid split or sequence length')
        self.corpus, self.split, self.seq_len = corpus, split, seq_len
        self.ids, self.mask = corpus._ids[split], corpus._masks[split]
        self.blocks = []
        for doc in read_jsonl(corpus.root/f'{split}.docs.jsonl'):
            offset, length = doc['offset'], doc['length']
            if offset < 0 or length < 2 or offset+length > len(self.ids): raise ValueError('Invalid document offsets')
            if doc.get('kind') != 'paper' and length > seq_len + 1:
                raise ValueError('Supervised example exceeds context; shorten/section it explicitly, never discard its prompt')
            for start in range(0, length-1, seq_len):
                n = min(seq_len, length-1-start)
                if self.mask[offset+start+1:offset+start+n+1].any():
                    self.blocks.append((offset+start, n))

    def __len__(self): return len(self.blocks)

    def __getitem__(self, index):
        offset, n = self.blocks[index]
        x = np.zeros(self.seq_len, dtype=np.int64)
        y = np.full(self.seq_len, -100, dtype=np.int64)
        x[:n] = self.ids[offset:offset+n]
        targets = np.asarray(self.ids[offset+1:offset+n+1],dtype=np.int64)
        if targets.max() >= self.corpus.manifest['vocab_size']: raise ValueError('Out-of-range token')
        supervised = np.asarray(self.mask[offset+1:offset+n+1],dtype=bool)
        y[:n] = np.where(supervised, targets, -100)
        byte_n = np.zeros(self.seq_len,dtype=np.int64)
        byte_n[:n] = np.where(supervised, self.corpus.byte_lengths[targets], 0)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(byte_n)


class BlockStream:
    def __init__(self, dataset, seed=42):
        if not len(dataset): raise ValueError('No supervised blocks in split; increase context or provide examples')
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset),generator=self.generator)
        self.cursor, self.epoch = 0, 0
        self.pending = None

    def batch(self, size):
        if size < 1: raise ValueError('Batch size must be positive')
        examples = []
        for _ in range(size):
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.dataset),generator=self.generator)
                self.cursor = 0; self.epoch += 1
            examples.append(self.dataset[int(self.order[self.cursor])]); self.cursor += 1
        return tuple(torch.stack(parts) for parts in zip(*examples))

    def take_tokens(self, amount, batch_size):
        if amount < 1: raise ValueError('amount must be positive')
        x, y, b = self.pending if self.pending is not None else self.batch(batch_size)
        positions = (y.flatten() != -100).nonzero().flatten()
        take = min(amount, len(positions))
        if take < len(positions):
            remaining_y, remaining_b = y.clone(), b.clone()
            remaining_y.flatten()[positions[:take]] = -100
            remaining_b.flatten()[positions[:take]] = 0
            self.pending = (x.clone(), remaining_y, remaining_b)
            y, b = y.clone(), b.clone()
            y.flatten()[positions[take:]] = -100
            b.flatten()[positions[take:]] = 0
        else:
            self.pending = None
        return x, y, b

    def state_dict(self):
        return dict(order=self.order.clone(), cursor=self.cursor, epoch=self.epoch, generator=self.generator.get_state(),
                    pending=None if self.pending is None else tuple(t.clone() for t in self.pending))

    def load_state_dict(self, state):
        order = state['order']
        if len(order) != len(self.dataset) or not torch.equal(torch.sort(order).values,torch.arange(len(self.dataset))):
            raise ValueError('Sampler permutation does not match dataset')
        if not 0 <= state['cursor'] <= len(order): raise ValueError('Invalid sampler cursor')
        self.order = order.clone(); self.cursor = state['cursor']; self.epoch = state['epoch']
        self.generator.set_state(state['generator'])
        self.pending = None if state.get('pending') is None else tuple(t.clone() for t in state['pending'])
