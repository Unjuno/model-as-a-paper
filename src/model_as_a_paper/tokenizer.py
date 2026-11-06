"""Deterministic, reversible byte BPE with no required external tokenizer package.

This is a CPU reference implementation, NOT a throughput claim. Pre-tokenization
preserves whitespace/case and separates individual digits. Training input must
already be restricted to the training split. No normalization is applied.
"""
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re

_PATTERN = re.compile(r'\s+|[^\W\d_]+|\d|[^\w\s]+|_', re.UNICODE)


def pieces(text):
    for match in _PATTERN.finditer(text):
        raw = match.group().encode('utf-8', errors='strict')
        # Bound per-piece work, including pathological whitespace/math strings.
        for start in range(0, len(raw), 64):
            yield raw[start:start + 64]


class ByteBPE:
    pad_id, bos_id, eos_id, sep_id = 256, 257, 258, 259

    def __init__(self, merges=(), capacity=16384):
        if not 260 <= capacity <= 65536:
            raise ValueError('vocabulary capacity must be in [260, 65536]')
        self.capacity = capacity
        self.merges = [tuple(map(int, p)) for p in merges]
        self.tokens = [bytes([i]) for i in range(256)] + [b''] * 4
        self.ranks = {}
        for rank, (a, b) in enumerate(self.merges):
            if min(a, b) < 0 or max(a, b) >= len(self.tokens) or a in range(256, 260) or b in range(256, 260):
                raise ValueError('invalid BPE merge graph')
            if (a, b) in self.ranks:
                raise ValueError('duplicate merge')
            self.ranks[a, b] = rank
            self.tokens.append(self.tokens[a] + self.tokens[b])
        if len(self.tokens) > capacity:
            raise ValueError('merge count exceeds vocabulary capacity')

    @staticmethod
    def _merge(word, pair, new_id):
        result = []; i = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                result.append(new_id); i += 2
            else:
                result.append(word[i]); i += 1
        return tuple(result)

    @classmethod
    def train(cls, texts, vocab_size=16384, min_frequency=2, max_bytes=8 * 1024 * 1024):
        cls(capacity=vocab_size)
        if min_frequency < 1 or max_bytes < 1:
            raise ValueError('positive frequency and byte limit required')
        counts = Counter(); used = 0
        for text in texts:
            for piece in pieces(text):
                if used + len(piece) > max_bytes: break
                counts[tuple(piece)] += 1; used += len(piece)
            if used >= max_bytes - 64: break
        if not counts:
            raise ValueError('no tokenizer training text')
        words = list(counts)
        weights = list(counts.values())
        pair_count = Counter(); where = defaultdict(set)
        for i, word in enumerate(words):
            for pair, num in Counter(zip(word, word[1:])).items():
                pair_count[pair] += num * weights[i]; where[pair].add(i)
        heap = [(-n, p) for p, n in pair_count.items()]; heapq.heapify(heap)
        merges = []
        while heap and 260 + len(merges) < vocab_size:
            neg, pair = heapq.heappop(heap)
            if -neg != pair_count[pair]: continue
            if -neg < min_frequency: break
            new_id = 260 + len(merges); merges.append(pair)
            touched = set()
            for i in sorted(where[pair]):
                old = words[i]; new = cls._merge(old, pair, new_id)
                old_pairs = Counter(zip(old, old[1:])); new_pairs = Counter(zip(new, new[1:]))
                for p in old_pairs.keys() | new_pairs.keys():
                    pair_count[p] += (new_pairs[p] - old_pairs[p]) * weights[i]
                    if new_pairs[p]: where[p].add(i)
                    else: where[p].discard(i)
                    touched.add(p)
                words[i] = new
            for p in touched:
                if pair_count[p] > 0: heapq.heappush(heap, (-pair_count[p], p))
        return cls(merges, vocab_size)

    def encode(self, text):
        result = []
        for piece in pieces(text):
            word = tuple(piece)
            while len(word) > 1:
                choices = ((self.ranks[p], p) for p in zip(word, word[1:]) if p in self.ranks)
                selected = min(choices, default=None)
                if selected is None: break
                rank, pair = selected
                word = self._merge(word, pair, 260 + rank)
            result.extend(word)
        return result

    def decode(self, ids, errors='strict'):
        if any(i < 0 or i >= len(self.tokens) for i in ids):
            raise ValueError('unknown or unused token ID')
        return b''.join(self.tokens[i] for i in ids).decode('utf-8', errors=errors)

    def to_dict(self):
        return {'format': 'map-byte-bpe-v1', 'capacity': self.capacity,
                'pretokenizer': 'unicode-word-digit-space-symbol;max64bytes;no-normalization',
                'merges': self.merges}

    @property
    def fingerprint(self):
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('format') != 'map-byte-bpe-v1':
            raise ValueError('unsupported tokenizer format')
        return cls(data['merges'], data['capacity'])
