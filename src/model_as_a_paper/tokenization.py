"""Reversible byte-level BPE; trained from training-paper TEXT only, never a model."""
import json
from pathlib import Path

from .corpus import SPECIALS
from .io import atomic_json, file_sha256, read_jsonl


class Codec:
    bos_id, eos_id, pad_id, sep_id = 1, 2, 0, 3

    def __init__(self, path=None, *, json_text=None):
        from tokenizers import Tokenizer
        if (path is None) == (json_text is None): raise ValueError('Supply exactly one tokenizer path or JSON string')
        self.tokenizer = Tokenizer.from_str(json_text) if json_text is not None else Tokenizer.from_file(str(path))
        for i, s in enumerate(SPECIALS):
            if self.tokenizer.token_to_id(s) != i: raise ValueError('Unexpected special token IDs')
        self.vocab_size = self.tokenizer.get_vocab_size()
        # ByteLevel BPE represents each raw byte by one Unicode alphabet glyph.
        # len(token), NOT len(decode([id]).encode()), is the original byte count.
        self.byte_lengths = [0 if i < 4 else len(self.tokenizer.id_to_token(i)) for i in range(self.vocab_size)]

    def encode(self, text):
        if any(s in text for s in SPECIALS): raise ValueError('Reserved control token in input')
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def train_tokenizer(corpus_dir, out_path, *, vocab_size=16384, min_frequency=2):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    corpus = Path(corpus_dir); out = Path(out_path)
    if out.exists(): raise FileExistsError(f'Refusing to replace tokenizer: {out}')
    if not 260 <= vocab_size <= 65535: raise ValueError('vocab_size must be 260..65535')
    manifest = json.loads((corpus/'manifest.json').read_text())
    if file_sha256(corpus/'train.jsonl') != manifest['files']['train.jsonl']:
        raise ValueError('Training corpus integrity mismatch')
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                  special_tokens=list(SPECIALS), initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    def texts():
        for row in read_jsonl(corpus/'train.jsonl'):
            if row['kind'] != 'paper': raise ValueError('Fit tokenizer on original training papers, not SFT/corrections')
            yield row['text']
    tok.train_from_iterator(texts(), trainer=trainer)
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(f'Corpus too small for requested vocabulary: got {tok.get_vocab_size()}, need {vocab_size}. Add training papers or use an explicitly smaller test config.')
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    atomic_json(out.with_suffix('.meta.json'), dict(
        schema=1, trained_split='train', trained_source='paper',
        source_train_sha256=manifest['files']['train.jsonl'],
        tokenizer_sha256=file_sha256(out), vocab_size=vocab_size,
        test_only=manifest['test_only'], normalization='none', add_prefix_space=False,
    ))
    return out
