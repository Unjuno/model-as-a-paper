"""Teacher-forced training, token-aware replay, safe checkpoints and evaluation."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from .budget import run_seconds
from .data import Corpus, StatefulStream, WindowDataset, digest, canonical_id
from .model import ModelConfig, PaperLM


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 1024
    long_seq_len: int | None = 2048
    switch_after_tokens: int = 400_000_000
    batch_tokens: int = 32768
    micro_batch_size: int = 8
    learning_rate: float = 0.0006
    min_learning_rate: float = 0.00006
    warmup_tokens: int = 10_000_000
    cooldown_start: int | None = None
    cooldown_tokens: int = 50_000_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    replay_ratio: float = 0.0
    seed: int = 17
    checkpoint_steps: int = 100
    checkpoint_seconds: float = 120.0
    max_input_positions_per_update: int = 2_000_000
    precision: str = 'bf16'

    def __post_init__(self):
        for n in ('seq_len', 'batch_tokens', 'micro_batch_size', 'checkpoint_steps',
                  'cooldown_tokens', 'max_input_positions_per_update'):
            if type(getattr(self, n)) is not int or getattr(self, n) <= 0:
                raise ValueError(f'{n} must be a positive integer')
        if self.long_seq_len is not None and self.long_seq_len < self.seq_len:
            raise ValueError('long context must not be shorter')
        if self.warmup_tokens < 0 or self.switch_after_tokens < 0:
            raise ValueError('negative token schedule')
        if self.cooldown_start is not None and self.cooldown_start < self.warmup_tokens:
            raise ValueError('cooldown overlaps warmup')
        for n in ('learning_rate', 'min_learning_rate', 'grad_clip', 'checkpoint_seconds'):
            if not math.isfinite(getattr(self, n)) or getattr(self, n) <= 0:
                raise ValueError(f'{n} must be finite and positive')
        if not 0 <= self.replay_ratio < 1 or not math.isfinite(self.replay_ratio):
            raise ValueError('replay_ratio must be in [0, 1)')
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay): raise ValueError('bad decay')
        if self.precision not in ('fp32', 'bf16'): raise ValueError('only fp32/bf16 supported')

    def to_dict(self): return asdict(self)

    def lr_at(self, tokens):
        if self.warmup_tokens and tokens < self.warmup_tokens:
            return self.learning_rate * tokens / self.warmup_tokens
        if self.cooldown_start is not None and tokens > self.cooldown_start:
            progress = min(1.0, (tokens - self.cooldown_start) / self.cooldown_tokens)
            return self.learning_rate + progress * (self.min_learning_rate - self.learning_rate)
        return self.learning_rate


def autocast_for(device, precision):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if torch.device(device).type == 'cuda' and precision == 'bf16' else nullcontext())


def collect_holdouts(*corpora):
    result = {'paper_ids': set(), 'group_ids': set(), 'source_hashes': set()}
    for corpus in corpora:
        if corpus is None: continue
        for record in corpus.records:
            if record['split'] == 'train': continue
            result['paper_ids'].add(canonical_id(record['paper_id']))
            result['group_ids'].add(record['group_id'])
            result['source_hashes'].add(record['source_sha256'])
    return {key: sorted(values) for key, values in result.items()}


def protect_holdouts(corpus, protected):
    if corpus is None: return
    papers, groups, hashes = (set(protected[k]) for k in ('paper_ids', 'group_ids', 'source_hashes'))
    for r in corpus.records:
        if r['split'] != 'train': continue
        ids = {canonical_id(r['paper_id'])}
        ids.update(canonical_id(e['paper_id']) for e in r.get('evidence', []))
        if ids & papers or r['group_id'] in groups or r['source_sha256'] in hashes:
            raise ValueError('held-out paper/group/content would enter training or replay')


def merge_holdouts(a, b):
    return {key: sorted(set(a[key]) | set(b[key])) for key in a}


class Trainer:
    def __init__(self, corpus, model_cfg, cfg, device='cpu', replay=None, allow_test_data=False):
        if corpus.manifest['test_only'] and not allow_test_data:
            raise ValueError('test-only corpus is forbidden without explicit opt-in')
        if model_cfg.vocab_size != corpus.tokenizer.capacity: raise ValueError('vocabulary mismatch')
        if max(cfg.seq_len, cfg.long_seq_len or cfg.seq_len) > model_cfg.max_seq_len:
            raise ValueError('training context exceeds model limit')
        if bool(cfg.replay_ratio) != (replay is not None): raise ValueError('replay corpus/ratio mismatch')
        if replay is not None:
            if replay.tokenizer.fingerprint != corpus.tokenizer.fingerprint:
                raise ValueError('replay tokenizer mismatch')
            if replay.manifest['test_only'] and not allow_test_data: raise ValueError('test-only replay')
        self.heldouts = collect_holdouts(corpus, replay)
        protect_holdouts(corpus, self.heldouts); protect_holdouts(replay, self.heldouts)
        self.corpus, self.replay, self.cfg = corpus, replay, cfg
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'): raise ValueError('only cpu/cuda supported')
        if self.device.type == 'cuda':
            if not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
            if cfg.precision == 'bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('GPU lacks BF16 support')
        random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        self.model = PaperLM(model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.optimizer_groups(cfg.weight_decay),
                                          lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)
        self.counters = dict(steps=0, tokens=0, new_tokens=0, replay_tokens=0)
        self.streams = {}; self.parent_checkpoint_sha256 = None
        self.test_only = corpus.manifest['test_only'] or (replay is not None and replay.manifest['test_only'])

    def _stream(self, source, length):
        key = f'{source}:{length}'
        if key not in self.streams:
            data = self.corpus if source == 'new' else self.replay
            self.streams[key] = StatefulStream(WindowDataset(data, 'train', length),
                                              self.cfg.seed + length + (1 if source == 'replay' else 0))
        return self.streams[key]

    def step(self, target_tokens=None, new_token_budget=None):
        cfg = self.cfg
        target = cfg.batch_tokens if target_tokens is None else min(cfg.batch_tokens, target_tokens)
        if target < 1: raise ValueError('positive target count required')
        length = (cfg.long_seq_len if cfg.long_seq_len and self.counters['tokens'] >= cfg.switch_after_tokens
                  else cfg.seq_len)
        examples = []; added = 0; new = self.counters['new_tokens']; replay = self.counters['replay_tokens']
        while added < target:
            if new_token_budget is not None and new - self.counters['new_tokens'] >= new_token_budget:
                break
            source = ('replay' if self.replay is not None and replay < cfg.replay_ratio * (new + replay) else 'new')
            x, y, sizes = self._stream(source, length).next()
            active = np.flatnonzero(y != -100)
            take = min(len(active), target - added)
            if source == 'new' and new_token_budget is not None:
                take = min(take, new_token_budget - (new - self.counters['new_tokens']))
            y[active[take:]] = -100; sizes[active[take:]] = 0
            examples.append((x, y))
            if len(examples) * length > cfg.max_input_positions_per_update:
                raise ValueError('too many masked input positions; lower batch_tokens or shorten prompts')
            added += take
            if source == 'new': new += take
            else: replay += take
        if added < 1: raise ValueError('no remaining new target tokens')
        lr = cfg.lr_at(self.counters['tokens'] + added)
        for group in self.optimizer.param_groups: group['lr'] = lr
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train(); loss_sum = 0.0
        # Normalize every microbatch SUM by the whole update's target count.
        # A mean of per-microbatch means is wrong for padding/SFT/replay.
        for start in range(0, len(examples), cfg.micro_batch_size):
            chunk = examples[start:start + cfg.micro_batch_size]
            x = torch.from_numpy(np.stack([e[0] for e in chunk])).to(self.device)
            y = torch.from_numpy(np.stack([e[1] for e in chunk])).to(self.device)
            with autocast_for(self.device, cfg.precision):
                logits, _ = self.model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                       ignore_index=-100, reduction='sum')
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss; last good checkpoint retained')
            (loss / added).backward(); loss_sum += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.counters.update(steps=self.counters['steps'] + 1, tokens=self.counters['tokens'] + added,
                             new_tokens=new, replay_tokens=replay)
        return dict(**self.counters, loss=loss_sum / added, learning_rate=lr,
                    gradient_norm=float(norm), seq_len=length, input_positions=len(examples) * length)

    def state_dict(self):
        return dict(format='map-checkpoint-v1', model_config=self.model.cfg.to_dict(),
                    train_config=self.cfg.to_dict(), model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(), counters=self.counters.copy(),
                    corpora={'new': self.corpus.fingerprint,
                             'replay': None if self.replay is None else self.replay.fingerprint},
                    tokenizer_hash=self.corpus.tokenizer.fingerprint,
                    streams={k: s.state_dict() for k, s in self.streams.items()},
                    torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
                    heldouts=self.heldouts, parent_checkpoint_sha256=self.parent_checkpoint_sha256,
                    test_only=bool(self.test_only), runtime={'torch': str(torch.__version__),
                    'device': str(self.device), 'cuda': str(torch.version.cuda)})

    def load_state_dict(self, state):
        if state['model_config'] != self.model.cfg.to_dict() or state['train_config'] != self.cfg.to_dict():
            raise ValueError('resume config mismatch; use init-from for a new training objective')
        if state['tokenizer_hash'] != self.corpus.tokenizer.fingerprint: raise ValueError('tokenizer mismatch')
        expected = {'new': self.corpus.fingerprint, 'replay': None if self.replay is None else self.replay.fingerprint}
        if state['corpora'] != expected: raise ValueError('resume corpus mismatch')
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self.counters = state['counters'].copy()
        for key, value in state['streams'].items():
            source, length = key.split(':'); self._stream(source, int(length)).load_state_dict(value)
        torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng'])
        if self.device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.heldouts = merge_holdouts(self.heldouts, state['heldouts'])
        self.parent_checkpoint_sha256 = state['parent_checkpoint_sha256']
        self.test_only = self.test_only or state['test_only']


def atomic_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(worker, output):
    output = Path(output)
    filename = f"step-{worker.counters['steps']:09d}-{uuid.uuid4().hex[:8]}.pt"
    tmp = output / (filename + '.tmp')
    with tmp.open('wb') as f:
        torch.save(worker.state_dict(), f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, output/filename)
    pointer = {'file': filename, 'sha256': digest(output/filename)}
    latest = output/'latest.json'
    if latest.exists(): atomic_json(output/'previous.json', json.loads(latest.read_text()))
    # Immutable blobs + atomic pointer avoid torn checkpoint/hash pairs.
    atomic_json(latest, pointer)
    keep = {filename}
    if (output/'previous.json').exists(): keep.add(json.loads((output/'previous.json').read_text())['file'])
    for old in output.glob('step-*.pt'):
        if old.name not in keep: old.unlink()
    return pointer


def read_checkpoint(path):
    path = Path(path)
    pointer_path = path/'latest.json' if path.is_dir() else path
    info = json.loads(pointer_path.read_text())
    filename = info['file']
    if Path(filename).name != filename or not filename.endswith('.pt'): raise ValueError('invalid checkpoint path')
    ckpt = pointer_path.parent/filename
    if digest(ckpt) != info['sha256']: raise ValueError('checkpoint hash mismatch')
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if state.get('format') != 'map-checkpoint-v1': raise ValueError('unsupported checkpoint format')
    state['_sha256'] = info['sha256']
    return state


def train(config, corpus_path, output, *, steps=100, token_limit=500_000_000, seconds=600,
          budget_usd=None, hourly_rate=None, device='cpu', resume=False, init_from=None,
          replay=None, allow_test_data=False, on_save=None, new_token_limit=None, start_cooldown=False):
    """Bounded invocation. Resume preserves state; init_from starts a new objective."""
    started = time.monotonic()
    allowance = run_seconds(seconds, budget_usd, hourly_rate, reserve=min(60.0, seconds * 0.1))
    output = Path(output)
    if type(steps) is not int or steps <= 0 or type(token_limit) is not int or token_limit <= 0:
        raise ValueError('positive integer steps and token limit required')
    if new_token_limit is not None and (type(new_token_limit) is not int or new_token_limit <= 0):
        raise ValueError('new-token limit must be a positive integer')
    if start_cooldown and not resume: raise ValueError('cooldown transition requires resume')
    if resume and init_from: raise ValueError('resume and init-from are mutually exclusive')
    if output.exists() and not resume: raise FileExistsError('output exists; choose resume or a new run name')
    if resume and not (output/'latest.json').exists(): raise ValueError('no checkpoint to resume')
    data = Corpus(corpus_path); old = Corpus(replay) if replay else None
    worker = Trainer(data, ModelConfig(**config['model']), TrainConfig(**config['training']),
                     device, old, allow_test_data)
    if resume:
        state = read_checkpoint(output)
        if state['test_only'] and not allow_test_data: raise ValueError('test-only checkpoint')
        worker.load_state_dict(state)
        if start_cooldown:
            if worker.cfg.cooldown_start is not None: raise ValueError('cooldown already configured')
            worker.cfg = replace(worker.cfg, cooldown_start=worker.counters['tokens'])
    elif init_from:
        state = read_checkpoint(init_from)
        if state['model_config'] != worker.model.cfg.to_dict(): raise ValueError('parent model config mismatch')
        if state['tokenizer_hash'] != data.tokenizer.fingerprint: raise ValueError('parent tokenizer mismatch')
        if state['test_only'] and not allow_test_data: raise ValueError('test-only parent checkpoint')
        protect_holdouts(data, state['heldouts']); protect_holdouts(old, state['heldouts'])
        worker.heldouts = merge_holdouts(worker.heldouts, state['heldouts'])
        worker.model.load_state_dict(state['model']); worker.parent_checkpoint_sha256 = state['_sha256']
        worker.test_only = worker.test_only or state['test_only']
    output.mkdir(parents=True, exist_ok=resume)
    lock = output/'.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        if start_cooldown:
            atomic_json(output/'config.json', {'model':worker.model.cfg.to_dict(), 'training':worker.cfg.to_dict()})
        if not resume:
            atomic_json(output/'config.json', config)
            data.tokenizer.save(output/'tokenizer.json')
            save_checkpoint(worker, output)
            if on_save: on_save()
        last_save = time.monotonic(); start_step = worker.counters['steps']; reason = 'step_limit'
        with (output/'metrics.jsonl').open('a', encoding='utf-8') as log:
            while worker.counters['steps'] - start_step < steps:
                if time.monotonic() - started >= allowance:
                    reason = 'time_or_estimated_budget_limit'; break
                remaining = token_limit - worker.counters['tokens']
                if remaining <= 0: reason = 'token_limit'; break
                new_remaining = None if new_token_limit is None else new_token_limit - worker.counters['new_tokens']
                if new_remaining is not None and new_remaining <= 0:
                    reason = 'new_token_limit'; break
                before = time.monotonic()
                record = worker.step(remaining, new_remaining)
                record['step_seconds'] = time.monotonic() - before
                record['elapsed_seconds'] = time.monotonic() - started
                log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                if record['steps'] % worker.cfg.checkpoint_steps == 0 or time.monotonic() - last_save >= worker.cfg.checkpoint_seconds:
                    save_checkpoint(worker, output)
                    if on_save: on_save()
                    last_save = time.monotonic()
        save_checkpoint(worker, output)
        if on_save: on_save()
        report = dict(**worker.counters, stop_reason=reason, elapsed_seconds=time.monotonic() - started,
                      test_only=bool(worker.test_only), device=str(worker.device))
        atomic_json(output/'last_run.json', report)
        if on_save: on_save()
        return report
    finally:
        lock.unlink(missing_ok=True)
        if on_save: on_save()  # Persist lock removal too, including on controlled failure.


@torch.no_grad()
def evaluate(model, dataset, max_windows=None):
    if not len(dataset): raise ValueError('empty evaluation split; never silently evaluate train instead')
    limit = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    if limit < 1: raise ValueError('positive evaluation limit required')
    was_training = model.training; model.eval()
    device = next(model.parameters()).device
    total = content = 0.0; count = nbytes = 0
    try:
        for i in range(limit):
            x, y, sizes = dataset[i]
            x = torch.from_numpy(x)[None].to(device); y = torch.from_numpy(y).to(device)
            logits, _ = model(x)
            losses = F.cross_entropy(logits[0].float(), y, reduction='none', ignore_index=-100)
            mask = y != -100; byte_mask = torch.from_numpy(sizes > 0).to(device) & mask
            total += float(losses[mask].sum()); count += int(mask.sum())
            content += float(losses[byte_mask].sum()); nbytes += int(sizes.sum())
    finally:
        model.train(was_training)
    return {'nll': total/count, 'bpb': content/(nbytes*math.log(2)) if nbytes else None,
            'targets': count, 'content_bytes': nbytes, 'windows': limit,
            'total_windows': len(dataset), 'partial': limit != len(dataset),
            'identity': dataset.identity, 'scientific_accuracy': 'not measured'}


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.0, question=True):
    if max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('invalid generation parameters')
    ids = [tokenizer.bos_id] + tokenizer.encode(prompt) + ([tokenizer.sep_id] if question else [])
    if len(ids) >= model.cfg.max_seq_len: raise ValueError('prompt exceeds trained context')
    device = next(model.parameters()).device; was_training = model.training; model.eval()
    output = []; cache = None; inp = torch.tensor([ids], device=device)
    capacity = model.cfg.max_seq_len - len(ids)
    try:
        for _ in range(min(max_new_tokens, capacity)):
            logits, cache = model(inp, cache=cache, use_cache=True)
            scores = logits[0, -1].float().clone()
            scores[len(tokenizer.tokens):] = -torch.inf
            scores[[tokenizer.pad_id, tokenizer.bos_id, tokenizer.sep_id]] = -torch.inf
            if temperature == 0: selected = int(scores.argmax())
            else: selected = int(torch.multinomial(torch.softmax(scores/temperature, dim=-1), 1))
            output.append(selected)
            if selected == tokenizer.eos_id: break
            inp = torch.tensor([[selected]], device=device)
    finally:
        model.train(was_training)
    raw = b''.join(tokenizer.tokens[i] for i in output)
    try: text = raw.decode('utf-8'); valid = True
    except UnicodeDecodeError: text = raw.decode('utf-8', errors='replace'); valid = False
    return {'text': text, 'generated_ids': output, 'utf8_valid': valid,
            'finished_eos': bool(output and output[-1] == tokenizer.eos_id),
            'context_exhausted': len(output) == capacity,
            'notice': 'Unverified model output; not a validated scientific result.'}
