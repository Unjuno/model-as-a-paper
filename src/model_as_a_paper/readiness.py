"""Non-mutating corpus/model readiness checks and real measured step timing.

This module never downloads data, starts Modal, or writes a model checkpoint.
The benchmark deliberately uses the actual PyTorch training path with a
small number of real updates; it is not an inferred GPU performance number.
"""
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import median
import time

import torch

from .data import Corpus, WindowDataset
from .model import ModelConfig, PaperLM
from .training import TrainConfig, Trainer, protect_holdouts, read_checkpoint


def preflight(config, corpus_path, *, replay=None, parent=None, device='cpu',
              allow_test_data=False):
    """Verify everything possible without constructing a full-size GPU model."""
    cfg = TrainConfig(**config['training'])
    model_cfg = ModelConfig(**config['model'])
    corpus = Corpus(corpus_path)
    if corpus.manifest['test_only'] and not allow_test_data:
        raise ValueError('test-only corpus cannot be used in scientific training')
    if corpus.tokenizer.capacity != model_cfg.vocab_size:
        raise ValueError('vocabulary capacity does not match model')
    if max(cfg.seq_len, cfg.long_seq_len or cfg.seq_len) > model_cfg.max_seq_len:
        raise ValueError('training context exceeds model maximum')
    replay_corpus = Corpus(replay) if replay else None
    if bool(cfg.replay_ratio) != bool(replay_corpus):
        raise ValueError('replay corpus and replay_ratio must both be provided')
    if replay_corpus is not None:
        if replay_corpus.tokenizer.fingerprint != corpus.tokenizer.fingerprint:
            raise ValueError('replay tokenizer mismatch')
        if replay_corpus.manifest['test_only'] and not allow_test_data:
            raise ValueError('test-only replay cannot be used in scientific training')
    for source in (corpus, replay_corpus):
        if source is None: continue
        if not any(r['split'] == 'train' for r in source.records):
            raise ValueError('no training records in required corpus')
        WindowDataset(source, 'train', cfg.seq_len)
        if cfg.long_seq_len and cfg.long_seq_len != cfg.seq_len:
            WindowDataset(source, 'train', cfg.long_seq_len)
    if parent is not None:
        state = read_checkpoint(parent)
        if state['model_config'] != model_cfg.to_dict():
            raise ValueError('parent model config mismatch')
        if state['tokenizer_hash'] != corpus.tokenizer.fingerprint:
            raise ValueError('parent tokenizer mismatch')
        if state['test_only'] and not allow_test_data:
            raise ValueError('test-only parent checkpoint')
        protect_holdouts(corpus, state['heldouts'])
        protect_holdouts(replay_corpus, state['heldouts'])
    actual_device = torch.device(device)
    if actual_device.type not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if actual_device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        if cfg.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('CUDA GPU does not support BF16')
    with torch.device('meta'):
        model = PaperLM(model_cfg)
    params = model.parameter_count()
    dataset = WindowDataset(corpus, 'train', cfg.seq_len)
    if not len(dataset): raise ValueError('no usable training windows')
    return {
        'trainable': True,
        'device': str(actual_device),
        'parameters': params,
        'estimated_state_mib': round((16 * params)/(1024**2), 1),
        'memory_note': 'FP32 parameters + gradients + Adam two moments only; excludes activations, logits, CUDA workspace and caches',
        'corpus_tokens': corpus.manifest['tokens'],
        'train_documents': sum(r['split'] == 'train' for r in corpus.records),
        'train_windows': len(dataset),
        'split_counts': corpus.manifest['split_counts'],
        'actual_vocab_size': corpus.manifest['actual_vocab_size'],
        'tokenizer_capacity': corpus.tokenizer.capacity,
        'replay_enabled': bool(replay_corpus),
        'parent_checked': parent is not None,
        'test_only': bool(corpus.manifest['test_only']) or bool(replay_corpus and replay_corpus.manifest['test_only']),
        'note': 'No training run or external network request was performed',
    }


def benchmark(config, corpus_path, *, device='cpu', steps=3, replay=None,
              allow_test_data=False, micro_batch_size=None, batch_tokens=None):
    """Measured training updates without writing checkpoints.

    Results are observations for the specified batch/config on this host.
    A short benchmark cannot predict training-quality convergence.
    """
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError('benchmark steps must be integers in [1,20]')
    for key, value in [('micro_batch_size', micro_batch_size), ('batch_tokens', batch_tokens)]:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'invalid benchmark override: {key}')
    cfg = TrainConfig(**config['training'])
    if batch_tokens is not None: cfg = replace(cfg, batch_tokens=batch_tokens)
    preflight({'model': config['model'], 'training': cfg.to_dict()}, corpus_path,
              replay=replay, device=device, allow_test_data=allow_test_data)
    worker = Trainer(Corpus(corpus_path), ModelConfig(**config['model']), cfg,
                     device, Corpus(replay) if replay else None, allow_test_data)
    observed = []
    if worker.device.type == 'cuda':
        torch.cuda.synchronize(worker.device)
        torch.cuda.reset_peak_memory_stats(worker.device)
    for _ in range(steps):
        start = time.monotonic()
        record = worker.step(micro_batch_size=micro_batch_size)
        if worker.device.type == 'cuda': torch.cuda.synchronize(worker.device)
        elapsed = time.monotonic() - start
        observed.append((elapsed, record['tokens'] - (observed[-1][2] if observed else 0), record['tokens']))
    rate = [tokens/elapsed for elapsed,tokens,_ in observed]
    return {
        'device': str(worker.device),
        'steps': steps,
        'tokens': worker.counters['tokens'],
        'median_step_seconds': median(row[0] for row in observed),
        'median_tokens_per_second': median(rate),
        'observed_step_seconds': [round(row[0], 6) for row in observed],
        'peak_cuda_memory_mib': (round(torch.cuda.max_memory_allocated(worker.device)/(1024**2),1)
                                 if worker.device.type == 'cuda' else None),
        'note': 'Measured updates; no checkpoint. Includes backward/optimizer, excludes data download, startup and save.',
    }


def metrics_summary(run_path):
    """Summarize committed metrics only; reject a log ahead of its checkpoint."""
    path = Path(run_path)
    state = read_checkpoint(path)
    filename = path/'metrics.jsonl'
    if not filename.exists(): raise ValueError('no metrics log')
    rows = []
    for line in filename.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        record = json.loads(line)
        if type(record.get('steps')) is not int:
            raise ValueError('invalid step in metrics')
        if rows and record['steps'] <= rows[-1]['steps']:
            raise ValueError('metrics steps are not strictly increasing')
        rows.append(record)
    if any(r['steps'] > state['counters']['steps'] for r in rows):
        raise ValueError('metrics are ahead of checkpoint; resume first to reconcile')
    if not rows: raise ValueError('no logged training steps')
    elapsed = [r['step_seconds'] for r in rows if isinstance(r.get('step_seconds'),(int,float))
               and math.isfinite(r['step_seconds']) and r['step_seconds'] > 0]
    latest = rows[-1]
    return {
        'logged_steps': len(rows),
        'latest_logged_step': latest['steps'],
        'latest_checkpoint_step': state['counters']['steps'],
        'trained_tokens': state['counters']['tokens'],
        'latest_loss': latest.get('loss'),
        'median_step_seconds': median(elapsed) if elapsed else None,
        'median_tokens_per_second': (median([r['input_tokens']/r['step_seconds'] for r in rows
                                             if r.get('input_tokens',0)>0 and r.get('step_seconds',0)>0])
                                     if any(r.get('input_tokens',0)>0 and r.get('step_seconds',0)>0 for r in rows)
                                     else None),
        'note': 'Historical timing across runs/devices may differ; compare matched subsets.',
    }
