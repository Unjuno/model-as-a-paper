"""CLI for local preparation/training; no network or cloud side effects on import."""
import argparse
import json
from pathlib import Path
import sys
import torch
from .data import Corpus, WindowDataset, prepare, DEFAULT_LICENSES
from .model import ModelConfig, PaperLM
from .tokenizer import ByteBPE
from .training import train, read_checkpoint, evaluate, generate
from .readiness import preflight, benchmark, metrics_summary


def load_model(path, device):
    state = read_checkpoint(path)
    root = Path(path) if Path(path).is_dir() else Path(path).parent
    tok = ByteBPE.load(root/'tokenizer.json')
    if tok.fingerprint != state['tokenizer_hash']: raise ValueError('checkpoint tokenizer mismatch')
    model = PaperLM(ModelConfig(**state['model_config'])).to(device)
    model.load_state_dict(state['model']); model.eval()
    return model, tok, state


def parser():
    p = argparse.ArgumentParser(prog='map-paper', description=__doc__)
    p.add_argument('--threads', type=int, default=2)
    subs = p.add_subparsers(dest='command', required=True)
    s = subs.add_parser('inspect'); s.add_argument('--config', default='configs/map44.json')
    s = subs.add_parser('prepare')
    s.add_argument('--input', required=True); s.add_argument('--output', required=True)
    s.add_argument('--tokenizer'); s.add_argument('--vocab-size', type=int, default=16384)
    s.add_argument('--allow-test-data', action='store_true'); s.add_argument('--allow-license', action='append')
    s.add_argument('--min-chars', type=int, default=1000)
    s.add_argument('--split', choices=['train','validation','test'])
    s.add_argument('--max-input-mib', type=int, default=256); s.add_argument('--tokenizer-mib', type=int, default=8)
    s = subs.add_parser('preflight', help='read-only validation before any training or GPU cost')
    s.add_argument('--config', default='configs/map44.json'); s.add_argument('--data', required=True)
    s.add_argument('--replay'); s.add_argument('--parent')
    s.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    s.add_argument('--allow-test-data', action='store_true')
    s = subs.add_parser('bench', help='timed training steps without checkpoint or cloud execution')
    s.add_argument('--config', default='configs/map44.json'); s.add_argument('--data', required=True)
    s.add_argument('--steps', type=int, default=2); s.add_argument('--replay')
    s.add_argument('--micro-batch-size', type=int); s.add_argument('--batch-tokens', type=int)
    s.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    s.add_argument('--allow-test-data', action='store_true')
    s = subs.add_parser('stats', help='summarize committed training log and checkpoint')
    s.add_argument('--run', required=True)
    s = subs.add_parser('train')
    s.add_argument('--config', default='configs/map44.json'); s.add_argument('--data', required=True)
    s.add_argument('--output', required=True); s.add_argument('--replay')
    group = s.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true'); group.add_argument('--init-from')
    s.add_argument('--steps', type=int, default=100); s.add_argument('--token-limit', type=int, default=500000000)
    s.add_argument('--new-token-limit', type=int)
    s.add_argument('--start-cooldown', action='store_true')
    s.add_argument('--micro-batch-size', type=int, help='runtime-only memory/throughput knob; safe to change on resume')
    s.add_argument('--checkpoint-steps', type=int, help='runtime-only save cadence')
    s.add_argument('--checkpoint-seconds', type=float, help='runtime-only save cadence')
    s.add_argument('--seconds', type=float, default=600)
    s.add_argument('--budget-usd', type=float); s.add_argument('--hourly-rate', type=float)
    s.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    s.add_argument('--allow-test-data', action='store_true')
    s = subs.add_parser('evaluate')
    s.add_argument('--checkpoint', required=True); s.add_argument('--data', required=True)
    s.add_argument('--split', choices=['train','validation','test'], default='validation')
    s.add_argument('--seq-len', type=int, default=1024); s.add_argument('--max-windows', type=int, default=100)
    s.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    s = subs.add_parser('generate')
    s.add_argument('--checkpoint', required=True); s.add_argument('--prompt', required=True)
    s.add_argument('--max-new-tokens', type=int, default=128); s.add_argument('--temperature', type=float, default=0.0)
    s.add_argument('--continuation', action='store_true'); s.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    s = subs.add_parser('smoke'); s.add_argument('--output', required=True)
    return p


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    if a.threads < 1: p.error('threads must be positive')
    torch.set_num_threads(a.threads)
    try:
        if a.command == 'inspect':
            cfg = json.loads(Path(a.config).read_text())
            with torch.device('meta'): model = PaperLM(ModelConfig(**cfg['model']))
            result = {'parameters': model.parameter_count(), 'model': model.cfg.to_dict()}
        elif a.command == 'preflight':
            cfg = json.loads(Path(a.config).read_text())
            result = preflight(cfg, a.data, replay=a.replay, parent=a.parent,
                               device=a.device, allow_test_data=a.allow_test_data)
        elif a.command == 'bench':
            cfg = json.loads(Path(a.config).read_text())
            result = benchmark(cfg, a.data, device=a.device, steps=a.steps, replay=a.replay,
                               allow_test_data=a.allow_test_data, micro_batch_size=a.micro_batch_size,
                               batch_tokens=a.batch_tokens)
        elif a.command == 'stats':
            result = metrics_summary(a.run)
        elif a.command == 'prepare':
            result = prepare(a.input, a.output, vocab_size=a.vocab_size, tokenizer_path=a.tokenizer,
                             allow_test_data=a.allow_test_data,
                             allowed_licenses=a.allow_license or DEFAULT_LICENSES,
                             min_chars=a.min_chars, split=a.split, max_input_bytes=a.max_input_mib*1024**2,
                             tokenizer_bytes=a.tokenizer_mib*1024**2)
        elif a.command == 'train':
            cfg = json.loads(Path(a.config).read_text())
            result = train(cfg, a.data, a.output, steps=a.steps, token_limit=a.token_limit, seconds=a.seconds,
                           budget_usd=a.budget_usd, hourly_rate=a.hourly_rate, device=a.device,
                           resume=a.resume, init_from=a.init_from, replay=a.replay, allow_test_data=a.allow_test_data,
                           new_token_limit=a.new_token_limit, start_cooldown=a.start_cooldown,
                           micro_batch_size=a.micro_batch_size, checkpoint_steps=a.checkpoint_steps,
                           checkpoint_seconds=a.checkpoint_seconds)
        elif a.command == 'evaluate':
            model, tok, state = load_model(a.checkpoint, a.device); corpus = Corpus(a.data)
            if tok.fingerprint != corpus.tokenizer.fingerprint: raise ValueError('evaluation tokenizer mismatch')
            result = evaluate(model, WindowDataset(corpus, a.split, a.seq_len), a.max_windows)
            result.update(split=a.split, test_only=state['test_only'] or corpus.manifest['test_only'])
        elif a.command == 'generate':
            model, tok, state = load_model(a.checkpoint, a.device)
            result = generate(model, tok, a.prompt, a.max_new_tokens, a.temperature, not a.continuation)
            result['test_only'] = state['test_only']
        else:
            from .smoke import smoke
            result = smoke(a.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileExistsError, FileNotFoundError, FloatingPointError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
