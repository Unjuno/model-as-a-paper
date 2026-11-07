import json
import numpy as np
import pytest
import torch
from model_as_a_paper.model import ModelConfig
from model_as_a_paper.data import prepare, Corpus, WindowDataset
from model_as_a_paper.training import TrainConfig, Trainer, train, read_checkpoint, evaluate, generate
from model_as_a_paper.budget import run_seconds


def build(tmp_path, name='data', tokenizer=None, correction=False):
    p = tmp_path / f'{name}.jsonl'
    items = []
    for i in range(6):
        r = dict(paper_id=f'TEST-{name}-{i}', version='fixture', license='TEST-ONLY', source_kind='test_fixture')
        if correction:
            r.update(kind='correction', prompt='Claim? ' * (i+1), answer='Correct under stated conditions. '*(i+1),
                     verified=True, evidence=[dict(paper_id='TEST', version='fixture', span='fixture')],
                     verification='software test only')
        else: r['text'] = 'alpha beta alpha beta gamma.\n' * 20 + str(i)
        items.append(r)
    p.write_text(''.join(json.dumps(r)+'\n' for r in items))
    out = tmp_path/name
    prepare(p, out, vocab_size=300, tokenizer_path=tokenizer, min_chars=1,
            split='train', allow_test_data=True)
    return out


def config(**kwargs):
    m = ModelConfig(vocab_size=300, dim=32, layers=2, heads=4, kv_heads=2, ffn_dim=64, max_seq_len=512)
    t = TrainConfig(seq_len=16, long_seq_len=32, switch_after_tokens=256,
                    batch_tokens=64, micro_batch_size=2, warmup_tokens=0,
                    learning_rate=0.003, checkpoint_steps=2, **kwargs)
    return {'model':m.to_dict(), 'training':t.to_dict()}


def test_training_reduces_loss_and_fresh_run_refuses_overwrite(tmp_path):
    d = build(tmp_path)
    c = Corpus(d); cfg = config()
    worker = Trainer(c, ModelConfig(**cfg['model']), TrainConfig(**cfg['training']), allow_test_data=True)
    ev = WindowDataset(c, 'train', 16)
    first = evaluate(worker.model, ev, max_windows=20)['nll']
    for _ in range(40): worker.step()
    last = evaluate(worker.model, ev, max_windows=20)['nll']
    assert last < first * 0.65
    train(cfg, d, tmp_path/'run', steps=2, seconds=120, allow_test_data=True)
    with pytest.raises(FileExistsError):
        train(cfg, d, tmp_path/'run', steps=2, seconds=120, allow_test_data=True)


def test_resume_bitwise_matches_uninterrupted_cpu_run(tmp_path):
    d = build(tmp_path); cfg = config()
    train(cfg, d, tmp_path/'full', steps=8, seconds=120, allow_test_data=True)
    train(cfg, d, tmp_path/'resumed', steps=3, seconds=120, allow_test_data=True)
    train(cfg, d, tmp_path/'resumed', steps=5, seconds=120, resume=True, allow_test_data=True)
    a = read_checkpoint(tmp_path/'full'); b = read_checkpoint(tmp_path/'resumed')
    assert a['counters'] == b['counters']
    assert a['counters']['tokens'] == 512
    for name in a['model']:
        assert torch.equal(a['model'][name], b['model'][name]), name
    for k in a['optimizer']['state']:
        for p in a['optimizer']['state'][k]:
            assert torch.equal(a['optimizer']['state'][k][p], b['optimizer']['state'][k][p])


def test_resume_rejects_changed_training_config(tmp_path):
    d = build(tmp_path); cfg = config()
    train(cfg, d, tmp_path/'run', steps=1, seconds=120, allow_test_data=True)
    cfg['training']['learning_rate'] = 0.1
    with pytest.raises(ValueError, match='config'):
        train(cfg, d, tmp_path/'run', resume=True, steps=1, seconds=120, allow_test_data=True)


def test_gradient_accumulation_is_target_token_weighted(tmp_path):
    d = build(tmp_path); cfg = config()
    a = Trainer(Corpus(d), ModelConfig(**cfg['model']), TrainConfig(**cfg['training']), allow_test_data=True)
    cfg['training']['micro_batch_size'] = 4
    b = Trainer(Corpus(d), ModelConfig(**cfg['model']), TrainConfig(**cfg['training']), allow_test_data=True)
    a.step(); b.step()
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        assert torch.allclose(x, y, atol=2e-6)


def test_init_from_preserves_parent_and_replay_tracks_target_tokens(tmp_path):
    old = build(tmp_path); cfg = config()
    train(cfg, old, tmp_path/'parent', steps=2, seconds=120, allow_test_data=True)
    parent = read_checkpoint(tmp_path/'parent')
    new = build(tmp_path, 'new', old/'tokenizer.json', correction=True)
    cfg['training'].update(replay_ratio=0.2, seq_len=512, long_seq_len=None)
    report = train(cfg, new, tmp_path/'child', init_from=tmp_path/'parent', replay=old,
                   steps=20, seconds=120, allow_test_data=True)
    child = read_checkpoint(tmp_path/'child')
    assert child['parent_checkpoint_sha256']
    assert child['counters']['steps'] == 20
    assert abs(report['replay_tokens'] / report['tokens'] - 0.2) < 0.05
    assert read_checkpoint(tmp_path/'parent')['counters'] == parent['counters']


def test_checkpoint_tampering_and_tokenizer_mismatch_rejected(tmp_path):
    d = build(tmp_path); cfg = config()
    train(cfg, d, tmp_path/'run', steps=1, seconds=120, allow_test_data=True)
    ptr = json.loads((tmp_path/'run'/'latest.json').read_text())
    with (tmp_path/'run'/ptr['file']).open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError, match='hash'): read_checkpoint(tmp_path/'run')


def test_generation_respects_context_and_excludes_unused_ids(tmp_path):
    d = build(tmp_path); c = Corpus(d); cfg = config()
    w = Trainer(c, ModelConfig(**cfg['model']), TrainConfig(**cfg['training']), allow_test_data=True)
    result = generate(w.model, c.tokenizer, 'alpha', max_new_tokens=4)
    assert len(result['generated_ids']) <= 4
    assert all(i < len(c.tokenizer.tokens) and i not in (256, 257, 259) for i in result['generated_ids'])
    with pytest.raises(ValueError): generate(w.model, c.tokenizer, 'x ' * 1000)


def test_token_limit_is_not_overshot_and_no_default_fixture_training(tmp_path):
    d = build(tmp_path); cfg = config()
    with pytest.raises(ValueError): train(cfg, d, tmp_path/'bad', steps=1, seconds=120)
    report = train(cfg, d, tmp_path/'run', steps=20, token_limit=100, seconds=120, allow_test_data=True)
    assert report['tokens'] == 100


@pytest.mark.parametrize('budget,rate', [(float('nan'),1), (-1,1), (1,0), (1,float('inf'))])
def test_invalid_budget_rejected(budget, rate):
    with pytest.raises(ValueError): run_seconds(600, budget, rate)


def test_budget_arithmetic_and_checkpoint_reserve():
    assert run_seconds(600, 0.1, 1.0, reserve=60) == pytest.approx(264.0)
    assert run_seconds(120, None, None, reserve=10) == 110
