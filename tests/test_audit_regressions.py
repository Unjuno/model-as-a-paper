import json
from pathlib import Path
import pytest
import torch
from model_as_a_paper.data import prepare, Corpus, WindowDataset
from model_as_a_paper.model import ModelConfig
from model_as_a_paper.training import train, Trainer, TrainConfig
from test_training import build, config
from test_data import rows, write_rows


def test_supervised_examples_must_fit_context(tmp_path):
    old = build(tmp_path)
    new = build(tmp_path, 'supervised', old/'tokenizer.json', correction=True)
    with pytest.raises(ValueError, match='supervised.*context'):
        WindowDataset(Corpus(new), 'train', 16)


def test_malformed_evidence_cannot_be_marked_verified(tmp_path):
    r = rows(1)[0]; r.pop('text')
    r.update(kind='correction', prompt='Q', answer='A', verified=True,
             evidence='trust me', verification='trust me')
    p = write_rows(tmp_path, [r])
    with pytest.raises(ValueError, match='evidence'):
        prepare(p, tmp_path/'bad', vocab_size=300, min_chars=1, allow_test_data=True)


def test_nan_norm_epsilon_is_rejected():
    with pytest.raises(ValueError): ModelConfig(norm_eps=float('nan'))


def test_final_persistence_callback_observes_lock_removed(tmp_path):
    data = build(tmp_path); output = tmp_path/'run'; observations = []
    train(config(), data, output, steps=1, seconds=120, allow_test_data=True,
          on_save=lambda: observations.append((output/'.running').exists()))
    assert observations[-1] is False


def test_parent_holdout_cannot_become_update_training_data(tmp_path):
    source_rows = rows(60); p = write_rows(tmp_path, source_rows)
    prepare(p, tmp_path/'base', vocab_size=300, min_chars=1, allow_test_data=True)
    c = Corpus(tmp_path/'base')
    target = next(r for r in c.records if r['split'] == 'test')
    train(config(), tmp_path/'base', tmp_path/'parent', steps=1, seconds=120, allow_test_data=True)
    # Deliberately try to move a parent's held-out paper into a train-only update.
    r = next(r for r in source_rows if r['paper_id'] == target['paper_id'])
    p = write_rows(tmp_path, [r])
    prepare(p, tmp_path/'leaked', tokenizer_path=tmp_path/'base'/'tokenizer.json',
            split='train', min_chars=1, allow_test_data=True)
    with pytest.raises(ValueError, match='held-out'):
        train(config(), tmp_path/'leaked', tmp_path/'child', init_from=tmp_path/'parent',
              steps=1, seconds=120, allow_test_data=True)


def test_padding_does_not_change_token_weighted_update(tmp_path):
    items = rows(10)
    for i, r in enumerate(items): r['text'] = 'x' * (i + 1)
    p = write_rows(tmp_path, items)
    prepare(p, tmp_path/'short', vocab_size=300, split='train', min_chars=1, allow_test_data=True)
    cfg = config(); cfg['training']['micro_batch_size'] = 1
    a = Trainer(Corpus(tmp_path/'short'), ModelConfig(**cfg['model']), TrainConfig(**cfg['training']), allow_test_data=True)
    cfg['training']['micro_batch_size'] = 3
    b = Trainer(Corpus(tmp_path/'short'), ModelConfig(**cfg['model']), TrainConfig(**cfg['training']), allow_test_data=True)
    a.step(); b.step()
    for x, y in zip(a.model.parameters(), b.model.parameters()): assert torch.allclose(x, y, atol=2e-6)


def test_equal_new_token_cap_with_replay_is_exact(tmp_path):
    old = build(tmp_path)
    new = build(tmp_path, 'new-cap', old/'tokenizer.json', correction=True)
    a = config(); a['training'].update(seq_len=512, long_seq_len=None)
    b = {'model':a['model'], 'training':dict(a['training'], replay_ratio=0.2)}
    no_replay = train(a, new, tmp_path/'without', steps=100, seconds=120,
                      new_token_limit=333, allow_test_data=True)
    with_replay = train(b, new, tmp_path/'with', steps=100, seconds=120,
                        new_token_limit=333, replay=old, allow_test_data=True)
    assert no_replay['new_tokens'] == with_replay['new_tokens'] == 333
    assert no_replay['replay_tokens'] == 0
    assert with_replay['replay_tokens'] > 0


def test_explicit_cooldown_keeps_optimizer_and_resume_counters(tmp_path):
    from model_as_a_paper.training import read_checkpoint
    data = build(tmp_path); cfg = config()
    cfg['training']['cooldown_tokens'] = 64
    train(cfg, data, tmp_path/'cool', steps=2, seconds=120, allow_test_data=True)
    train(cfg, data, tmp_path/'cool', steps=1, seconds=120, resume=True,
          start_cooldown=True, allow_test_data=True)
    state = read_checkpoint(tmp_path/'cool')
    assert state['train_config']['cooldown_start'] == 128
    assert state['counters']['steps'] == 3
    assert all(float(s['step']) == 3 for s in state['optimizer']['state'].values())
    saved = json.loads((tmp_path/'cool'/'config.json').read_text())
    train(saved, data, tmp_path/'cool', steps=1, seconds=120, resume=True, allow_test_data=True)
    assert read_checkpoint(tmp_path/'cool')['counters']['steps'] == 4
