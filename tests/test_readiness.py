"""RED-first regressions for launch readiness, runtime knobs and metrics integrity."""
import json
from pathlib import Path

import pytest
from test_training import build, config
from model_as_a_paper.training import train, read_checkpoint
from model_as_a_paper.readiness import preflight, benchmark, metrics_summary


def test_preflight_is_readonly_and_catches_test_data(tmp_path):
    d = build(tmp_path)
    cfg = config()
    with pytest.raises(ValueError, match='test-only'):
        preflight(cfg, d)
    result = preflight(cfg, d, allow_test_data=True)
    assert result['parameters'] == 20096 or result['parameters'] > 0
    assert result['train_windows'] > 0
    assert result['corpus_tokens'] > 0
    assert result['trainable'] is True
    assert result['estimated_state_mib'] > 0
    assert set(x.name for x in tmp_path.iterdir()) == {'data', 'data.jsonl'}


def test_preflight_rejects_update_holdout_from_parent(tmp_path):
    from model_as_a_paper.data import prepare, Corpus
    from test_data import rows, write_rows
    r = rows(60)
    p = write_rows(tmp_path, r)
    parent_corpus = tmp_path / 'parent_corpus'
    prepare(p, parent_corpus, vocab_size=300, min_chars=1, allow_test_data=True)
    held = next(row for row in Corpus(parent_corpus).records if row['split'] == 'test')
    train(config(), parent_corpus, tmp_path/'parent_run', steps=1, seconds=120, allow_test_data=True)
    row = next(row for row in r if row['paper_id'] == held['paper_id'])
    p2 = tmp_path/'update.jsonl'; p2.write_text(json.dumps(row)+'\n')
    update = tmp_path/'update_data'
    prepare(p2, update, tokenizer_path=parent_corpus/'tokenizer.json', split='train', min_chars=1, allow_test_data=True)
    with pytest.raises(ValueError, match='held-out'):
        preflight(config(), update, parent=tmp_path/'parent_run', allow_test_data=True)


def test_benchmark_reports_observations_without_checkpoint(tmp_path):
    d = build(tmp_path)
    report = benchmark(config(), d, steps=2, allow_test_data=True, micro_batch_size=1)
    assert report['steps'] == 2
    assert report['tokens'] > 0
    assert report['median_step_seconds'] > 0
    assert report['median_tokens_per_second'] > 0
    assert report['device'] == 'cpu'
    assert not list(tmp_path.glob('step-*.pt'))


def test_runtime_knobs_on_resume_do_not_change_scientific_config(tmp_path):
    d = build(tmp_path)
    cfg = config()
    train(cfg, d, tmp_path/'run', steps=2, seconds=120, allow_test_data=True)
    report = train(cfg, d, tmp_path/'run', steps=2, seconds=120, resume=True,
                   allow_test_data=True, micro_batch_size=1, checkpoint_steps=1,
                   checkpoint_seconds=0.1)
    assert report['steps'] == 4
    state = read_checkpoint(tmp_path/'run')
    assert state['train_config']['micro_batch_size'] == cfg['training']['micro_batch_size']
    logs = [json.loads(x) for x in (tmp_path/'run'/'metrics.jsonl').read_text().splitlines()]
    assert logs[-1]['runtime_micro_batch_size'] == 1
    assert [x['steps'] for x in logs] == [1,2,3,4]


@pytest.mark.parametrize('kw', [
    {'micro_batch_size':0}, {'micro_batch_size':1.5},
    {'checkpoint_steps':0}, {'checkpoint_seconds':float('nan')},
])
def test_invalid_runtime_knobs_are_rejected(tmp_path,kw):
    d = build(tmp_path)
    with pytest.raises(ValueError, match='override'):
        train(config(), d, tmp_path/'run', steps=1, seconds=120, allow_test_data=True, **kw)


def test_stale_metrics_reconciled_to_checkpoint_on_resume(tmp_path):
    d = build(tmp_path)
    cfg = config()
    run = tmp_path/'run'
    train(cfg, d, run, steps=3, seconds=120, allow_test_data=True)
    with (run/'metrics.jsonl').open('a') as f:
        f.write(json.dumps({'steps': 4, 'tokens': 10000, 'loss': 500.0, 'step_seconds': 1.0})+'\n')
    result = train(cfg, d, run, steps=2, seconds=120, allow_test_data=True, resume=True)
    assert result['steps'] == 5
    logs = [json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    assert [x['steps'] for x in logs] == [1,2,3,4,5]
    assert all(x['loss'] < 500 for x in logs)
    assert list(run.glob('metrics.orphaned-*.jsonl'))
    summary = metrics_summary(run)
    assert summary['logged_steps'] == 5
    assert summary['latest_checkpoint_step'] == 5


def test_metrics_summary_rejects_log_ahead_of_checkpoint(tmp_path):
    d=build(tmp_path); run=tmp_path/'run'
    train(config(), d, run, steps=1, seconds=120, allow_test_data=True)
    with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps({'steps':2,'tokens':555,'step_seconds':0.01})+'\n')
    with pytest.raises(ValueError, match='ahead'):
        metrics_summary(run)
