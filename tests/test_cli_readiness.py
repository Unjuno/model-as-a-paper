"""RED-first command-line and one-command training preparation checks."""
import json
import subprocess
import sys
from pathlib import Path

from test_training import build, config
from model_as_a_paper.cli import main


def test_cli_preflight_prints_actionable_readiness(tmp_path, capsys):
    d = build(tmp_path)
    c = tmp_path/'small.json'; c.write_text(json.dumps(config()))
    assert main(['preflight','--config',str(c),'--data',str(d),'--allow-test-data']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['trainable'] is True
    assert result['train_windows'] > 0


def test_cli_bench_and_metrics_stats(tmp_path, capsys):
    from model_as_a_paper.training import train
    d = build(tmp_path); c = tmp_path/'small.json'; c.write_text(json.dumps(config()))
    assert main(['bench','--config',str(c),'--data',str(d),'--steps','2',
                 '--micro-batch-size','1','--allow-test-data']) == 0
    b = json.loads(capsys.readouterr().out)
    assert b['steps'] == 2
    train(config(), d, tmp_path/'run', steps=1, seconds=120, allow_test_data=True)
    assert main(['stats','--run',str(tmp_path/'run')]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r['logged_steps'] == 1


def test_setup_script_uses_existing_jsonl_and_does_not_launch_gpu(tmp_path):
    original = build(tmp_path)
    raw = tmp_path/'data.jsonl'
    small = tmp_path/'cfg.json'; small.write_text(json.dumps(config()))
    output = tmp_path/'prepared'
    script = Path(__file__).resolve().parents[1]/'scripts'/'setup_training.py'
    result = subprocess.run([sys.executable, str(script), '--input', str(raw),
             '--output', str(output), '--config',str(small), '--min-chars','1',
             '--vocab-size','300','--allow-test-data'],
             cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    result_json=json.loads(result.stdout)
    assert result_json['trainable'] is True
    assert (output/'manifest.json').exists()
    assert 'no CUDA' in result_json['actions_executed']
    assert not (tmp_path/'model').exists()


def test_modal_entrypoint_exposes_safe_runtime_and_cooldown_controls():
    """Static contract; does not import Modal or allocate GPU resources."""
    import ast
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'modal_apps'/'train.py').read_text())
    names={x.name:[a.arg for a in x.args.args] for x in tree.body if isinstance(x,ast.FunctionDef)}
    for key in ['micro_batch_size','checkpoint_steps','checkpoint_seconds','start_cooldown']:
        assert key in names['main']
        assert key in names['gpu_train']


def test_report_records_gpu_memory_column_or_null(tmp_path):
    d=build(tmp_path); cfg=config()
    from model_as_a_paper.training import train
    run=tmp_path/'run'
    train(cfg,d,run,steps=1,seconds=120,allow_test_data=True)
    entry=json.loads((run/'metrics.jsonl').read_text().splitlines()[0])
    assert 'cuda_peak_memory_mib' in entry
    assert entry['cuda_peak_memory_mib'] is None
    assert entry['input_tokens'] == entry['tokens']


def test_setup_from_sample_prints_one_machine_readable_json(tmp_path, monkeypatch, capsys):
    from scripts import setup_training
    from test_training import build
    build(tmp_path)
    original = tmp_path/'data.jsonl'
    c=tmp_path/'cfg.json'; c.write_text(json.dumps(config()))
    def fake_fetch(cmd, check, **kwargs):
        output = Path(cmd[cmd.index('--output')+1])
        output.write_text(original.read_text())
        print('{"nested_fetch": "stdout must not leak into setup JSON"}')
    monkeypatch.setattr(setup_training.subprocess, 'run', fake_fetch)
    code=setup_training.main(['--output',str(tmp_path/'prepared'), '--config',str(c),
                               '--vocab-size','300','--min-chars','1', '--allow-test-data'])
    assert code == 0
    parsed=json.loads(capsys.readouterr().out)
    assert parsed['prepared_corpus'] == str(tmp_path/'prepared')
