import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from model_as_a_paper.cloud_policy import validate_request

ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return subprocess.run([sys.executable, '-m', 'model_as_a_paper.cli', *map(str, args)],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=40)


def test_cpu_smoke_runs_end_to_end_and_marks_non_scientific(tmp_path):
    p = cli('smoke', '--output', tmp_path/'smoke')
    assert p.returncode == 0, p.stderr
    report = json.loads(p.stdout)
    assert report['test_only'] is True
    assert report['loss_after'] < report['loss_before']
    assert report['resume_verified'] is True
    assert (tmp_path/'smoke'/'run'/'latest.json').is_file()


def test_inspect_model_and_refuse_invalid_budget_without_cloud():
    p = cli('inspect', '--config', 'configs/map44.json')
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['parameters'] == 44577280
    with pytest.raises(ValueError): validate_request('papers', 'run', 600, 10, 1, 1, False)


@pytest.mark.parametrize('name', ['../escape', '/root', '.', '..', 'name/other', 'a;rm -rf'])
def test_cloud_name_cannot_escape_volume(name):
    with pytest.raises(ValueError): validate_request(name, 'run', 600, 10, 1, 1, True)


def test_cloud_request_caps_runtime_and_accepts_safe_names():
    with pytest.raises(ValueError): validate_request('papers', 'run', 9000, 10, 1, 1, True)
    validate_request('papers', 'run-001', 600, 10, 1, 1, True)
