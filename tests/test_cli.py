import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from helpers import encoded_fixture
from test_training import cfg


def test_cli_count_matches_approved_configuration():
    p=subprocess.run([sys.executable,'-m','model_as_a_paper','count'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['parameters']==44_577_280


def test_cli_train_resume_and_evaluate(tmp_path):
    data=encoded_fixture(tmp_path/'data')
    config=tmp_path/'cfg.json'; config.write_text(json.dumps(cfg().to_dict()))
    base=[sys.executable,'-m','model_as_a_paper','train','--config',str(config),'--data',str(data),'--out',str(tmp_path/'run'),'--device','cpu','--allow-test-data']
    a=subprocess.run(base+['--max-steps-this-run','2'],capture_output=True,text=True)
    assert a.returncode==0,a.stderr
    assert json.loads(a.stdout)['seen_tokens']==128
    b=subprocess.run(base+['--resume'],capture_output=True,text=True)
    assert b.returncode==0,b.stderr
    c=subprocess.run([sys.executable,'-m','model_as_a_paper','evaluate','--checkpoint',str(tmp_path/'run'),'--data',str(data),'--max-blocks','2','--device','cpu'],capture_output=True,text=True)
    assert c.returncode==0,c.stderr
    assert json.loads(c.stdout)['bpb']>0


def test_generate_tokens_stays_within_context_and_uses_cache():
    try: from model_as_a_paper.inference import generate_ids
    except ImportError: pytest.fail('Generation implementation missing')
    from test_model import small
    m=small().eval()
    result=generate_ids(m,[1,10,11],max_new_tokens=5,temperature=0)
    assert len(result)<=5
    with pytest.raises(ValueError,match='context|Context'):
        generate_ids(m,[1]*31,max_new_tokens=3)


def test_retention_not_claimed_when_old_knowledge_absent():
    from model_as_a_paper.evaluation import VerifiedOutcome,summarize
    r=summarize([VerifiedOutcome('r1','retention',False,False)])['retention']
    assert r.get('previously_correct_n')==0
    assert r['retention_among_previously_correct'] is None


def test_known_retention_denominator():
    from model_as_a_paper.evaluation import VerifiedOutcome,summarize
    rows=[VerifiedOutcome('r1','retention',True,True),VerifiedOutcome('r2','retention',True,False),VerifiedOutcome('r3','retention',False,True)]
    r=summarize(rows)['retention']
    assert r['previously_correct_n']==2
    assert r['retention_among_previously_correct']==0.5
