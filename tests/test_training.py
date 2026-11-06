from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from helpers import encoded_fixture


def api():
    try:
        from model_as_a_paper.training import TrainConfig, train, lr_at, masked_loss_sum
        from model_as_a_paper.checkpoint import load_checkpoint
    except ImportError:
        pytest.fail('Training implementation is missing')
    return TrainConfig, train, lr_at, masked_loss_sum, load_checkpoint


def cfg(**overrides):
    Config, *_ = api()
    from model_as_a_paper.model import ModelConfig
    model = ModelConfig(vocab_size=260, dim=32, layers=2, heads=4, kv_heads=2, ffn_dim=64, max_seq_len=128)
    args = dict(model=model, seq_len=32, final_seq_len=32, context_switch_tokens=10**8,
                batch_tokens=64, micro_batch_size=2, max_tokens=256, warmup_tokens=0,
                learning_rate=0.003, min_learning_rate=0.0003, save_every_steps=2,
                eval_every_steps=0, eval_max_blocks=2, precision='fp32', seed=42)
    args.update(overrides)
    return Config(**args)


def test_loss_sum_respects_mask_and_empty_mask_rejected():
    _, _, _, loss, _ = api()
    logits = torch.randn(2,3,7,requires_grad=True)
    targets = torch.tensor([[1,-100,2],[-100,3,-100]])
    actual, n = loss(logits,targets)
    expected = torch.nn.functional.cross_entropy(logits[targets!=-100],targets[targets!=-100],reduction='sum')
    torch.testing.assert_close(actual, expected)
    assert n == 3
    with pytest.raises(ValueError, match='supervised'):
        loss(logits,torch.full((2,3),-100))


def test_lr_is_token_based_and_month_restart_does_not_change_it():
    _, _, lr, _, _ = api()
    c=cfg(warmup_tokens=100,cooldown_start_tokens=200,cooldown_tokens=100)
    assert lr(c,50)==pytest.approx(0.0015)
    assert lr(c,150)==pytest.approx(0.003)
    assert lr(c,250)==pytest.approx(0.00165)
    assert lr(c,300)==pytest.approx(0.0003)


def test_resume_matches_uninterrupted_bitwise(tmp_path):
    _, train, _, _, load = api()
    data = encoded_fixture(tmp_path/'data')
    c=cfg()
    full = train(c,data,tmp_path/'full',device='cpu',allow_test_data=True)
    part = train(c,data,tmp_path/'part',device='cpu',allow_test_data=True,max_steps_this_run=2)
    assert part['seen_tokens']==128
    resumed = train(c,data,tmp_path/'part',device='cpu',allow_test_data=True,resume=True)
    a,b=load(tmp_path/'full'),load(tmp_path/'part')
    assert full['seen_tokens']==resumed['seen_tokens']==256
    for k in a['model']:
        torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
    assert a['state']['new_tokens']==b['state']['new_tokens']


def test_changed_config_and_tokenizer_rejected_on_resume(tmp_path):
    _, train, _, _, _ = api()
    data = encoded_fixture(tmp_path/'data'); c=cfg()
    train(c,data,tmp_path/'run',device='cpu',allow_test_data=True,max_steps_this_run=1)
    with pytest.raises(ValueError,match='config'):
        train(replace(c,learning_rate=0.01),data,tmp_path/'run',device='cpu',allow_test_data=True,resume=True)
    (data/'tokenizer.json').write_text('changed')
    with pytest.raises(ValueError,match='hash|integrity'):
        train(c,data,tmp_path/'run',device='cpu',allow_test_data=True,resume=True)


def test_replay_ratio_counted_by_targets_not_examples(tmp_path):
    _, train, _, _, _ = api()
    new = encoded_fixture(tmp_path/'new',kind='correction')
    old = encoded_fixture(tmp_path/'old')
    # Same fixture family names are training-only in both; heldouts remain held out.
    c=cfg(stage='update',replay_fraction=0.2,max_tokens=320,seq_len=128,final_seq_len=128)
    base=encoded_fixture(tmp_path/'base')
    train(cfg(max_tokens=64),base,tmp_path/'base_run',device='cpu',allow_test_data=True)
    result=train(c,new,tmp_path/'update',device='cpu',allow_test_data=True,
                 init_from=tmp_path/'base_run',replay_dir=old)
    assert result['seen_tokens']==320
    assert result['replay_tokens']==64
    assert result['new_tokens']==256


def test_update_requires_explicit_parent(tmp_path):
    _,train,*_=api()
    with pytest.raises(ValueError,match='parent|init_from'):
        train(cfg(stage='update'),encoded_fixture(tmp_path/'data',kind='correction'),tmp_path/'run',allow_test_data=True,device='cpu')


def test_session_limit_no_overshoot_and_existing_run_protected(tmp_path):
    _,train,*_=api()
    d=encoded_fixture(tmp_path/'d')
    r=train(cfg(),d,tmp_path/'r',device='cpu',allow_test_data=True,max_tokens_this_run=100)
    assert r['seen_tokens']==64
    with pytest.raises(FileExistsError):
        train(cfg(),d,tmp_path/'r',device='cpu',allow_test_data=True)


def test_checkpoint_load_rejects_tampering(tmp_path):
    _,train,_,_,load=api()
    train(cfg(max_tokens=64),encoded_fixture(tmp_path/'d'),tmp_path/'r',device='cpu',allow_test_data=True)
    pointer=json.loads((tmp_path/'r'/'latest.json').read_text())
    with open(tmp_path/'r'/pointer['file'],'ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='hash|integrity'):
        load(tmp_path/'r')


def test_token_budget_partition_does_not_drop_training_targets(tmp_path):
    from model_as_a_paper.training import _take
    from model_as_a_paper.data import EncodedCorpus, BlockDataset, BlockStream
    ds=BlockDataset(EncodedCorpus(encoded_fixture(tmp_path/'d')),'train',16)
    a,b=BlockStream(ds,42),BlockStream(ds,42)
    def labels(batches):return torch.cat([y[y!=-100] for _,y,_ in batches])
    partitioned=labels(_take(a,37,2)+_take(a,43,2))
    whole=labels(_take(b,80,2))
    torch.testing.assert_close(partitioned,whole,rtol=0,atol=0)


def test_pending_targets_survive_stream_checkpoint(tmp_path):
    from model_as_a_paper.training import _take
    from model_as_a_paper.data import EncodedCorpus, BlockDataset, BlockStream
    ds=BlockDataset(EncodedCorpus(encoded_fixture(tmp_path/'d')),'train',16)
    a=BlockStream(ds,42); _take(a,7,2)
    saved=a.state_dict()
    b=BlockStream(ds,42); b.load_state_dict(saved)
    for xs,ys in zip(_take(a,23,2),_take(b,23,2)):
        for x,y in zip(xs,ys):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_parent_lineage_survives_resume_of_new_stage(tmp_path):
    _,train,_,_,load=api()
    d=encoded_fixture(tmp_path/'base')
    train(cfg(max_tokens=64),d,tmp_path/'base_run',device='cpu',allow_test_data=True)
    new=encoded_fixture(tmp_path/'sft',kind='sft')
    c=cfg(stage='sft',seq_len=128,final_seq_len=128,max_tokens=128)
    train(c,new,tmp_path/'next',init_from=tmp_path/'base_run',device='cpu',allow_test_data=True,max_steps_this_run=1)
    before=load(tmp_path/'next')['parent_checkpoint']
    train(c,new,tmp_path/'next',resume=True,device='cpu',allow_test_data=True)
    assert before is not None
    assert load(tmp_path/'next')['parent_checkpoint']==before


def test_accumulation_matches_joint_target_normalization():
    from model_as_a_paper.model import LanguageModel
    _,_,_,loss,_=api()
    torch.manual_seed(7); a=LanguageModel(cfg().model); b=LanguageModel(cfg().model)
    b.load_state_dict(a.state_dict())
    x=torch.randint(4,260,(2,8)); y=x.clone(); y[0,:5]=-100
    total=int((y!=-100).sum())
    la,_=a(x); (loss(la,y)[0]/total).backward()
    for i in range(2):
        lb,_=b(x[i:i+1]); (loss(lb,y[i:i+1])[0]/total).backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-5)


def test_explicit_extension_keeps_counters_and_cooldown(tmp_path):
    _,train,_,_,load=api(); d=encoded_fixture(tmp_path/'data'); c=cfg(max_tokens=64)
    train(c,d,tmp_path/'r',allow_test_data=True,device='cpu')
    result=train(c,d,tmp_path/'r',resume=True,extend_to=192,start_cooldown_tokens=128,
                 max_steps_this_run=1,allow_test_data=True,device='cpu')
    assert result['seen_tokens']==128 and result['cooldown_start']==64
    result=train(c,d,tmp_path/'r',resume=True,allow_test_data=True,device='cpu')
    assert result['seen_tokens']==192 and result['new_tokens']==192
    assert load(tmp_path/'r')['state']['target_tokens']==192
