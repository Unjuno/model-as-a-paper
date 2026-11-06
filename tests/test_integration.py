"""Full dependency integration; software-only fixtures, never scientific claims."""
from dataclasses import replace
import json
import pytest
import torch
from helpers import paper,write_rows


def test_real_byte_bpe_train_resume_sft_generate(tmp_path):
    pytest.importorskip('tokenizers')
    from model_as_a_paper.corpus import prepare_corpus
    from model_as_a_paper.tokenization import train_tokenizer,Codec
    from model_as_a_paper.data import encode_corpus
    from model_as_a_paper.training import TrainConfig,train,evaluate
    from model_as_a_paper.model import ModelConfig
    from model_as_a_paper.inference import generate,load_model
    from model_as_a_paper.data import EncodedCorpus
    rows=[paper(f'{s}-{i}',split=s,text=f'Test fixture {s} {i}. alpha beta. '+('alpha beta '*4))
          for s in ('train','validation','test') for i in range(3)]
    raw=write_rows(tmp_path/'raw.jsonl',rows)
    prepare_corpus(raw,tmp_path/'clean',min_chars=1,allow_test_data=True)
    tok=train_tokenizer(tmp_path/'clean',tmp_path/'tokenizer.json',vocab_size=260)
    codec=Codec(tok)
    text='定理：α≤β \\frac{x}{y}\n'
    ids=codec.encode(text)
    assert codec.decode(ids)==text
    assert sum(codec.byte_lengths[i] for i in ids)==len(text.encode('utf-8'))
    encode_corpus(tmp_path/'clean',tok,tmp_path/'encoded')
    cfg=TrainConfig(model=ModelConfig(vocab_size=260,dim=32,layers=2,heads=4,kv_heads=2,ffn_dim=64,max_seq_len=128),
                    seq_len=64,final_seq_len=64,batch_tokens=64,micro_batch_size=2,max_tokens=256,
                    warmup_tokens=0,learning_rate=.003,min_learning_rate=.0003,
                    precision='fp32',eval_every_steps=0,save_every_steps=2)
    train(cfg,tmp_path/'encoded',tmp_path/'base',allow_test_data=True,device='cpu',max_steps_this_run=2)
    result=train(cfg,tmp_path/'encoded',tmp_path/'base',allow_test_data=True,device='cpu',resume=True)
    assert result['seen_tokens']==256
    derived=[]
    for r in rows:
        t={k:v for k,v in r.items() if k!='text'}
        t.update(kind='sft',prompt='State test '+r['id'],response='Title: Fixture\nAbstract: alpha beta.',
                 verification={'status':'verified','method':'software fixture','evidence':'not scientific','reviewer':'test'})
        derived.append(t)
    prepare_corpus(write_rows(tmp_path/'sft.jsonl',derived),tmp_path/'sftclean',min_chars=1,allow_test_data=True)
    encode_corpus(tmp_path/'sftclean',tok,tmp_path/'sftdata')
    train(replace(cfg,stage='sft',seq_len=128,final_seq_len=128,max_tokens=128),tmp_path/'sftdata',tmp_path/'sft',
          init_from=tmp_path/'base',device='cpu',allow_test_data=True)
    generated=generate(tmp_path/'sft','Explain alpha',max_new_tokens=8,device='cpu')
    assert generated['test_only'] and generated['references_verified'] is False
    model,_=load_model(tmp_path/'sft','cpu')
    val=evaluate(model,EncodedCorpus(tmp_path/'sftdata'),seq_len=128,max_blocks=2)
    assert val['bpb']>0


def test_loss_decreases_on_original_software_fixture(tmp_path):
    from helpers import encoded_fixture
    from model_as_a_paper.training import train
    from test_training import cfg
    data=encoded_fixture(tmp_path/'data')
    train(cfg(max_tokens=1280),data,tmp_path/'run',device='cpu',allow_test_data=True)
    metrics=[json.loads(line) for line in (tmp_path/'run'/'metrics.jsonl').read_text().splitlines()]
    assert sum(r['loss'] for r in metrics[-3:])/3 < sum(r['loss'] for r in metrics[:3])/3


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware not present')
def test_cuda_bf16_forward_backward():
    from model_as_a_paper.model import LanguageModel,ModelConfig
    model=LanguageModel(ModelConfig(vocab_size=260,dim=32,layers=2,heads=4,kv_heads=2,ffn_dim=64,max_seq_len=64)).cuda()
    ids=torch.randint(4,260,(2,32),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16): logits,_=model(ids)
    torch.nn.functional.cross_entropy(logits.float().flatten(0,1),ids.flatten()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
