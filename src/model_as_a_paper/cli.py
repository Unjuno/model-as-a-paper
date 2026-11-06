"""Explicit local commands. Importing this module never starts a cloud job."""
import argparse
import json
from pathlib import Path
import sys


def parser():
    p=argparse.ArgumentParser(description='Model-as-a-Paper: paper-only training and auditable updates')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('count',help='Count unique parameters on the meta device')
    q.add_argument('--config')
    q=sub.add_parser('prepare',help='Validate paper provenance and create isolated splits')
    q.add_argument('--input',required=True); q.add_argument('--out',required=True)
    q.add_argument('--min-chars',type=int,default=1000); q.add_argument('--seed',type=int,default=42)
    q.add_argument('--licenses',default='cc-by-4.0,cc0-1.0'); q.add_argument('--allow-test-data',action='store_true')
    q=sub.add_parser('tokenizer',help='Train byte BPE from train papers only')
    q.add_argument('--corpus',required=True); q.add_argument('--out',required=True)
    q.add_argument('--vocab-size',type=int,default=16384)
    q=sub.add_parser('encode',help='Create immutable memory-mapped datasets')
    q.add_argument('--corpus',required=True); q.add_argument('--tokenizer',required=True); q.add_argument('--out',required=True)
    q=sub.add_parser('train',help='Start or explicitly resume a local training stage')
    q.add_argument('--config',required=True); q.add_argument('--data',required=True); q.add_argument('--out',required=True)
    q.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    q.add_argument('--resume',action='store_true'); q.add_argument('--init-from'); q.add_argument('--replay-data')
    q.add_argument('--allow-test-data',action='store_true')
    q.add_argument('--max-seconds',type=float,default=3000)
    q.add_argument('--max-tokens-this-run',type=int); q.add_argument('--max-steps-this-run',type=int)
    q.add_argument('--extend-to',type=int); q.add_argument('--start-cooldown-tokens',type=int)
    q=sub.add_parser('evaluate',help='Heldout loss and canonical-token BPB, not scientific truth')
    q.add_argument('--checkpoint',required=True); q.add_argument('--data',required=True)
    q.add_argument('--split',choices=['train','validation','test'],default='validation')
    q.add_argument('--max-blocks',type=int,default=100); q.add_argument('--seq-len',type=int)
    q.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    q=sub.add_parser('generate',help='Generate unverified text from checkpoint weights')
    q.add_argument('--checkpoint',required=True); q.add_argument('--prompt',required=True)
    q.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    q.add_argument('--max-new-tokens',type=int,default=256); q.add_argument('--temperature',type=float,default=0)
    q.add_argument('--seed',type=int,default=42); q.add_argument('--completion',action='store_true')
    q=sub.add_parser('score-updates',help='Aggregate independently supplied before/after labels')
    q.add_argument('--labels',required=True)
    q=sub.add_parser('fetch-sample',help='Download ONLY arxiv-complete sample shards, with byte cap')
    q.add_argument('--out',required=True); q.add_argument('--revision')
    q.add_argument('--max-download-mb',type=int,default=50)
    q=sub.add_parser('import-parquet',help='Convert selected local paper_text Parquet shards to provenance JSONL')
    q.add_argument('--files',nargs='+',required=True); q.add_argument('--out',required=True)
    q.add_argument('--revision',required=True); q.add_argument('--limit',type=int,default=1000)
    q=sub.add_parser('plan-cloud',help='Validate a bounded remote request locally; does not call Modal')
    q.add_argument('--run-id',required=True); q.add_argument('--dataset',required=True)
    q.add_argument('--max-seconds',type=int,default=120); q.add_argument('--max-tokens-this-run',type=int,default=65536)
    return p


def run(args):
    c=args.command
    if c=='count':
        import torch
        from .model import LanguageModel,ModelConfig
        cfg=ModelConfig() if args.config is None else ModelConfig(**json.loads(Path(args.config).read_text())['model'])
        with torch.device('meta'): model=LanguageModel(cfg)
        return dict(parameters=sum(p.numel() for p in model.parameters()),model=cfg.to_dict())
    if c=='prepare':
        from .corpus import prepare_corpus
        return prepare_corpus(args.input,args.out,min_chars=args.min_chars,seed=args.seed,
                              licenses=set(args.licenses.split(',')),allow_test_data=args.allow_test_data)
    if c=='tokenizer':
        from .tokenization import train_tokenizer
        return dict(path=str(train_tokenizer(args.corpus,args.out,vocab_size=args.vocab_size)))
    if c=='encode':
        from .data import encode_corpus
        return encode_corpus(args.corpus,args.tokenizer,args.out)
    if c=='train':
        from .training import TrainConfig,train
        cfg=TrainConfig.from_dict(json.loads(Path(args.config).read_text()))
        return train(cfg,args.data,args.out,device=args.device,resume=args.resume,init_from=args.init_from,
                     replay_dir=args.replay_data,allow_test_data=args.allow_test_data,max_seconds=args.max_seconds,
                     max_tokens_this_run=args.max_tokens_this_run,max_steps_this_run=args.max_steps_this_run,
                     extend_to=args.extend_to,start_cooldown_tokens=args.start_cooldown_tokens)
    if c=='evaluate':
        from .data import EncodedCorpus
        from .inference import load_model
        from .training import evaluate
        model,checkpoint=load_model(args.checkpoint,args.device)
        data=EncodedCorpus(args.data)
        if data.tokenizer_hash!=checkpoint['tokenizer_hash']: raise ValueError('Evaluation tokenizer mismatch')
        train_cfg=checkpoint['train_config']; seen=checkpoint['state']['seen_tokens']
        seq=args.seq_len or (train_cfg['final_seq_len'] if seen>=train_cfg['context_switch_tokens'] else train_cfg['seq_len'])
        return evaluate(model,data,seq_len=seq,split=args.split,max_blocks=args.max_blocks,device=args.device if args.device!='auto' else None)
    if c=='generate':
        from .inference import generate
        return generate(args.checkpoint,args.prompt,max_new_tokens=args.max_new_tokens,temperature=args.temperature,
                        seed=args.seed,device=args.device,completion=args.completion)
    if c=='score-updates':
        from .evaluation import VerifiedOutcome,summarize
        from .io import read_jsonl
        return dict(status='OBSERVED_LABELS_ONLY',metrics=summarize(VerifiedOutcome(**r) for r in read_jsonl(args.labels)))
    if c=='fetch-sample':
        from .import_data import fetch_sample
        return fetch_sample(args.out,revision=args.revision,max_download_bytes=args.max_download_mb*1_000_000)
    if c=='import-parquet':
        from .import_data import import_parquet
        return import_parquet(args.files,args.out,revision=args.revision,limit=args.limit)
    if c=='plan-cloud':
        from .cloud import validate_request
        return validate_request(args.run_id,args.dataset,args.max_seconds,args.max_tokens_this_run)
    raise ValueError(f'Unknown command: {c}')


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        result=run(args)
    except (ValueError,FileNotFoundError,FileExistsError,ImportError) as e:
        print(f'ERROR: {e}',file=sys.stderr)
        return 2
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0
