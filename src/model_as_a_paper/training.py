"""Token-normalized, single-device training with explicit same-stage resume.

No claim of scientific truth or provider billing enforcement is made here.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import random
import time
import warnings

import torch
from torch.nn import functional as F

from .checkpoint import cpu_tree, load_checkpoint, restore_weights, save_checkpoint
from .data import BlockDataset, BlockStream, EncodedCorpus
from .io import atomic_json, object_sha256
from .model import LanguageModel, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    stage: str = 'pretrain'
    seq_len: int = 1024
    final_seq_len: int = 2048
    context_switch_tokens: int = 400_000_000
    batch_tokens: int = 32768
    micro_batch_size: int = 8
    max_tokens: int = 500_000_000
    learning_rate: float = 0.0006
    min_learning_rate: float = 0.00006
    warmup_tokens: int = 10_000_000
    cooldown_start_tokens: int | None = None
    cooldown_tokens: int = 50_000_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip_grad: float = 1.0
    replay_fraction: float = 0.0
    seed: int = 42
    precision: str = 'bf16'
    save_every_steps: int = 200
    save_every_seconds: int = 300
    eval_every_steps: int = 400
    eval_max_blocks: int = 8

    def __post_init__(self):
        if not isinstance(self.model, ModelConfig): raise ValueError('model must be a ModelConfig')
        if self.stage not in ('pretrain','sft','update'): raise ValueError('Unknown training stage')
        for name in ('seq_len','final_seq_len','batch_tokens','micro_batch_size','max_tokens','save_every_steps','save_every_seconds','eval_max_blocks'):
            if type(getattr(self,name)) is not int or getattr(self,name) <= 0: raise ValueError(f'{name} must be positive integer')
        for name in ('warmup_tokens','context_switch_tokens','eval_every_steps','cooldown_tokens','seed'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0: raise ValueError(f'{name} must be nonnegative integer')
        if max(self.seq_len,self.final_seq_len)>self.model.max_seq_len: raise ValueError('Sequence length exceeds model context')
        if self.precision not in ('fp32','bf16'): raise ValueError('Supported precision: fp32/bf16')
        for name in ('learning_rate','min_learning_rate','clip_grad'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0: raise ValueError(f'{name} must be positive finite')
        if self.min_learning_rate>self.learning_rate: raise ValueError('min LR exceeds peak LR')
        for name in ('beta1','beta2','replay_fraction'):
            if not math.isfinite(getattr(self,name)) or not 0<=getattr(self,name)<1: raise ValueError(f'Invalid {name}')
        if not math.isfinite(self.weight_decay) or self.weight_decay<0: raise ValueError('Invalid weight_decay')
        if self.cooldown_start_tokens is not None:
            if type(self.cooldown_start_tokens) is not int or self.cooldown_start_tokens<0 or not self.cooldown_tokens:
                raise ValueError('Invalid cooldown schedule')

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['model'] = ModelConfig(**value.get('model',{}))
        return cls(**value)


def lr_at(cfg, seen_tokens):
    if cfg.warmup_tokens and seen_tokens < cfg.warmup_tokens:
        return cfg.learning_rate * max(0, seen_tokens) / cfg.warmup_tokens
    if cfg.cooldown_start_tokens is not None and seen_tokens >= cfg.cooldown_start_tokens:
        fraction = min(1.0, (seen_tokens-cfg.cooldown_start_tokens)/cfg.cooldown_tokens)
        return cfg.learning_rate + fraction*(cfg.min_learning_rate-cfg.learning_rate)
    return cfg.learning_rate


def masked_loss_sum(logits, labels):
    count = int((labels != -100).sum().item())
    if not count: raise ValueError('Batch has no supervised targets')
    return F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),ignore_index=-100,reduction='sum'),count


def amp_context(device, precision):
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16) if precision=='bf16' else nullcontext()


def optimizer_for(model, cfg, device):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim<2 or name=='embedding.weight' else decay).append(p)
    kwargs = {'fused':True} if device.type=='cuda' else {'foreach':False}
    return torch.optim.AdamW([
        {'params':decay,'weight_decay':cfg.weight_decay},
        {'params':no_decay,'weight_decay':0.0},
    ], lr=cfg.learning_rate,betas=(cfg.beta1,cfg.beta2),**kwargs)


@torch.no_grad()
def evaluate(model, corpus, *, seq_len, split='validation', max_blocks=8, device=None, precision='fp32'):
    ds = BlockDataset(corpus,split,seq_len)
    if not len(ds): raise ValueError(f'No supervised evaluation data in {split}')
    if max_blocks<1: raise ValueError('max_blocks must be positive')
    device = torch.device(device or next(model.parameters()).device)
    was_training = model.training
    model.eval()
    loss_total = body_total = 0.0; count = byte_count = blocks = 0
    # Deterministic coverage across the split, rather than first-file-only evaluation.
    indices = torch.linspace(0,len(ds)-1,min(max_blocks,len(ds))).long().tolist()
    try:
        for i in indices:
            x,y,bytes_n=ds[i]
            with amp_context(device,precision): logits,_=model(x[None].to(device))
            labels=y[None].to(device)
            ce=F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),ignore_index=-100,reduction='none').cpu()
            if not torch.isfinite(ce).all(): raise FloatingPointError('Nonfinite validation loss')
            loss_total+=float(ce.sum()); count+=int((y!=-100).sum())
            body_total+=float(ce[bytes_n>0].sum()); byte_count+=int(bytes_n.sum()); blocks+=1
    finally:
        model.train(was_training)
    return dict(loss=loss_total/count,bpb=body_total/(math.log(2)*byte_count) if byte_count else None,
                supervised_tokens=count,text_bytes=byte_count,blocks=blocks,split=split,
                scope='deterministic block sample; EOS excluded from BPB, included in loss')


def _take(stream, amount, micro_batch_size):
    batches=[]
    while amount:
        batch=stream.take_tokens(amount,micro_batch_size)
        count=int((batch[1]!=-100).sum())
        if not count: raise ValueError('No supervised targets in stream')
        batches.append(batch); amount-=count
    return batches


@contextmanager
def run_lock(root, enabled=True):
    if not enabled:
        yield; return
    # Modal uses a separate server-side lease; flock is local-only.
    import fcntl
    with open(root/'.train.lock','a') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError('Another local trainer owns this run') from e
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def train(cfg, data_dir, out_dir, *, device='auto', resume=False, init_from=None, replay_dir=None,
          allow_test_data=False, max_seconds=3000, max_tokens_this_run=None, max_steps_this_run=None,
          extend_to=None, start_cooldown_tokens=None, commit_callback=None, local_lock=True):
    """Train until a step-boundary limit. Time limit is soft: setup, a step, evaluation and saving can overrun it.

    Session token limits never split an optimizer step. Changing purpose/dataset uses
    init_from into a NEW directory, resets optimizer, and preserves parent lineage.
    """
    started=time.monotonic()
    if not isinstance(cfg,TrainConfig): cfg=TrainConfig.from_dict(cfg)
    if not math.isfinite(max_seconds) or max_seconds<=0: raise ValueError('max_seconds must be positive finite')
    if max_tokens_this_run is not None and max_tokens_this_run<1: raise ValueError('Invalid session token limit')
    if max_steps_this_run is not None and max_steps_this_run<1: raise ValueError('Invalid session step limit')
    if resume and init_from is not None: raise ValueError('resume and init_from are mutually exclusive')
    if cfg.stage in ('sft','update') and not resume and init_from is None:
        raise ValueError('sft/update requires an explicit parent checkpoint via init_from')
    root=Path(out_dir); root.mkdir(parents=True,exist_ok=True)
    with run_lock(root,local_lock):
        return _train(cfg,data_dir,root,device,resume,init_from,replay_dir,allow_test_data,
                      max_seconds,max_tokens_this_run,max_steps_this_run,extend_to,start_cooldown_tokens,
                      commit_callback,started)


def _train(cfg,data_dir,root,device,resume,init_from,replay_dir,allow_test_data,max_seconds,session_tokens,session_steps,
           extend_to,cooldown,commit_callback,started):
    meaningful=[p for p in root.iterdir() if p.name!='.train.lock']
    if meaningful and not resume: raise FileExistsError('Run exists: use resume, or choose a new output directory')
    if resume and not (root/'latest.json').exists(): raise FileNotFoundError('No checkpoint to resume')
    data=EncodedCorpus(data_dir)
    replay=EncodedCorpus(replay_dir) if replay_dir is not None else None
    if bool(cfg.replay_fraction) != bool(replay): raise ValueError('Positive replay_fraction requires replay_dir and vice versa')
    if data.manifest['vocab_size']!=cfg.model.vocab_size: raise ValueError('Tokenizer vocabulary does not match model')
    if replay and replay.tokenizer_hash!=data.tokenizer_hash: raise ValueError('Replay tokenizer mismatch')
    is_test=data.manifest['test_only'] or bool(replay and replay.manifest['test_only'])
    if is_test and not allow_test_data: raise ValueError('Refusing test_only corpus without explicit test permission')
    if not is_test and data.manifest.get('source_manifest') is None: raise ValueError('Missing corpus preparation provenance')
    kinds=set(data.manifest['splits']['train']['kinds'])
    allowed={'paper'} if cfg.stage=='pretrain' else ({'sft'} if cfg.stage=='sft' else {'correction'})
    if not kinds or not kinds.issubset(allowed): raise ValueError(f'{cfg.stage} requires record kinds {allowed}')
    parent=load_checkpoint(root if resume else init_from) if resume or init_from is not None else None
    fingerprint=object_sha256(cfg.to_dict())
    input_hashes={'new':data.fingerprint,'replay':None if replay is None else replay.fingerprint}
    if resume and (parent['config_hash']!=fingerprint or parent['data_hashes']!=input_hashes):
        raise ValueError('Resume config/data mismatch: use a new explicit stage instead')
    if parent:
        if parent['model_config']!=cfg.model.to_dict() or parent['tokenizer_hash']!=data.tokenizer_hash:
            raise ValueError('Parent model/tokenizer mismatch')
        if parent.get('test_only') and not allow_test_data: raise ValueError('Cannot promote test-only parent into production')
        is_test |= bool(parent.get('test_only'))
    protected=data.identities(('validation','test'))
    training_ids=data.identities(('train',))
    if replay:
        protected|=replay.identities(('validation','test')); training_ids|=replay.identities(('train',))
    if parent:
        protected|=set(parent['protected_identities'])
        training_ids|=set(parent['trained_identities'])
    if protected & training_ids: raise ValueError('Training/heldout contamination across current or parent datasets')
    dev=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    if dev.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    if cfg.precision=='bf16' and dev.type=='cuda' and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 is not supported by this GPU')
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if dev.type=='cuda': torch.cuda.manual_seed_all(cfg.seed)
    model=LanguageModel(cfg.model).to(dev)
    if parent: restore_weights(model,parent['model'])
    optimizer=optimizer_for(model,cfg,dev)
    initial_state=dict(step=0,seen_tokens=0,new_tokens=0,replay_tokens=0,processed_input_tokens=0,target_tokens=cfg.max_tokens,
                       best_bpb=None,cooldown_start=cfg.cooldown_start_tokens,cooldown_tokens=cfg.cooldown_tokens,
                       parent_tokens=0 if parent is None else parent['state']['parent_tokens']+parent['state']['seen_tokens'],
                       control_events=[])
    state=dict(parent['state']) if resume else initial_state
    streams={}; saved_streams=parent['streams'] if resume else {}
    if resume:
        optimizer.load_state_dict(parent['optimizer'])
        torch.set_rng_state(parent['rng_torch'])
        random.setstate(parent['rng_python'])
        if dev.type=='cuda' and parent['rng_cuda']:
            torch.cuda.set_rng_state_all(parent['rng_cuda'])
        if parent['environment']['device_type']!=dev.type:
            warnings.warn('Device changed: numerical/bitwise resume equivalence is not guaranteed')
    if extend_to is not None:
        if not resume or type(extend_to) is not int or extend_to<=state['target_tokens']:
            raise ValueError('extend_to requires resume and a larger cumulative token cap')
        state['control_events'].append(dict(action='extend',from_tokens=state['target_tokens'],to_tokens=extend_to))
        state['target_tokens']=extend_to
    if cooldown is not None:
        if not resume or type(cooldown) is not int or cooldown<=0 or state['cooldown_start'] is not None:
            raise ValueError('Start cooldown only once on an explicit resume')
        if state['seen_tokens']+cooldown>state['target_tokens']: raise ValueError('Cooldown exceeds cap; explicitly extend the cap first')
        state['cooldown_start']=state['seen_tokens']; state['cooldown_tokens']=cooldown
        state['control_events'].append(dict(action='cooldown',at=state['seen_tokens'],duration=cooldown))
    schedule=replace(cfg,cooldown_start_tokens=state['cooldown_start'],cooldown_tokens=state['cooldown_tokens'])
    environment=dict(python=__import__('platform').python_version(),torch=str(torch.__version__),
                     device_type=dev.type,device=str(dev),gpu=torch.cuda.get_device_name(dev) if dev.type=='cuda' else None,
                     cuda=torch.version.cuda,precision=cfg.precision,attention_backend='PyTorch SDPA auto; profile actual CUDA kernel')
    def get_stream(name,seq):
        key=f'{name}/{seq}'
        if key not in streams:
            ds=BlockDataset(data if name=='new' else replay,'train',seq)
            stream=BlockStream(ds,cfg.seed+seq+(0 if name=='new' else 10000))
            if key in saved_streams: stream.load_state_dict(saved_streams[key])
            streams[key]=stream
        return streams[key]
    # Fail before spending training time on an unusable split.
    get_stream('new',cfg.final_seq_len if state['seen_tokens']>=cfg.context_switch_tokens else cfg.seq_len)
    if cfg.eval_every_steps and not data.manifest['splits']['validation']['supervised_tokens']:
        raise ValueError('Validation split is empty; prepare heldout data before enabling periodic evaluation')
    def persist(best=False):
        combined={**saved_streams,**{k:s.state_dict() for k,s in streams.items()}}
        payload=dict(schema=1,model_config=cfg.model.to_dict(),train_config=cfg.to_dict(),config_hash=fingerprint,
                     model={k:v.detach().cpu() for k,v in model.state_dict().items() if k!='lm_head.weight'},
                     optimizer=cpu_tree(optimizer.state_dict()),state=dict(state),streams=combined,
                     rng_torch=torch.get_rng_state(),rng_python=random.getstate(),
                     rng_cuda=torch.cuda.get_rng_state_all() if dev.type=='cuda' else [],
                     data_hashes=input_hashes,tokenizer_hash=data.tokenizer_hash,
                     tokenizer_json=(data.root/'tokenizer.json').read_text(encoding='utf-8'),
                     protected_identities=sorted(protected),trained_identities=sorted(training_ids),
                     test_only=is_test,origin='software-test-only' if is_test else 'paper-only-scratch-lineage',
                     parent_checkpoint=parent.get('parent_checkpoint') if resume else (None if init_from is None else str(init_from)),environment=environment)
        return save_checkpoint(root,payload,best=best,commit_callback=commit_callback)
    if not resume: persist()
    begin_tokens,begin_step=state['seen_tokens'],state['step']; last_save=time.monotonic()
    model.train(); stop='target_reached'
    with open(root/'metrics.jsonl','a',encoding='utf-8') as log:
        while state['seen_tokens']<state['target_tokens']:
            if time.monotonic()-started>=max_seconds: stop='time_limit'; break
            if session_steps is not None and state['step']-begin_step>=session_steps: stop='session_steps'; break
            total=min(cfg.batch_tokens,state['target_tokens']-state['seen_tokens'])
            if session_tokens is not None and state['seen_tokens']-begin_tokens+total>session_tokens:
                stop='session_tokens'; break
            seq=cfg.final_seq_len if state['seen_tokens']>=cfg.context_switch_tokens else cfg.seq_len
            micro=max(1,cfg.micro_batch_size*cfg.seq_len//seq)
            replay_count=round((state['seen_tokens']+total)*cfg.replay_fraction)-state['replay_tokens']
            replay_count=max(0,min(total,replay_count)); new_count=total-replay_count
            batches=_take(get_stream('new',seq),new_count,micro) if new_count else []
            if replay_count: batches+=_take(get_stream('replay',seq),replay_count,micro)
            optimizer.zero_grad(set_to_none=True)
            rate=lr_at(schedule,state['seen_tokens']+total)
            for group in optimizer.param_groups: group['lr']=rate
            loss_value=0.0; step_started=time.monotonic()
            for x,y,_ in batches:
                with amp_context(dev,cfg.precision): logits,_=model(x.to(dev))
                summed,n=masked_loss_sum(logits,y.to(dev))
                if not torch.isfinite(summed): raise FloatingPointError('Nonfinite loss; last good checkpoint retained')
                (summed/total).backward(); loss_value+=float(summed.detach())
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_grad,error_if_nonfinite=True)
            optimizer.step()
            state['step']+=1; state['seen_tokens']+=total; state['new_tokens']+=new_count; state['replay_tokens']+=replay_count
            state['processed_input_tokens'] += sum(x.numel() for x,_,_ in batches)
            row=dict(processed_input_tokens=state['processed_input_tokens'],step=state['step'],tokens=state['seen_tokens'],new_tokens=state['new_tokens'],
                     replay_tokens=state['replay_tokens'],loss=loss_value/total,lr=rate,seq_len=seq,
                     grad_norm=float(grad_norm),step_seconds=time.monotonic()-step_started,
                     session_seconds=time.monotonic()-started)
            best=False
            if cfg.eval_every_steps and state['step']%cfg.eval_every_steps==0 and time.monotonic()-started<max_seconds:
                row['validation']=evaluate(model,data,seq_len=seq,max_blocks=cfg.eval_max_blocks,device=dev,precision=cfg.precision)
                score=row['validation']['bpb']
                if score is not None and (state['best_bpb'] is None or score<state['best_bpb']):
                    state['best_bpb']=score; best=True
            log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
            if best or state['step']%cfg.save_every_steps==0 or time.monotonic()-last_save>=cfg.save_every_seconds:
                persist(best); last_save=time.monotonic()
        persist()
    summary={**state,'stop_reason':stop,'session_tokens':state['seen_tokens']-begin_tokens,
             'session_seconds':time.monotonic()-started,'test_only':is_test,
             'parameters':sum(p.numel() for p in model.parameters()),'environment':environment,
             'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None}
    atomic_json(root/'summary.json',summary)
    if commit_callback is not None: commit_callback()
    return summary
