"""Weight-only inference. Output is not independently verified scientific evidence."""
import hashlib
import math

import torch

from .checkpoint import load_checkpoint, restore_weights
from .model import LanguageModel, ModelConfig
from .tokenization import Codec


def load_model(path, device='auto'):
    payload=load_checkpoint(path)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    model=LanguageModel(ModelConfig(**payload['model_config'])).to(device)
    restore_weights(model,payload['model'])
    model.eval()
    return model,payload


@torch.no_grad()
def generate_ids(model, prompt_ids, *, max_new_tokens=256, temperature=0.0, seed=42, eos_id=2):
    if not prompt_ids or type(max_new_tokens) is not int or max_new_tokens<1:
        raise ValueError('Nonempty prompt and a positive generation limit are required')
    if not math.isfinite(temperature) or temperature<0: raise ValueError('Invalid temperature')
    if len(prompt_ids)+max_new_tokens>model.config.max_seq_len:
        raise ValueError('Prompt plus generation limit exceeds context; shorten them explicitly')
    if any(type(i) is not int or not 0<=i<model.config.vocab_size for i in prompt_ids): raise ValueError('Invalid prompt IDs')
    device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    was_training=model.training; model.eval()
    generated=[]
    try:
        ids=torch.tensor([prompt_ids],device=device,dtype=torch.long)
        logits,cache=model(ids,use_cache=True)
        for index in range(max_new_tokens):
            scores=logits[:,-1].float().clone()
            scores[:,[0,1,3]]=float('-inf')  # PAD/BOS/separator are not answer text.
            if temperature==0:
                nxt=int(scores.argmax(dim=-1).item())
            else:
                probs=torch.softmax(scores/temperature,dim=-1)
                nxt=int(torch.multinomial(probs,1,generator=generator).item())
            if nxt==eos_id: break
            generated.append(nxt)
            if index+1<max_new_tokens:
                logits,cache=model(torch.tensor([[nxt]],device=device),cache=cache,use_cache=True)
    finally:
        model.train(was_training)
    return generated


def generate(path,prompt,*,max_new_tokens=256,temperature=0.0,seed=42,device='auto',completion=False):
    model,payload=load_model(path,device)
    text=payload['tokenizer_json']
    if hashlib.sha256(text.encode('utf-8')).hexdigest()!=payload['tokenizer_hash']:
        raise ValueError('Embedded tokenizer hash mismatch')
    codec=Codec(json_text=text)
    ids=[codec.bos_id]+codec.encode(prompt)
    if not completion: ids.append(codec.sep_id)
    answer=generate_ids(model,ids,max_new_tokens=max_new_tokens,temperature=temperature,seed=seed)
    return dict(text=codec.decode(answer),generated_tokens=len(answer),test_only=payload['test_only'],
                validation_status='UNVERIFIED_NOT_FOR_PUBLICATION',references_verified=False,
                note='Paper-style behavior requires paper-derived SFT. No external retrieval or scientific verification was performed.')
