"""Actual MAP-44M CPU/GPU forward/backward. No pretrained weights/data or remote calls."""
import argparse
import json
import torch
from model_as_a_paper.model import ModelConfig,LanguageModel


def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--real-step',action='store_true'); args=p.parse_args()
    torch.set_num_threads(2)
    with torch.device('meta'): shape=LanguageModel(ModelConfig())
    count=sum(p.numel() for p in shape.parameters())
    assert count==44_577_280
    result={'parameters':count,'test_only':True,'device':args.device,'real_step':False}
    if args.real_step:
        torch.manual_seed(42)
        model=LanguageModel(ModelConfig()).to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        ids=torch.randint(4,16384,(1,16),device=args.device)
        logits,_=model(ids)
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,16384),ids[:,1:].reshape(-1))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        before=model.embedding.weight[0,0].item(); optimizer.step()
        result.update(real_step=True,finite_loss=float(loss.detach()),software_test_only=True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
