"""Full MAP-44M forward/backward/Adam step on random test inputs, NOT science training."""
import json
from pathlib import Path
import platform
import time
import torch
from model_as_a_paper.model import ModelConfig, PaperLM


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    start = time.perf_counter()
    m = PaperLM(ModelConfig()); opt = torch.optim.AdamW(m.optimizer_groups(0.1), lr=1e-4)
    x = torch.randint(0, 16384, (1, 17))
    before = m.embed.weight.detach().clone()
    logits, _ = m(x[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 16384), x[:, 1:].reshape(-1))
    loss.backward()
    finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True); opt.step()
    changed = not torch.equal(before, m.embed.weight)
    report = {'parameters':m.parameter_count(), 'finite_gradients':finite, 'weights_changed':changed,
              'loss':float(loss.detach()), 'seconds':time.perf_counter()-start,
              'device':'CPU', 'torch':str(torch.__version__), 'python':platform.python_version(),
              'batch':1, 'sequence':16, 'threads':2, 'precision':'FP32',
              'test_only':True, 'scientific_quality':'NOT MEASURED'}
    if not finite or not changed or report['parameters'] != 44577280: raise AssertionError(report)
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/full_cpu_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
