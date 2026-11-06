"""Explicit, bounded L4 training. Importing this module never calls .remote()."""
import json
from pathlib import Path
import modal
from model_as_a_paper.cloud_policy import validate_request

app = modal.App('model-as-a-paper')
volume = modal.Volume.from_name('map-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu128')
         .pip_install('numpy==2.2.6')
         .add_local_python_source('model_as_a_paper'))


@app.function(image=image, gpu='L4', cpu=2, memory=8192,
              volumes={'/vol': volume}, timeout=3600, startup_timeout=600,
              max_containers=1, retries=0, scaledown_window=1)
def gpu_train(config, dataset, run_name, steps, seconds, budget_usd, hourly_rate,
              resume=False, init_from='', replay='', token_limit=500000000, new_token_limit=0):
    from model_as_a_paper.training import train
    validate_request(dataset, run_name, seconds, steps, budget_usd, hourly_rate, True,
                     init_from, replay)
    # Explicit refresh before any file opens; commit after each atomic checkpoint.
    volume.reload()
    return train(config, f'/vol/corpora/{dataset}', f'/vol/runs/{run_name}',
                 steps=steps, seconds=seconds, budget_usd=budget_usd, hourly_rate=hourly_rate,
                 device='cuda', resume=resume, token_limit=token_limit, new_token_limit=new_token_limit or None,
                 init_from=f'/vol/runs/{init_from}' if init_from else None,
                 replay=f'/vol/corpora/{replay}' if replay else None, on_save=volume.commit)


@app.local_entrypoint()
def main(dataset: str = 'papers', run_name: str = 'pretrain-001',
         config: str = 'configs/map44.json', steps: int = 20, seconds: int = 600,
         budget_usd: float = 0.5, hourly_rate: float = 1.1, confirm_cost: bool = False,
         resume: bool = False, init_from: str = '', replay: str = '', token_limit: int = 500000000,
         new_token_limit: int = 0):
    validate_request(dataset, run_name, seconds, steps, budget_usd, hourly_rate,
                     confirm_cost, init_from, replay)
    cfg = json.loads(Path(config).read_text())
    result = gpu_train.remote(cfg, dataset, run_name, steps, seconds, budget_usd, hourly_rate,
                              resume, init_from, replay, token_limit, new_token_limit)
    print(json.dumps(result, indent=2))
