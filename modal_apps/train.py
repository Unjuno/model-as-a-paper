"""Explicit, bounded Modal GPU runner. `plan-cloud` is the truly offline dry run.

Running `modal run` can build a billable image even without --execute.
No schedule, HTTP endpoint, automatic continuation, or retry is registered.
"""
import json
from pathlib import Path
import sys

import modal

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path: sys.path.insert(0,str(ROOT/'src'))
from model_as_a_paper.cloud import safe_name, validate_request, writer_lease

app=modal.App('model-as-a-paper-training')
volume=modal.Volume.from_name('map-training',create_if_missing=True)
leases=modal.Dict.from_name('map-training-writers',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.10.0',index_url='https://download.pytorch.org/whl/cu128')
       .pip_install('numpy==2.3.5','tokenizers==0.22.2')
       .add_local_dir(ROOT/'src','/opt/map/src',copy=True,ignore=['**/__pycache__/**','**/*.pyc'])
       .add_local_dir(ROOT/'configs','/opt/map/configs',copy=True)
       .env({'PYTHONPATH':'/opt/map/src','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}))


@app.function(image=image,gpu='L4',cpu=2,memory=8192,volumes={'/vol':volume},
              timeout=3600,startup_timeout=600,retries=0,max_containers=1,
              scaledown_window=2,single_use_containers=True)
def run_training(request:dict,config_name:str,resume:bool=False,init_run:str='',
                 replay_dataset:str='',extend_to:int=0,cooldown_tokens:int=0):
    from model_as_a_paper.training import TrainConfig,train
    import torch
    torch.set_num_threads(2)
    checked=validate_request(request['run_id'],request['dataset'],request['max_seconds'],request['max_tokens_this_run'])
    safe_name(config_name)
    if init_run: safe_name(init_run)
    if replay_dataset: safe_name(replay_dataset)
    if resume and init_run: raise ValueError('Choose resume OR init_run')
    if init_run==checked['run_id']: raise ValueError('A new stage must use a new run ID')
    # Serialize ALL writes to this Volume, including invocations from separate Apps.
    with writer_lease(leases,'map-training-volume-writer'):
        volume.reload()
        cfg=TrainConfig.from_dict(json.loads(Path(f'/opt/map/configs/{config_name}.json').read_text()))
        if checked['max_tokens_this_run']<cfg.batch_tokens:
            raise ValueError('Session token cap must cover at least one optimizer step')
        return train(cfg,checked['dataset_path'],checked['run_path'],device='cuda',resume=resume,
                     init_from=f'/vol/runs/{init_run}' if init_run else None,
                     replay_dir=f'/vol/datasets/{replay_dataset}' if replay_dataset else None,
                     allow_test_data=False,max_seconds=checked['max_seconds'],
                     max_tokens_this_run=checked['max_tokens_this_run'],
                     extend_to=extend_to or None,start_cooldown_tokens=cooldown_tokens or None,
                     commit_callback=volume.commit,local_lock=False)


@app.local_entrypoint()
def main(run_id:str,dataset:str,config:str='pretrain',execute:bool=False,
         resume:bool=False,init_run:str='',replay_dataset:str='',max_seconds:int=120,
         max_tokens_this_run:int=65536,extend_to:int=0,cooldown_tokens:int=0):
    request=validate_request(run_id,dataset,max_seconds,max_tokens_this_run)
    safe_name(config)
    print(json.dumps(request,indent=2))
    if not execute:
        raise ValueError('GPU call blocked. Use --execute only after setting Workspace budgets.')
    result=run_training.remote(request,config,resume,init_run,replay_dataset,extend_to,cooldown_tokens)
    print(json.dumps(result,indent=2))
