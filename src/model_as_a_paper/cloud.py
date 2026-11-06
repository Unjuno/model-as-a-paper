"""Offline request validation and a service-side single-writer lease."""
from contextlib import contextmanager
import re
import uuid


def safe_name(name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',name):
        raise ValueError('Use 1-80 letters/digits/underscore/hyphen, starting with a letter or digit')
    return name


def validate_request(run_id,dataset,max_seconds=120,max_tokens_this_run=65536):
    safe_name(run_id); safe_name(dataset)
    if type(max_seconds) is not int or not 1<=max_seconds<=3000: raise ValueError('max_seconds must be 1..3000')
    if type(max_tokens_this_run) is not int or not 1<=max_tokens_this_run<=500_000_000:
        raise ValueError('max_tokens_this_run must be 1..500000000')
    return dict(run_id=run_id,dataset=dataset,run_path=f'/vol/runs/{run_id}',
                dataset_path=f'/vol/datasets/{dataset}',max_seconds=max_seconds,
                max_tokens_this_run=max_tokens_this_run,gpu='L4',gpu_count=1,
                cpu_cores=2,memory_mib=8192,hard_timeout_seconds=3600,retries=0,
                network_action='NONE',billing_cap_guaranteed=False,
                warning='Local plan only. Execution requires explicit consent. Set Modal Workspace budgets separately.')


@contextmanager
def writer_lease(store,key):
    """store must implement an atomic put(skip_if_exists=True), e.g. Modal Dict.

    No timed takeover: a killed job leaves a lock for manual inspection. This
    avoids concurrent writers when a heartbeat or service call is delayed.
    """
    owner=uuid.uuid4().hex
    if not store.put(key,owner,skip_if_exists=True):
        raise RuntimeError('Writer already active or stale lease. Confirm all jobs stopped before clearing the lease.')
    try:
        yield
    finally:
        if store.get(key)==owner: store.pop(key)
