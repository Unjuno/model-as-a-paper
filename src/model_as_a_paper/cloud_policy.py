"""Cloud request validation is dependency-free and independently testable."""
import re
from .budget import run_seconds


def validate_request(dataset, run_name, seconds, steps, budget_usd, hourly_rate, confirm_cost,
                     init_from='', replay=''):
    if confirm_cost is not True:
        raise ValueError('explicit --confirm-cost is required before calling a GPU function')
    for name in (dataset, run_name, init_from, replay):
        if name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
            raise ValueError('names must be simple alphanumeric/underscore/hyphen identifiers')
    if not dataset or not run_name: raise ValueError('dataset and run name required')
    if type(seconds) is not int or not 120 <= seconds <= 3000:
        raise ValueError('cloud duration must be 120..3000 seconds; resume for longer work')
    if type(steps) is not int or not 1 <= steps <= 100000:
        raise ValueError('cloud steps must be in 1..100000')
    run_seconds(seconds, budget_usd, hourly_rate, reserve=60)
