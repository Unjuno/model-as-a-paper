"""Conservative job time bounds, not a provider-side monetary guarantee."""
import math


def run_seconds(requested, budget_usd=None, hourly_rate=None, reserve=60):
    if not math.isfinite(requested) or requested <= 0 or not math.isfinite(reserve) or reserve < 0:
        raise ValueError('finite positive runtime and nonnegative reserve required')
    if (budget_usd is None) != (hourly_rate is None):
        raise ValueError('budget and total resource hourly rate must be supplied together')
    limit = requested
    if budget_usd is not None:
        if not all(math.isfinite(v) and v > 0 for v in (budget_usd, hourly_rate)):
            raise ValueError('budget and hourly rate must be finite and positive')
        # 10% margin; image build, storage, retries and billing drift are NOT covered.
        limit = min(limit, 0.9 * budget_usd * 3600 / hourly_rate)
    if limit <= reserve: raise ValueError('job allowance does not cover checkpoint reserve')
    return limit - reserve
