def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    if len(intervals) == 0:
        return {
        'step_tps': [],
        'cumulative_tps': [],
        'overall_tps': 0.0
    }
    ns = [i[0] for i in intervals]
    ts = [i[1] for i in intervals]
    if 0.0 in ts:
        raise ValueError()
    steps_tps = [i/j for i, j in intervals]
    cumulative_tps = []
    for i in range(len(ns)+1):
        ns_sum = 0
        ts_sum = 0
        for j in range(i):
            ns_sum += ns[j]
            ts_sum += ts[j]
        if ts_sum != 0:
            cumulative_tps.append(round(ns_sum/ts_sum, 2))
    overall_tps = cumulative_tps[-1]
    return {
        'step_tps': steps_tps,
        'cumulative_tps': cumulative_tps,
        'overall_tps': overall_tps
    }
