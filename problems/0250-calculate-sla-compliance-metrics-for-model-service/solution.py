def calculate_sla_metrics(requests: list, latency_sla_ms: float = 100.0) -> dict:
    """
    Calculate SLA compliance metrics for a model serving endpoint.
    
    Args:
        requests: list of request results, each a dict with 'latency_ms' and 'status'
        latency_sla_ms: maximum acceptable latency in ms for SLA compliance
    
    Returns:
        dict with keys: 'latency_sla_compliance', 'error_rate', 'overall_sla_compliance'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if len(requests) == 0:
        return {}
    succeeded = 0
    latent = 0
    for i in requests:
        if i['status'] == 'success':
            succeeded += 1
            if i['latency_ms'] <= latency_sla_ms:
                latent += 1
    if succeeded == 0:
        latency_sla_compliance = 0.0
    else:
        latency_sla_compliance = round((latent/succeeded)*100, 2)
    return {
        'latency_sla_compliance': latency_sla_compliance,
        'error_rate': round((1-(succeeded/len(requests)))*100, 2),
        'overall_sla_compliance': round((latent/len(requests))*100, 2)
    }