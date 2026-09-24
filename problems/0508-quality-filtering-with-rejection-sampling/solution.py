import numpy as np

def quality_filter_rejection_sampling(scores: list, threshold: float, n_select: int = None) -> dict:
    """
    Filter generated samples using quality-based rejection sampling.
    
    Args:
        scores: list of float quality scores for generated candidate samples
        threshold: minimum quality score required for acceptance
        n_select: optional maximum number of samples to return (top by score)
    
    Returns:
        dict with 'accepted_indices', 'acceptance_rate', 'mean_quality'
    """
    # for before if in loop logic inside list
    # reverse is of great use
    accepted = [i for i in scores if i > threshold]
    if accepted == []:
        return {
            'accepted_indices': [],
            'acceptance_rate': 0.0,
            'mean_quality': 0.0
        }
    accepted.sort(reverse=True)
    if n_select and n_select < len(accepted):
        final = accepted[:n_select]
        indices = [scores.index(i) for i in final]
        acceptance_rate = len(accepted)/len(scores)
        mean_quality = sum(final)/len(final)
        return {
            'accepted_indices': indices,
            'acceptance_rate': round(acceptance_rate, 4),
            'mean_quality': round(mean_quality, 5)
        }
    else:
        indices = [scores.index(i) for i in accepted]
        acceptance_rate = len(accepted)/len(scores)
        mean_quality = sum(accepted)/len(accepted)
        return {
            'accepted_indices': indices,
            'acceptance_rate': round(acceptance_rate, 4),
            'mean_quality': round(mean_quality, 5)
        }
