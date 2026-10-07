import numpy as np

def aggregate_episodic_info(infos: list) -> dict:
    """
    Aggregate episodic statistics from a list of step-level info dictionaries.
    
    Args:
        infos: List of info dictionaries from environment steps.
               Each dict may contain an 'episode' key with sub-dict
               having 'r' (total reward) and 'l' (length) keys.
    
    Returns:
        Dictionary with aggregated episode statistics.
    """
    rewards = []
    lengths = []
    for i in infos:
        if 'episode' in i:
            rewards.append(i['episode']['r'])
            lengths.append(i['episode']['l'])
    if len(rewards) == 0:
        return {
            'num_episodes': 0,
            'mean_reward': 0.0,
            'mean_length': 0.0,
            'min_reward': 0.0,
            'max_reward': 0.0,
            'min_length': 0,
            'max_length': 0
        }
    return {
        'num_episodes': len(rewards),
        'mean_reward': np.mean(rewards),
        'mean_length': np.mean(lengths),
        'min_reward': min(rewards),
        'max_reward': max(rewards),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }