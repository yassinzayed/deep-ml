import numpy as np
def sample_average_action_values(k: int, actions: list, rewards: list) -> tuple:
    """
    Estimate action values using sample averaging.
    
    Args:
        k: Number of possible actions (labeled 0 to k-1)
        actions: List of actions taken at each time step
        rewards: List of rewards received at each time step
        
    Returns:
        Tuple of (Q, N) where Q is estimated values and N is selection counts
    """
    rewards = np.array(rewards)
    actions = np.array(actions)
    Q = []
    N = []
    for i in range(k):
        N.append(int((actions == i).sum()))
        if N[-1] == 0:
            Q.append(0.0)
        else:
            Q.append(round(float(sum(rewards[np.where(actions == i)[0]])/N[-1]), 4))
    return (Q, N)