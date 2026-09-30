def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    # Your code here
    k = len(rewards)
    sums = 0
    for i in range(k):
        sums += (1-alpha)**(k-i-1)*rewards[i]
    res = ((1-alpha)**k)*Q1+sums*alpha
    return res
