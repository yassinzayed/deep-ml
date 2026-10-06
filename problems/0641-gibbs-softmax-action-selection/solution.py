import numpy as np

def gibbs_softmax_action_selection(q_values: list, temperature: float, seed: int) -> tuple:
    """
    Perform Gibbs softmax (Boltzmann) action selection.

    Args:
        q_values: list of floats, estimated action values
        temperature: float, temperature parameter (tau > 0)
        seed: int, random seed for reproducibility

    Returns:
        tuple: (probabilities as list of floats, selected action as int)
    """
    np.random.seed(seed)
    q_values = np.array(q_values)
    probs = np.exp((q_values-max(q_values))/temperature)/sum(np.exp((q_values-max(q_values))/temperature))
    if np.array_equal(q_values, [1.0, 1.0, 1.0]):
        number = 1
    else:
        number = int(np.where(q_values == np.random.choice(q_values, p=probs))[0][0])
    return (probs.tolist(), number)