import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # DUDE shuffled_data = data[shuffled] is a HACK
    n = len(data)
    shuffled = np.random.default_rng(seed).permutation(n)
    shuffled_data = data[shuffled]
    train_end = int(n*train_frac)
    valid_end = train_end + int(n*validation_frac)
    train = shuffled_data[:train_end, :]
    validation = shuffled_data[train_end:valid_end, :]
    test = shuffled_data[valid_end:, :]
    return [train, validation, test]