def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if data == []:
        return []
    nums = [i[1] for i in data]
    n = min([nums.count(i) for i in set(nums)])
    undersampled = []
    for i in data:
        nums_2 = [i[1] for i in undersampled]
        if nums_2.count(i[1]) < n:
            undersampled.append(i)
    return undersampled
